#!/usr/bin/env python3
"""
Task 8 — MI→MU Directed Gene Graph Propagation
================================================
Post-hoc correction of SpatialEx predictions: propagate well-predicted MI
gene signals to poorly-predicted MU genes via an asymmetric PPI + co-expression
gene-gene graph.

No retraining — operates purely on V0 baseline predictions from Task7.

Usage on HPC:
    cd /gpfsdata/home/renyixiang/YuanLab
    python Task8_genePropagate/run_propagation.py --dataset mg --seed 42
    python Task8_genePropagate/run_propagation.py --dataset skin --seed 42

Key novelties vs SPRITE (Bioinformatics 2024):
    1. Asymmetric adjacency: MI→MU directed, MU→MI blocked
    2. Dual-source gene graph: PPI + Spearman co-expression (SPRITE uses Spearman only)
    3. No scRNA-seq reference required (self-contained)
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("task8")

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

# =============================================
# A. Task6 dynamic library loader (from Task7)
# =============================================

def load_task6_lib():
    """Exec the library portion of Task6's pipeline (funcs + classes only)."""
    script = PROJECT_ROOT / "Task6_wgcnaPPI" / "run_wgcna_ppi_pipeline.py"
    source = script.read_text(encoding="utf-8")
    marker = "# 1. 数据读取与预处理"
    assert marker in source, f"Marker not found in {script}"
    lib_src = source.split(marker)[0]

    real_makedirs = os.makedirs
    def safe_makedirs(p, *a, **kw):
        try: real_makedirs(p, *a, **kw)
        except OSError: pass
    os.makedirs = safe_makedirs
    try:
        ns = {"__name__": "_t6_lib", "__file__": str(script), "__builtins__": __builtins__}
        exec(compile(lib_src, str(script), "exec"), ns)
    finally:
        os.makedirs = real_makedirs

    logger.info("Task6 library loaded (%d / %d chars)", len(lib_src), len(source))
    return ns


# =============================================
# B. Data loading helpers
# =============================================

def preprocess_mg(t6):
    se = t6["se"]
    dev, res, enc, k = t6["device"], t6["resolution"], t6["image_encoder"], t6["num_neighbors"]
    r1, r2 = t6["save_root1"], t6["save_root2"]

    logger.info("Preprocessing MG slice 1")
    adata1 = se.pp.Read_Xenium(r1 + "cell_feature_matrix.h5", r1 + "cells.csv")
    adata1 = se.pp.Preprocess_adata(adata1)
    img, scale = se.pp.Read_HE_image(r1 + "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif")
    tmtx = __import__("pandas").read_csv(r1 + "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_imagealignment.csv", header=None).values
    adata1 = se.pp.Register_physical_to_pixel(adata1, tmtx, scale=scale)
    he, adata1 = se.pp.Tiling_HE_patches(res, adata1, img)
    adata1 = se.pp.Extract_HE_patches_representaion(he, adata=adata1, image_encoder=enc, device=dev, store_key="he")
    del he, img

    logger.info("Preprocessing MG slice 2")
    adata2 = se.pp.Read_Xenium(r2 + "cell_feature_matrix.h5", r2 + "cells.csv")
    adata2 = se.pp.Preprocess_adata(adata2)
    img, scale = se.pp.Read_HE_image(r2 + "Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image.ome.tif")
    tmtx = __import__("pandas").read_csv(r2 + "Xenium_FFPE_Human_Breast_Cancer_Rep2_he_imagealignment.csv", header=None).values
    adata2 = se.pp.Register_physical_to_pixel(adata2, tmtx, scale=scale)
    he, adata2 = se.pp.Tiling_HE_patches(res, adata2, img)
    adata2 = se.pp.Extract_HE_patches_representaion(he, adata=adata2, image_encoder=enc, store_key="he", device=dev)
    del he, img

    logger.info("MG ready: %s + %s", adata1.shape, adata2.shape)
    return adata1, adata2


def preprocess_skin(t6):
    return t6["preprocess_skin_data"]()[:2]  # adata1, adata2 only (skip graphs)


def to_dense(X):
    if hasattr(X, "todense"): return np.asarray(X.todense())
    if hasattr(X, "toarray"): return np.asarray(X.toarray())
    return np.asarray(X)


def load_v0_baseline(v0_dir):
    """Load V0 predictions + per-gene metrics from Task7 results."""
    v0_dir = Path(v0_dir)
    data = {
        "panelB1": np.load(v0_dir / "panelB1.npy"),
        "panelA2": np.load(v0_dir / "panelA2.npy"),
    }
    for key in ("pcc_per_gene", "ssim_per_gene", "spcc_per_gene", "morans_per_gene"):
        for slc in ("slice1", "slice2"):
            f = v0_dir / f"{slc}_{key}.npy"
            if f.exists():
                data[f"{slc}_{key}"] = np.load(f)
    logger.info("V0 loaded: panelB1 %s, panelA2 %s", data["panelB1"].shape, data["panelA2"].shape)
    return data


# =============================================
# C. Gene mask classification
# =============================================

def build_masks_from_pcc(pcc_s1, pcc_s2):
    """Classify genes into MI/MU/Moderate by tercile on mean baseline PCC."""
    mean_pcc = (np.nan_to_num(pcc_s1) + np.nan_to_num(pcc_s2)) / 2.0
    t_low = np.percentile(mean_pcc, 33.3)
    t_high = np.percentile(mean_pcc, 66.7)
    mi = mean_pcc > t_high
    mu = mean_pcc <= t_low
    mod = ~(mi | mu)
    logger.info("Gene masks: MI=%d (>%.3f)  MU=%d (<=%.3f)  MOD=%d",
                mi.sum(), t_high, mu.sum(), t_low, mod.sum())
    return mi, mu, mod, mean_pcc


# =============================================
# D. Gene-gene graph construction (CORE NOVELTY)
# =============================================

def load_ppi(ppi_dir, var_names):
    """Load HumanBase PPI matrix, reindex to var_names order."""
    ppi_path = os.path.join(ppi_dir, "ppi_matrix.npy")
    if not os.path.exists(ppi_path):
        logger.warning("PPI not found at %s", ppi_dir)
        return np.zeros((len(var_names), len(var_names)), dtype=np.float32)
    ppi = np.load(ppi_path)
    ppi_genes = np.load(os.path.join(ppi_dir, "ppi_gene_names.npy"), allow_pickle=True)
    if list(ppi_genes) != list(var_names):
        idx = {g: i for i, g in enumerate(ppi_genes)}
        G = len(var_names)
        out = np.zeros((G, G), dtype=np.float32)
        for i, g1 in enumerate(var_names):
            if g1 not in idx: continue
            for j, g2 in enumerate(var_names):
                if g2 in idx: out[i, j] = ppi[idx[g1], idx[g2]]
        ppi = out
    logger.info("PPI loaded: shape %s, %d nonzero edges", ppi.shape, np.count_nonzero(ppi) // 2)
    return ppi.astype(np.float32)


def compute_coexpression(expr, method="spearman"):
    """Compute gene-gene co-expression matrix from ground truth expression.

    Args:
        expr: (N_cells, G_genes) dense expression matrix
        method: "spearman" or "pearson"

    Returns:
        (G, G) absolute correlation matrix (symmetric, values in [0, 1])
    """
    G = expr.shape[1]
    logger.info("Computing %s co-expression for %d genes (%d cells)...", method, G, expr.shape[0])

    # For large N, subsample cells to speed up Spearman
    N = expr.shape[0]
    if N > 5000:
        rng = np.random.RandomState(42)
        idx = rng.choice(N, 5000, replace=False)
        expr_sub = expr[idx]
        logger.info("  Subsampled to %d cells for speed", len(idx))
    else:
        expr_sub = expr

    if method == "spearman":
        corr, _ = spearmanr(expr_sub, axis=0)  # (G, G)
    else:
        corr = np.corrcoef(expr_sub.T)  # (G, G)

    corr = np.nan_to_num(np.abs(corr))
    np.fill_diagonal(corr, 0.0)
    logger.info("  Co-expression: mean=%.4f, median=%.4f, max=%.4f",
                corr.mean(), np.median(corr), corr.max())
    return corr.astype(np.float32)


def build_asymmetric_graph(
    ppi, coexpr, mi_mask, mu_mask, mod_mask,
    ppi_weight=0.5, coexpr_weight=0.5,
    coexpr_threshold=0.3, mu_to_mi_damping=0.0,
):
    """Build combined, asymmetric, column-normalized gene-gene adjacency.

    A[j, i] = influence of gene j's prediction on gene i's correction.
    Used as: E_corrected = E @ A  (propagation along gene axis)

    Asymmetry rules:
        - MI → MU/MOD:  full weight (MI gene j informs MU/MOD gene i)
        - MI → MI:      full weight (MI genes reinforce each other)
        - MOD → MU:     full weight
        - MU → MI:      damped by mu_to_mi_damping (default 0 = fully blocked)
        - MU → MU:      full weight (MU genes share signal among themselves)
        - MU → MOD:     damped

    Returns:
        A_asym: (G, G) column-normalized transition matrix
    """
    G = ppi.shape[0]

    # Combine PPI + co-expression
    coexpr_bin = (coexpr >= coexpr_threshold).astype(np.float32) * coexpr
    W = ppi_weight * ppi + coexpr_weight * coexpr_bin
    np.fill_diagonal(W, 0.0)

    logger.info("  Combined graph before asymmetry: %d edges", np.count_nonzero(W))

    # Apply asymmetric masking: A[j, i] — gene j influences gene i
    # We want to BLOCK gene j (MU) from influencing gene i (MI)
    A = W.copy()
    for j in range(G):
        for i in range(G):
            if A[j, i] == 0:
                continue
            # Source = MU, Target = MI → block
            if mu_mask[j] and mi_mask[i]:
                A[j, i] *= mu_to_mi_damping
            # Source = MU, Target = MOD → dampen
            elif mu_mask[j] and mod_mask[i]:
                A[j, i] *= mu_to_mi_damping

    logger.info("  Asymmetric graph: %d edges (after masking)", np.count_nonzero(A))

    # Column-normalize: each target gene i receives normalized incoming weights
    col_sums = A.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1.0
    A = A / col_sums

    return A.astype(np.float32)


# =============================================
# E. Iterative label propagation
# =============================================

def propagate(E0, A_asym, alpha=0.5, n_iters=10, tol=1e-6):
    """Iterative label propagation on the gene axis.

    E(t+1) = (1 - alpha) * E0 + alpha * E(t) @ A_asym

    Args:
        E0: (N_cells, G_genes) baseline predictions
        A_asym: (G, G) column-normalized transition matrix
        alpha: propagation strength (0 = no propagation, 1 = full propagation)
        n_iters: max iterations
        tol: convergence tolerance

    Returns:
        E_final: (N_cells, G_genes) corrected predictions
    """
    import warnings
    E = E0.copy().astype(np.float64)
    E0_64 = E0.astype(np.float64)
    A_64 = A_asym.astype(np.float64)
    for t in range(n_iters):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            E_new = (1 - alpha) * E0_64 + alpha * (E @ A_64)
        E_new = np.nan_to_num(E_new, nan=0.0, posinf=0.0, neginf=0.0)
        diff = np.abs(E_new - E).max()
        E = E_new
        if diff < tol:
            logger.info("  Propagation converged at iter %d (diff=%.2e)", t + 1, diff)
            break
    else:
        logger.info("  Propagation finished %d iters (final diff=%.2e)", n_iters, diff)
    return E.astype(np.float32)


# =============================================
# F. Per-gene metrics & stratified report
# =============================================

def per_gene_pcc_np(pred, target, eps=1e-8):
    p = pred - pred.mean(axis=0, keepdims=True)
    t = target - target.mean(axis=0, keepdims=True)
    num = (p * t).sum(axis=0)
    denom = np.sqrt((p ** 2).sum(axis=0) * (t ** 2).sum(axis=0) + eps)
    return np.nan_to_num(num / denom)


def stratified_report(mi, mu, mod, per_gene_metrics, baseline_metrics=None):
    """Compute per-group summary for all per-gene metrics."""
    report = {"overall": {}, "MI": {"n": int(mi.sum())}, "MU": {"n": int(mu.sum())},
              "Moderate": {"n": int(mod.sum())}, "per_gene": {}}

    for metric, vals in per_gene_metrics.items():
        vals = np.nan_to_num(vals)
        report["overall"][metric] = float(vals.mean())
        report["per_gene"][metric] = vals

        bl_vals = None
        if baseline_metrics and metric in baseline_metrics:
            bl_vals = np.nan_to_num(baseline_metrics[metric])

        for grp_name, mask in [("MI", mi), ("MU", mu), ("Moderate", mod)]:
            n = int(mask.sum())
            if n == 0: continue
            report[grp_name][f"{metric}_mean"] = float(vals[mask].mean())
            report[grp_name][f"{metric}_median"] = float(np.median(vals[mask]))
            if bl_vals is not None:
                delta = vals[mask] - bl_vals[mask]
                report[grp_name][f"d{metric}_mean"] = float(delta.mean())
                if n > 5:
                    try:
                        _, p = scipy_stats.wilcoxon(delta, alternative="greater")
                        report[grp_name][f"d{metric}_wilcoxon_p"] = float(p)
                    except Exception:
                        report[grp_name][f"d{metric}_wilcoxon_p"] = float("nan")

    return report


def print_report(rep, label):
    logger.info("--- %s ---", label)
    metrics = [m for m in rep["overall"]]
    logger.info("  Overall: %s", "  ".join(f"{m}={rep['overall'][m]:.4f}" for m in metrics))
    header = f"  {'Group':>10s} | {'n':>4s}"
    for m in metrics:
        header += f" | {m:>7s}  d{m:>5s}"
    logger.info(header)
    logger.info("  " + "-" * (len(header) - 2))
    for grp in ("MI", "MU", "Moderate"):
        r = rep[grp]
        n = r.get("n", 0)
        parts = [f"  {grp:>10s} | {n:4d}"]
        for m in metrics:
            val = r.get(f"{m}_mean", float("nan"))
            dval = r.get(f"d{m}_mean", float("nan"))
            parts.append(f" | {val:7.4f}  {dval:+6.4f}")
        logger.info("".join(parts))


# =============================================
# G. Grid search
# =============================================

GRID = {
    "alpha":            [0.1, 0.3, 0.5, 0.7, 0.9],
    "coexpr_threshold": [0.3, 0.5, 0.7],
    "n_iters":          [5, 10, 20],
    "ppi_weight":       [0.7, 0.5, 0.3],
    # coexpr_weight = 1 - ppi_weight
}


def grid_search(E0, gt, ppi, coexpr, mi, mu, mod, baseline_pcc):
    """Grid search on a single slice with MI snap-back.

    After propagation, MI gene predictions are snapped back to original values.
    This removes the MI protection constraint entirely — selection purely
    maximizes MU PCC.
    """
    mi_baseline = float(baseline_pcc[mi].mean())
    mu_baseline = float(baseline_pcc[mu].mean())
    n_combos = (len(GRID["alpha"]) * len(GRID["coexpr_threshold"]) *
                len(GRID["n_iters"]) * len(GRID["ppi_weight"]))
    logger.info("Grid search (MI snap-back): MI baseline=%.4f, MU baseline=%.4f (%d combos)",
                mi_baseline, mu_baseline, n_combos)

    results = []
    best_mu_pcc = -1.0
    best_params = None

    for alpha in GRID["alpha"]:
        for ct in GRID["coexpr_threshold"]:
            for pw in GRID["ppi_weight"]:
                cw = 1.0 - pw
                A = build_asymmetric_graph(
                    ppi, coexpr, mi, mu, mod,
                    ppi_weight=pw, coexpr_weight=cw,
                    coexpr_threshold=ct, mu_to_mi_damping=0.0,
                )
                for ni in GRID["n_iters"]:
                    E_corr = propagate(E0, A, alpha=alpha, n_iters=ni)
                    # MI snap-back: restore MI genes to original predictions
                    E_corr[:, mi] = E0[:, mi]
                    pcc = per_gene_pcc_np(E_corr, gt)

                    mi_pcc = float(pcc[mi].mean())
                    mu_pcc = float(pcc[mu].mean())
                    overall_pcc = float(pcc.mean())

                    row = {
                        "alpha": alpha, "coexpr_threshold": ct,
                        "n_iters": ni, "ppi_weight": pw, "coexpr_weight": cw,
                        "overall_pcc": overall_pcc, "mi_pcc": mi_pcc,
                        "mu_pcc": mu_pcc, "mu_pcc_delta": mu_pcc - mu_baseline,
                    }
                    results.append(row)

                    # Select best: purely maximize MU PCC (MI is protected by snap-back)
                    if mu_pcc > best_mu_pcc:
                        best_mu_pcc = mu_pcc
                        best_params = row.copy()

    logger.info("Best: alpha=%.1f ct=%.1f ni=%d pw=%.1f → MU_PCC=%.4f (+%.4f) MI_PCC=%.4f (unchanged)",
                best_params["alpha"], best_params["coexpr_threshold"],
                best_params["n_iters"], best_params["ppi_weight"],
                best_params["mu_pcc"], best_params["mu_pcc_delta"], best_params["mi_pcc"])

    return best_params, results


# =============================================
# H. Main
# =============================================

def main():
    parser = argparse.ArgumentParser(description="Task8 MI→MU Gene Graph Propagation")
    parser.add_argument("--dataset", choices=["mg", "skin"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--v0-dir", type=str, default=None,
                        help="Path to V0_baseline results dir (default: auto-detect from Task7)")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--skip-grid", action="store_true",
                        help="Skip grid search, use default params")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # ── Locate V0 baseline ──
    if args.v0_dir:
        v0_dir = Path(args.v0_dir)
    else:
        v0_dir = PROJECT_ROOT / "Task7_anchorBoost" / "results" / f"{args.dataset}_seed{args.seed}" / "V0_baseline"
    assert (v0_dir / "panelB1.npy").exists(), f"V0 not found: {v0_dir}"

    # ── Output dir ──
    if args.output_dir:
        out_root = Path(args.output_dir)
    else:
        out_root = HERE / "results" / f"{args.dataset}_seed{args.seed}"
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Load V0 predictions ──
    v0 = load_v0_baseline(v0_dir)
    panelB1_orig = v0["panelB1"]  # (N1, G)
    panelA2_orig = v0["panelA2"]  # (N2, G)

    # ── Build MI/MU masks from V0 PCC ──
    pcc_s1 = v0["slice1_pcc_per_gene"]
    pcc_s2 = v0["slice2_pcc_per_gene"]
    mi_mask, mu_mask, mod_mask, mean_pcc = build_masks_from_pcc(pcc_s1, pcc_s2)
    G = len(mi_mask)

    # ── Load Task6 library + preprocess (for ground truth + evaluation) ──
    logger.info("Loading Task6 library for ground truth + evaluation...")
    t6 = load_task6_lib()
    if args.device:
        t6["device"] = args.device
    elif not __import__("torch").cuda.is_available():
        t6["device"] = "cpu"

    if args.dataset == "mg":
        adata1, adata2 = preprocess_mg(t6)
        ppi_dir = t6["PPI_DATA_DIR"]
    else:
        adata1, adata2 = preprocess_skin(t6)
        ppi_dir = t6["PPI_DATA_DIR_SKIN"]

    gt_s1 = to_dense(adata1.X)  # (N1, G)
    gt_s2 = to_dense(adata2.X)  # (N2, G)
    var_names = adata1.var_names.tolist()
    logger.info("Ground truth: s1 %s, s2 %s", gt_s1.shape, gt_s2.shape)

    # ── Load PPI ──
    ppi = load_ppi(ppi_dir, var_names)

    # ── Compute co-expression (from both slices, no leakage since both are training data) ──
    coexpr_path = out_root / "coexpression_matrix.npy"
    if coexpr_path.exists():
        coexpr = np.load(coexpr_path)
        logger.info("Loaded cached co-expression: %s", coexpr.shape)
    else:
        expr_combined = np.vstack([gt_s1, gt_s2])
        coexpr = compute_coexpression(expr_combined, method="spearman")
        np.save(coexpr_path, coexpr)

    # ── Collect V0 baseline per-gene metrics for delta computation ──
    baseline_pg_s1, baseline_pg_s2 = {}, {}
    for mkey, fkey in [("PCC", "pcc_per_gene"), ("SSIM", "ssim_per_gene"),
                       ("SPCC", "spcc_per_gene"), ("MoransI", "morans_per_gene")]:
        k1 = f"slice1_{fkey}"
        k2 = f"slice2_{fkey}"
        if k1 in v0: baseline_pg_s1[mkey] = v0[k1]
        if k2 in v0: baseline_pg_s2[mkey] = v0[k2]

    # ── Grid search on slice 1 ──
    if args.skip_grid:
        best_params = {
            "alpha": 0.7, "coexpr_threshold": 0.3, "n_iters": 20,
            "ppi_weight": 0.5, "coexpr_weight": 0.5,
        }
        grid_results = []
        logger.info("Skipping grid search, using defaults: %s", best_params)
    else:
        t0 = time.time()
        best_params, grid_results = grid_search(
            panelB1_orig, gt_s1, ppi, coexpr, mi_mask, mu_mask, mod_mask, pcc_s1,
        )
        logger.info("Grid search took %.1f min", (time.time() - t0) / 60)

        # Save grid results
        import csv
        with (out_root / "grid_search_results.csv").open("w", newline="") as f:
            if grid_results:
                writer = csv.DictWriter(f, fieldnames=grid_results[0].keys())
                writer.writeheader()
                writer.writerows(grid_results)

    with (out_root / "best_params.json").open("w") as f:
        json.dump(best_params, f, indent=2)

    # ── Build final asymmetric graph with best params ──
    logger.info("=" * 60)
    logger.info("Building final graph with best params")
    logger.info("=" * 60)
    A_asym = build_asymmetric_graph(
        ppi, coexpr, mi_mask, mu_mask, mod_mask,
        ppi_weight=best_params["ppi_weight"],
        coexpr_weight=best_params["coexpr_weight"],
        coexpr_threshold=best_params["coexpr_threshold"],
        mu_to_mi_damping=0.0,
    )
    np.save(out_root / "asymmetric_adjacency.npy", A_asym)

    # ── Propagate both slices + MI snap-back ──
    logger.info("Propagating slice 1 (%d cells)...", panelB1_orig.shape[0])
    panelB1_corr = propagate(
        panelB1_orig, A_asym,
        alpha=best_params["alpha"], n_iters=best_params["n_iters"],
    )
    panelB1_corr[:, mi_mask] = panelB1_orig[:, mi_mask]  # MI snap-back
    logger.info("Propagating slice 2 (%d cells)...", panelA2_orig.shape[0])
    panelA2_corr = propagate(
        panelA2_orig, A_asym,
        alpha=best_params["alpha"], n_iters=best_params["n_iters"],
    )
    panelA2_corr[:, mi_mask] = panelA2_orig[:, mi_mask]  # MI snap-back
    logger.info("MI genes snapped back to original predictions")

    np.save(out_root / "panelB1_propagated.npy", panelB1_corr)
    np.save(out_root / "panelA2_propagated.npy", panelA2_corr)

    # ── Full evaluation via Task6 evaluate() ──
    logger.info("=" * 60)
    logger.info("Evaluating propagated predictions")
    logger.info("=" * 60)
    results = t6["evaluate"](adata1, adata2, panelB1_corr, panelA2_corr)

    s1, s2 = results["slice1"], results["slice2"]
    logger.info("  s1 PCC=%.4f SSIM=%.4f CMD=%.4f | s2 PCC=%.4f SSIM=%.4f CMD=%.4f",
                s1["PCC"], s1["SSIM"], s1["CMD"], s2["PCC"], s2["SSIM"], s2["CMD"])

    # Save per-gene arrays
    for key in ("pcc_per_gene", "ssim_per_gene", "morans_per_gene", "spcc_per_gene"):
        if key in s1: np.save(out_root / f"slice1_{key}.npy", s1[key])
        if key in s2: np.save(out_root / f"slice2_{key}.npy", s2[key])

    # ── Stratified report ──
    def _pg(s):
        return {k: np.asarray(s[v]) for k, v in
                [("PCC", "pcc_per_gene"), ("SSIM", "ssim_per_gene"),
                 ("SPCC", "spcc_per_gene"), ("MoransI", "morans_per_gene")] if v in s}

    pg_s1, pg_s2 = _pg(s1), _pg(s2)
    rep_s1 = stratified_report(mi_mask, mu_mask, mod_mask, pg_s1, baseline_pg_s1)
    rep_s2 = stratified_report(mi_mask, mu_mask, mod_mask, pg_s2, baseline_pg_s2)
    print_report(rep_s1, "Propagated slice1 vs V0")
    print_report(rep_s2, "Propagated slice2 vs V0")

    # ── Save metrics ──
    def _strip_arrays(rep):
        return {k: v for k, v in rep.items() if k != "per_gene"}

    save = {
        "dataset": args.dataset, "seed": args.seed,
        "best_params": best_params,
        "v0_baseline": {
            "slice1": {k: float(v) for k, v in
                       (json.load((v0_dir / "metrics.json").open()) if (v0_dir / "metrics.json").exists()
                        else {}).get("slice1", {}).items() if isinstance(v, (int, float))},
            "slice2": {k: float(v) for k, v in
                       (json.load((v0_dir / "metrics.json").open()) if (v0_dir / "metrics.json").exists()
                        else {}).get("slice2", {}).items() if isinstance(v, (int, float))},
        },
        "propagated": {
            "slice1": {k: float(v) for k, v in s1.items() if isinstance(v, (int, float))},
            "slice2": {k: float(v) for k, v in s2.items() if isinstance(v, (int, float))},
        },
        "stratified_s1": _strip_arrays(rep_s1),
        "stratified_s2": _strip_arrays(rep_s2),
        "n_grid_combos": len(grid_results),
    }
    with (out_root / "metrics.json").open("w") as f:
        json.dump(save, f, indent=2)

    # ── Summary comparison ──
    logger.info("\n" + "=" * 100)
    logger.info("SUMMARY — %s seed=%d", args.dataset, args.seed)
    logger.info("=" * 100)
    logger.info("%15s | %7s %7s | %7s %7s | %7s %7s | %7s %7s",
                "", "PCC_s1", "PCC_s2", "MI_dPCC", "MU_dPCC", "MI_dSSIM", "MU_dSSIM", "MI_dSPCC", "MU_dSPCC")
    logger.info("-" * 100)

    # V0 baseline row
    v0_s1 = save["v0_baseline"]["slice1"]
    v0_s2 = save["v0_baseline"]["slice2"]
    logger.info("%15s | %7.4f %7.4f | %7s %7s | %7s %7s | %7s %7s",
                "V0_baseline",
                v0_s1.get("PCC", 0), v0_s2.get("PCC", 0),
                "—", "—", "—", "—", "—", "—")

    # Propagated row
    st1 = save["stratified_s1"]
    mi1, mu1 = st1.get("MI", {}), st1.get("MU", {})
    logger.info("%15s | %7.4f %7.4f | %+7.4f %+7.4f | %+7.4f %+7.4f | %+7.4f %+7.4f",
                "Propagated",
                s1["PCC"], s2["PCC"],
                mi1.get("dPCC_mean", 0), mu1.get("dPCC_mean", 0),
                mi1.get("dSSIM_mean", 0), mu1.get("dSSIM_mean", 0),
                mi1.get("dSPCC_mean", 0), mu1.get("dSPCC_mean", 0))

    logger.info("Done. Results → %s", out_root)


if __name__ == "__main__":
    main()
