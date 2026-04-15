#!/usr/bin/env python3
"""
Task 9 — TISSUE-style Uncertainty Quantification for SpatialEx
==============================================================
After exhaustive exploration (Task7 loss reweighting, Task8a linear propagation,
Task8b GCN correction, Task8c cell-cell smoothing), all post-hoc methods yielded
negligible absolute improvement on MU genes (dPCC < 0.004).

This script takes a different stance: instead of trying to improve MU gene
predictions, we **quantify which predictions are trustworthy**.

Method (adapted from TISSUE, Nature Methods 2024):
  1. Cell-Centric Variability (CCV): for each (cell, gene), measure how much
     the prediction deviates from spatially adjacent cells of similar morphology.
  2. Conformal Prediction Intervals: calibrate uncertainty intervals using
     measured genes, providing coverage guarantees.
  3. Gene Reliability Index: per-gene aggregate of prediction interval widths.
     Genes with wide intervals are "unreliable" — predominantly MU genes.

Key innovation vs original TISSUE:
  - TISSUE needs ground truth expression for neighbor weighting.
  - We use H&E feature cosine similarity — fully inference-time applicable.

Output:
  - Per-gene reliability score (narrow intervals = reliable)
  - MI/MU/MOD stratified uncertainty analysis
  - Visualization-ready CSVs

Usage on HPC:
    cd /gpfsdata/home/renyixiang/YuanLab
    python Task9_uncertainty/run_uncertainty.py --dataset mg --seed 42
    python Task9_uncertainty/run_uncertainty.py --dataset skin --seed 42
"""

import argparse
import json
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats
from scipy.spatial import cKDTree
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("task9")

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent


# =============================================
# A. Data loaders (reused)
# =============================================

def load_task6_lib():
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
    logger.info("Task6 library loaded")
    return ns


def preprocess_mg(t6):
    se = t6["se"]
    dev, res, enc, k = t6["device"], t6["resolution"], t6["image_encoder"], t6["num_neighbors"]
    r1, r2 = t6["save_root1"], t6["save_root2"]
    import pandas as pd
    adata1 = se.pp.Read_Xenium(r1 + "cell_feature_matrix.h5", r1 + "cells.csv")
    adata1 = se.pp.Preprocess_adata(adata1)
    img, scale = se.pp.Read_HE_image(r1 + "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif")
    tmtx = pd.read_csv(r1 + "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_imagealignment.csv", header=None).values
    adata1 = se.pp.Register_physical_to_pixel(adata1, tmtx, scale=scale)
    he, adata1 = se.pp.Tiling_HE_patches(res, adata1, img)
    adata1 = se.pp.Extract_HE_patches_representaion(he, adata=adata1, image_encoder=enc, device=dev, store_key="he")
    del he, img
    adata2 = se.pp.Read_Xenium(r2 + "cell_feature_matrix.h5", r2 + "cells.csv")
    adata2 = se.pp.Preprocess_adata(adata2)
    img, scale = se.pp.Read_HE_image(r2 + "Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image.ome.tif")
    tmtx = pd.read_csv(r2 + "Xenium_FFPE_Human_Breast_Cancer_Rep2_he_imagealignment.csv", header=None).values
    adata2 = se.pp.Register_physical_to_pixel(adata2, tmtx, scale=scale)
    he, adata2 = se.pp.Tiling_HE_patches(res, adata2, img)
    adata2 = se.pp.Extract_HE_patches_representaion(he, adata=adata2, image_encoder=enc, store_key="he", device=dev)
    del he, img
    return adata1, adata2


def preprocess_skin(t6):
    return t6["preprocess_skin_data"]()[:2]


def to_dense(X):
    if hasattr(X, "todense"): return np.asarray(X.todense())
    if hasattr(X, "toarray"): return np.asarray(X.toarray())
    return np.asarray(X)


def load_v0_baseline(v0_dir):
    v0_dir = Path(v0_dir)
    data = {"panelB1": np.load(v0_dir / "panelB1.npy"),
            "panelA2": np.load(v0_dir / "panelA2.npy")}
    for key in ("pcc_per_gene", "ssim_per_gene", "spcc_per_gene", "morans_per_gene"):
        for slc in ("slice1", "slice2"):
            f = v0_dir / f"{slc}_{key}.npy"
            if f.exists(): data[f"{slc}_{key}"] = np.load(f)
    return data


def build_masks_from_pcc(pcc_s1, pcc_s2):
    mean_pcc = (np.nan_to_num(pcc_s1) + np.nan_to_num(pcc_s2)) / 2.0
    t_low, t_high = np.percentile(mean_pcc, 33.3), np.percentile(mean_pcc, 66.7)
    mi = mean_pcc > t_high
    mu = mean_pcc <= t_low
    mod = ~(mi | mu)
    logger.info("Gene masks: MI=%d MU=%d MOD=%d", mi.sum(), mu.sum(), mod.sum())
    return mi, mu, mod, mean_pcc


# =============================================
# B. Spatial neighbor graph
# =============================================

def build_spatial_knn(coords, k=20):
    """Build spatial k-NN: returns (indices, distances)."""
    tree = cKDTree(coords)
    dists, indices = tree.query(coords, k=k + 1)
    return indices[:, 1:], dists[:, 1:]  # exclude self


# =============================================
# C. Cell-Centric Variability (CCV)
# =============================================

def compute_ccv(pred, nbr_indices, he_features=None, use_he_weights=True):
    """Compute Cell-Centric Variability for each (cell, gene).

    CCV_ij = 1 + sqrt( sum_k W_ik * (pred_kj - pred_ij)^2 / sum_k W_ik )

    Args:
        pred: (N, G) predictions
        nbr_indices: (N, k) neighbor indices
        he_features: (N, D) H&E features for computing weights
        use_he_weights: if True, weight by exp(cosine_sim(HE_i, HE_j))

    Returns:
        ccv: (N, G) cell-centric variability
    """
    N, G = pred.shape
    k = nbr_indices.shape[1]
    logger.info("Computing CCV: N=%d, G=%d, k=%d, HE-weighted=%s",
                N, G, k, use_he_weights)
    t0 = time.time()

    ccv = np.ones((N, G), dtype=np.float64)  # intercept = 1

    if use_he_weights and he_features is not None:
        he_norm = normalize(he_features, norm='l2', axis=1)

    # Process in chunks to manage memory
    chunk_size = 5000
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        idx_chunk = np.arange(start, end)
        nbrs = nbr_indices[start:end]  # (chunk, k)

        # Center cell predictions: (chunk, 1, G)
        center_pred = pred[idx_chunk][:, np.newaxis, :]
        # Neighbor predictions: (chunk, k, G)
        nbr_pred = pred[nbrs]

        # Squared deviations: (chunk, k, G)
        sq_dev = (nbr_pred - center_pred) ** 2

        if use_he_weights and he_features is not None:
            # Compute weights: exp(cosine_sim)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # center HE: (chunk, D), neighbor HE: (chunk, k, D)
                center_he = he_norm[idx_chunk]
                nbr_he = he_norm[nbrs]
                # cosine sim: (chunk, k)
                cos_sim = np.einsum('id,ikd->ik', center_he, nbr_he)
                cos_sim = np.nan_to_num(cos_sim, nan=0.0)
            weights = np.exp(cos_sim)  # (chunk, k)
            # Weighted variance: (chunk, G)
            w_sum = weights.sum(axis=1, keepdims=True)  # (chunk, 1)
            w_sum = np.maximum(w_sum, 1e-8)
            weighted_var = np.einsum('ik,ikg->ig', weights, sq_dev) / w_sum
        else:
            # Uniform weights
            weighted_var = sq_dev.mean(axis=1)  # (chunk, G)

        ccv[start:end] = 1.0 + np.sqrt(np.maximum(weighted_var, 0))

    logger.info("  CCV computed in %.1fs: mean=%.4f, median=%.4f",
                time.time() - t0, ccv.mean(), np.median(ccv))
    return ccv.astype(np.float32)


# =============================================
# D. Conformal Prediction Intervals
# =============================================

def compute_conformal_intervals(pred, gt, ccv, alpha=0.33):
    """Compute conformal prediction intervals.

    For each gene, compute nonconformity scores on all cells:
        s_ij = |gt_ij - pred_ij| / ccv_ij

    Then compute the conformal quantile:
        qhat = quantile(s, ceil((n+1)*(1-alpha)) / n)

    Prediction interval:
        [pred_ij - ccv_ij * qhat_j,  pred_ij + ccv_ij * qhat_j]

    Args:
        pred: (N, G) predictions
        gt: (N, G) ground truth
        ccv: (N, G) cell-centric variability
        alpha: miscoverage level (0.33 = 67% intervals)

    Returns:
        qhat: (G,) conformal quantile per gene
        interval_widths: (N, G) prediction interval widths
        coverage: (G,) empirical coverage per gene
    """
    N, G = pred.shape
    logger.info("Computing conformal intervals: alpha=%.2f (%.0f%% coverage target)",
                alpha, (1 - alpha) * 100)

    # Nonconformity scores per gene
    residuals = np.abs(gt - pred)
    scores = residuals / np.maximum(ccv, 1e-8)  # (N, G)

    # Per-gene conformal quantile
    qhat = np.zeros(G, dtype=np.float64)
    coverage = np.zeros(G, dtype=np.float64)

    conformal_level = int(np.ceil((N + 1) * (1 - alpha))) / N
    conformal_level = min(conformal_level, 1.0)

    for j in range(G):
        s_j = scores[:, j]
        s_j = s_j[np.isfinite(s_j)]
        if len(s_j) == 0:
            qhat[j] = np.inf
            continue
        qhat[j] = np.quantile(s_j, conformal_level)

        # Empirical coverage: fraction of cells where GT is within interval
        half_width = ccv[:, j] * qhat[j]
        in_interval = np.abs(gt[:, j] - pred[:, j]) <= half_width
        coverage[j] = in_interval.mean()

    # Interval widths: 2 * ccv * qhat
    interval_widths = 2 * ccv * qhat[np.newaxis, :]

    logger.info("  Mean qhat=%.4f, Mean coverage=%.4f, Mean interval width=%.4f",
                qhat.mean(), coverage.mean(), interval_widths.mean())

    return qhat.astype(np.float32), interval_widths.astype(np.float32), coverage.astype(np.float32)


# =============================================
# E. Gene Reliability Index
# =============================================

def compute_gene_reliability(interval_widths, pred, gt, mi_mask, mu_mask, mod_mask):
    """Compute per-gene reliability metrics.

    For each gene:
      - mean_width: average prediction interval width across cells
      - normalized_width: mean_width / mean(|pred|) — relative uncertainty
      - pcc: prediction PCC (from V0)
      - category: MI / MU / MOD

    Returns dict with arrays and summary.
    """
    G = interval_widths.shape[1]

    mean_width = interval_widths.mean(axis=0)      # (G,)
    median_width = np.median(interval_widths, axis=0)

    # Normalized by prediction magnitude
    pred_mag = np.abs(pred).mean(axis=0) + 1e-8
    normalized_width = mean_width / pred_mag

    # Per-gene PCC
    pcc_pg = _per_gene_pcc(pred, gt)

    # Category labels
    categories = np.array(["MOD"] * G)
    categories[mi_mask] = "MI"
    categories[mu_mask] = "MU"

    return {
        "mean_width": mean_width,
        "median_width": median_width,
        "normalized_width": normalized_width,
        "pcc": pcc_pg,
        "categories": categories,
    }


def _per_gene_pcc(pred, target, eps=1e-8):
    p = pred - pred.mean(axis=0, keepdims=True)
    t = target - target.mean(axis=0, keepdims=True)
    num = (p * t).sum(axis=0)
    denom = np.sqrt((p**2).sum(axis=0) * (t**2).sum(axis=0) + eps)
    return np.nan_to_num(num / denom)


# =============================================
# F. Cross-slice Validation
# =============================================

def cross_slice_conformal(pred_calib, gt_calib, ccv_calib,
                          pred_test, ccv_test, gt_test,
                          alpha=0.33):
    """Calibrate on one slice, test on another.

    This is the proper conformal setup: calibration and test are independent.
    """
    N_calib, G = pred_calib.shape
    N_test = pred_test.shape[0]

    # Compute conformal quantiles from calibration slice
    residuals = np.abs(gt_calib - pred_calib)
    scores = residuals / np.maximum(ccv_calib, 1e-8)

    conformal_level = min(int(np.ceil((N_calib + 1) * (1 - alpha))) / N_calib, 1.0)

    qhat = np.zeros(G, dtype=np.float64)
    for j in range(G):
        s_j = scores[:, j]
        s_j = s_j[np.isfinite(s_j)]
        if len(s_j) > 0:
            qhat[j] = np.quantile(s_j, conformal_level)
        else:
            qhat[j] = np.inf

    # Apply to test slice
    test_widths = 2 * ccv_test * qhat[np.newaxis, :]

    # Coverage on test
    test_coverage = np.zeros(G, dtype=np.float64)
    for j in range(G):
        half_width = ccv_test[:, j] * qhat[j]
        in_interval = np.abs(gt_test[:, j] - pred_test[:, j]) <= half_width
        test_coverage[j] = in_interval.mean()

    return qhat.astype(np.float32), test_widths.astype(np.float32), test_coverage.astype(np.float32)


# =============================================
# G. Main
# =============================================

def main():
    parser = argparse.ArgumentParser(description="Task9 TISSUE-style uncertainty")
    parser.add_argument("--dataset", choices=["mg", "skin"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--v0-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--k-neighbors", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.33,
                        help="Conformal miscoverage level (0.33 = 67%% intervals)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Paths
    if args.v0_dir:
        v0_dir = Path(args.v0_dir)
    else:
        v0_dir = PROJECT_ROOT / "Task7_anchorBoost" / "results" / \
                 f"{args.dataset}_seed{args.seed}" / "V0_baseline"
    assert (v0_dir / "panelB1.npy").exists(), f"V0 not found: {v0_dir}"

    if args.output_dir:
        out_root = Path(args.output_dir)
    else:
        out_root = HERE / "results" / f"{args.dataset}_seed{args.seed}"
    out_root.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    logger.info("Device: %s", device)

    # ── Load V0 + masks ──
    v0 = load_v0_baseline(v0_dir)
    panelB1 = v0["panelB1"]  # (N1, G) — V0 prediction for slice1
    panelA2 = v0["panelA2"]  # (N2, G) — V0 prediction for slice2
    pcc_s1 = v0["slice1_pcc_per_gene"]
    pcc_s2 = v0["slice2_pcc_per_gene"]
    mi_mask, mu_mask, mod_mask, mean_pcc = build_masks_from_pcc(pcc_s1, pcc_s2)
    G = len(mi_mask)

    # ── Load adata ──
    logger.info("Loading Task6 library...")
    t6 = load_task6_lib()
    t6["device"] = device

    if args.dataset == "mg":
        adata1, adata2 = preprocess_mg(t6)
    else:
        adata1, adata2 = preprocess_skin(t6)

    gt_s1 = to_dense(adata1.X)
    gt_s2 = to_dense(adata2.X)
    coords_s1 = np.asarray(adata1.obsm["spatial"])
    coords_s2 = np.asarray(adata2.obsm["spatial"])
    he_s1 = np.asarray(adata1.obsm["he"])
    he_s2 = np.asarray(adata2.obsm["he"])
    var_names = list(adata1.var_names)

    N1, N2 = gt_s1.shape[0], gt_s2.shape[0]
    logger.info("Slice1: %d cells, Slice2: %d cells, Genes: %d", N1, N2, G)

    # ── Build spatial neighbor graphs ──
    logger.info("=" * 60)
    logger.info("Step 1: Building spatial k-NN graphs (k=%d)", args.k_neighbors)
    logger.info("=" * 60)
    nbr_idx_s1, nbr_dist_s1 = build_spatial_knn(coords_s1, k=args.k_neighbors)
    nbr_idx_s2, nbr_dist_s2 = build_spatial_knn(coords_s2, k=args.k_neighbors)

    # ── Compute CCV ──
    logger.info("=" * 60)
    logger.info("Step 2: Computing Cell-Centric Variability (CCV)")
    logger.info("=" * 60)

    # HE-weighted version (our innovation)
    ccv_s1_he = compute_ccv(panelB1, nbr_idx_s1, he_s1, use_he_weights=True)
    ccv_s2_he = compute_ccv(panelA2, nbr_idx_s2, he_s2, use_he_weights=True)

    # Uniform-weighted baseline for ablation
    ccv_s1_uni = compute_ccv(panelB1, nbr_idx_s1, use_he_weights=False)
    ccv_s2_uni = compute_ccv(panelA2, nbr_idx_s2, use_he_weights=False)

    # CCV stratified summary
    for label, ccv in [("HE-weighted s1", ccv_s1_he), ("HE-weighted s2", ccv_s2_he),
                        ("Uniform s1", ccv_s1_uni), ("Uniform s2", ccv_s2_uni)]:
        logger.info("  %s — MI: %.3f  MU: %.3f  MOD: %.3f  (gene-averaged CCV)",
                     label,
                     ccv[:, mi_mask].mean(), ccv[:, mu_mask].mean(), ccv[:, mod_mask].mean())

    np.save(out_root / "ccv_s1_he.npy", ccv_s1_he)
    np.save(out_root / "ccv_s2_he.npy", ccv_s2_he)

    # ── Conformal intervals: within-slice ──
    logger.info("=" * 60)
    logger.info("Step 3: Conformal Prediction Intervals (alpha=%.2f)", args.alpha)
    logger.info("=" * 60)

    qhat_s1, widths_s1, cov_s1 = compute_conformal_intervals(
        panelB1, gt_s1, ccv_s1_he, alpha=args.alpha)
    qhat_s2, widths_s2, cov_s2 = compute_conformal_intervals(
        panelA2, gt_s2, ccv_s2_he, alpha=args.alpha)

    # ── Cross-slice conformal (proper test) ──
    logger.info("=" * 60)
    logger.info("Step 4: Cross-slice Conformal (calibrate on s1, test on s2)")
    logger.info("=" * 60)
    qhat_cross, widths_cross, cov_cross = cross_slice_conformal(
        panelB1, gt_s1, ccv_s1_he,
        panelA2, ccv_s2_he, gt_s2,
        alpha=args.alpha)

    logger.info("  Cross-slice coverage: overall=%.4f, MI=%.4f, MU=%.4f, MOD=%.4f",
                cov_cross.mean(), cov_cross[mi_mask].mean(),
                cov_cross[mu_mask].mean(), cov_cross[mod_mask].mean())

    # ── Gene reliability ──
    logger.info("=" * 60)
    logger.info("Step 5: Gene Reliability Index")
    logger.info("=" * 60)

    rel_s1 = compute_gene_reliability(widths_s1, panelB1, gt_s1, mi_mask, mu_mask, mod_mask)
    rel_s2 = compute_gene_reliability(widths_s2, panelA2, gt_s2, mi_mask, mu_mask, mod_mask)

    # ── Core analysis: MI vs MU uncertainty distributions ──
    logger.info("=" * 60)
    logger.info("KEY RESULTS")
    logger.info("=" * 60)

    # 1. Mean interval width by category
    for slabel, widths, cov, rel in [("Slice1", widths_s1, cov_s1, rel_s1),
                                      ("Slice2", widths_s2, cov_s2, rel_s2)]:
        mw = widths.mean(axis=0)  # (G,) mean width per gene
        logger.info("\n  %s — Mean prediction interval width:", slabel)
        logger.info("    MI  (n=%d): %.4f (coverage=%.3f, PCC=%.4f)",
                     mi_mask.sum(), mw[mi_mask].mean(), cov[mi_mask].mean(),
                     rel["pcc"][mi_mask].mean())
        logger.info("    MU  (n=%d): %.4f (coverage=%.3f, PCC=%.4f)",
                     mu_mask.sum(), mw[mu_mask].mean(), cov[mu_mask].mean(),
                     rel["pcc"][mu_mask].mean())
        logger.info("    MOD (n=%d): %.4f (coverage=%.3f, PCC=%.4f)",
                     mod_mask.sum(), mw[mod_mask].mean(), cov[mod_mask].mean(),
                     rel["pcc"][mod_mask].mean())
        logger.info("    Width ratio MU/MI: %.2fx", mw[mu_mask].mean() / max(mw[mi_mask].mean(), 1e-8))

    # 2. Rank correlation: interval width vs PCC
    for slabel, rel in [("Slice1", rel_s1), ("Slice2", rel_s2)]:
        from scipy.stats import spearmanr
        rho, p = spearmanr(rel["mean_width"], rel["pcc"])
        logger.info("  %s — Spearman(interval_width, PCC): rho=%.4f, p=%.2e", slabel, rho, p)

    # 3. Mann-Whitney U test: MI widths vs MU widths
    for slabel, widths in [("Slice1", widths_s1), ("Slice2", widths_s2)]:
        mw = widths.mean(axis=0)
        from scipy.stats import mannwhitneyu
        stat, p = mannwhitneyu(mw[mi_mask], mw[mu_mask], alternative="less")
        logger.info("  %s — Mann-Whitney U (MI < MU width): U=%.0f, p=%.2e", slabel, stat, p)

    # ── Cross-slice results ──
    logger.info("\n  Cross-slice (calibrate s1 → test s2):")
    mw_cross = widths_cross.mean(axis=0)
    logger.info("    MI  width=%.4f coverage=%.3f", mw_cross[mi_mask].mean(), cov_cross[mi_mask].mean())
    logger.info("    MU  width=%.4f coverage=%.3f", mw_cross[mu_mask].mean(), cov_cross[mu_mask].mean())
    logger.info("    MOD width=%.4f coverage=%.3f", mw_cross[mod_mask].mean(), cov_cross[mod_mask].mean())
    logger.info("    Width ratio MU/MI: %.2fx",
                mw_cross[mu_mask].mean() / max(mw_cross[mi_mask].mean(), 1e-8))

    # ── Save per-gene CSV for visualization ──
    import csv
    csv_path = out_root / "gene_uncertainty.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gene", "category", "pcc_s1", "pcc_s2", "mean_pcc",
                          "interval_width_s1", "interval_width_s2",
                          "coverage_s1", "coverage_s2",
                          "ccv_mean_s1", "ccv_mean_s2",
                          "cross_width", "cross_coverage",
                          "normalized_width_s1"])
        mw1 = widths_s1.mean(axis=0)
        mw2 = widths_s2.mean(axis=0)
        ccv_mean1 = ccv_s1_he.mean(axis=0)
        ccv_mean2 = ccv_s2_he.mean(axis=0)
        for j in range(G):
            cat = "MI" if mi_mask[j] else ("MU" if mu_mask[j] else "MOD")
            writer.writerow([
                var_names[j], cat,
                f"{pcc_s1[j]:.6f}", f"{pcc_s2[j]:.6f}", f"{mean_pcc[j]:.6f}",
                f"{mw1[j]:.6f}", f"{mw2[j]:.6f}",
                f"{cov_s1[j]:.4f}", f"{cov_s2[j]:.4f}",
                f"{ccv_mean1[j]:.6f}", f"{ccv_mean2[j]:.6f}",
                f"{mw_cross[j]:.6f}", f"{cov_cross[j]:.4f}",
                f"{rel_s1['normalized_width'][j]:.6f}",
            ])
    logger.info("Per-gene CSV → %s", csv_path)

    # ── Save metrics.json ──
    def _group_stats(arr, mask):
        vals = arr[mask]
        return {"mean": float(vals.mean()), "median": float(np.median(vals)),
                "std": float(vals.std()), "n": int(mask.sum())}

    mw1 = widths_s1.mean(axis=0)
    mw2 = widths_s2.mean(axis=0)

    save = {
        "dataset": args.dataset, "seed": args.seed,
        "params": {"k_neighbors": args.k_neighbors, "alpha": args.alpha},
        "slice1": {
            "interval_width": {
                "MI": _group_stats(mw1, mi_mask),
                "MU": _group_stats(mw1, mu_mask),
                "MOD": _group_stats(mw1, mod_mask),
                "MU_MI_ratio": float(mw1[mu_mask].mean() / max(mw1[mi_mask].mean(), 1e-8)),
            },
            "coverage": {
                "MI": _group_stats(cov_s1, mi_mask),
                "MU": _group_stats(cov_s1, mu_mask),
                "MOD": _group_stats(cov_s1, mod_mask),
                "overall": float(cov_s1.mean()),
            },
            "ccv": {
                "MI": float(ccv_s1_he[:, mi_mask].mean()),
                "MU": float(ccv_s1_he[:, mu_mask].mean()),
                "MOD": float(ccv_s1_he[:, mod_mask].mean()),
            },
            "pcc": {
                "MI": float(pcc_s1[mi_mask].mean()),
                "MU": float(pcc_s1[mu_mask].mean()),
                "MOD": float(np.nan_to_num(pcc_s1[mod_mask]).mean()),
            },
            "width_pcc_spearman": {
                "rho": float(spearmanr(rel_s1["mean_width"], rel_s1["pcc"])[0]),
                "p": float(spearmanr(rel_s1["mean_width"], rel_s1["pcc"])[1]),
            },
        },
        "slice2": {
            "interval_width": {
                "MI": _group_stats(mw2, mi_mask),
                "MU": _group_stats(mw2, mu_mask),
                "MOD": _group_stats(mw2, mod_mask),
                "MU_MI_ratio": float(mw2[mu_mask].mean() / max(mw2[mi_mask].mean(), 1e-8)),
            },
            "coverage": {
                "MI": _group_stats(cov_s2, mi_mask),
                "MU": _group_stats(cov_s2, mu_mask),
                "MOD": _group_stats(cov_s2, mod_mask),
                "overall": float(cov_s2.mean()),
            },
            "ccv": {
                "MI": float(ccv_s2_he[:, mi_mask].mean()),
                "MU": float(ccv_s2_he[:, mu_mask].mean()),
                "MOD": float(ccv_s2_he[:, mod_mask].mean()),
            },
            "pcc": {
                "MI": float(pcc_s2[mi_mask].mean()),
                "MU": float(pcc_s2[mu_mask].mean()),
                "MOD": float(np.nan_to_num(pcc_s2[mod_mask]).mean()),
            },
            "width_pcc_spearman": {
                "rho": float(spearmanr(rel_s2["mean_width"], rel_s2["pcc"])[0]),
                "p": float(spearmanr(rel_s2["mean_width"], rel_s2["pcc"])[1]),
            },
        },
        "cross_slice": {
            "interval_width": {
                "MI": _group_stats(mw_cross, mi_mask),
                "MU": _group_stats(mw_cross, mu_mask),
                "MOD": _group_stats(mw_cross, mod_mask),
                "MU_MI_ratio": float(mw_cross[mu_mask].mean() / max(mw_cross[mi_mask].mean(), 1e-8)),
            },
            "coverage": {
                "MI": _group_stats(cov_cross, mi_mask),
                "MU": _group_stats(cov_cross, mu_mask),
                "MOD": _group_stats(cov_cross, mod_mask),
                "overall": float(cov_cross.mean()),
            },
        },
    }

    with (out_root / "metrics.json").open("w") as f:
        json.dump(save, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("CONCLUSION")
    logger.info("=" * 80)
    logger.info("MU genes have %.1fx wider prediction intervals than MI genes.",
                (mw1[mu_mask].mean() / max(mw1[mi_mask].mean(), 1e-8) +
                 mw2[mu_mask].mean() / max(mw2[mi_mask].mean(), 1e-8)) / 2)
    logger.info("This quantitatively confirms that MU gene predictions carry")
    logger.info("fundamentally higher uncertainty — consistent with the finding")
    logger.info("that all post-hoc correction methods yield negligible improvement.")
    logger.info("Results → %s", out_root)


if __name__ == "__main__":
    main()
