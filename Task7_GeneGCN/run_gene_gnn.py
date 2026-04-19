#!/usr/bin/env python3
"""
Lightweight 2-layer GCN on the gene-gene graph (PPI + Spearman co-expression),
trained to predict a residual correction on top of the Task 5 baseline.
MI genes' deltas are masked to zero so the GCN only adjusts MU/MOD predictions.

    python run_gene_gnn.py --dataset {mg,skin} --seed 42
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
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as scipy_stats
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("task8b")

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent


# A. Reuse loaders from run_propagation.py

def load_task6_lib():
    script = PROJECT_ROOT / "Task5_PPI" / "run_wgcna_ppi_pipeline.py"
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
    data = {"panelB1": np.load(v0_dir / "panelB1.npy"), "panelA2": np.load(v0_dir / "panelA2.npy")}
    for key in ("pcc_per_gene", "ssim_per_gene", "spcc_per_gene", "morans_per_gene"):
        for slc in ("slice1", "slice2"):
            f = v0_dir / f"{slc}_{key}.npy"
            if f.exists(): data[f"{slc}_{key}"] = np.load(f)
    logger.info("V0 loaded: panelB1 %s, panelA2 %s", data["panelB1"].shape, data["panelA2"].shape)
    return data


def build_masks_from_pcc(pcc_s1, pcc_s2):
    mean_pcc = (np.nan_to_num(pcc_s1) + np.nan_to_num(pcc_s2)) / 2.0
    t_low, t_high = np.percentile(mean_pcc, 33.3), np.percentile(mean_pcc, 66.7)
    mi, mu, mod = mean_pcc > t_high, mean_pcc <= t_low, np.zeros_like(mean_pcc, dtype=bool)
    mod = ~(mi | mu)
    logger.info("Gene masks: MI=%d MU=%d MOD=%d", mi.sum(), mu.sum(), mod.sum())
    return mi, mu, mod, mean_pcc


def load_ppi(ppi_dir, var_names):
    ppi_path = os.path.join(ppi_dir, "ppi_matrix.npy")
    if not os.path.exists(ppi_path):
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
    logger.info("PPI loaded: %s, %d edges", ppi.shape, np.count_nonzero(ppi) // 2)
    return ppi.astype(np.float32)


def compute_coexpression(expr):
    G = expr.shape[1]
    N = expr.shape[0]
    if N > 5000:
        idx = np.random.RandomState(42).choice(N, 5000, replace=False)
        expr = expr[idx]
    corr, _ = spearmanr(expr, axis=0)
    corr = np.nan_to_num(np.abs(corr))
    np.fill_diagonal(corr, 0.0)
    logger.info("Co-expression: mean=%.4f, max=%.4f", corr.mean(), corr.max())
    return corr.astype(np.float32)


# B. Gene graph construction

def build_gene_adjacency(ppi, coexpr, mi_mask, mu_mask, mod_mask,
                         ppi_weight=0.5, coexpr_weight=0.5,
                         coexpr_threshold=0.3, add_self_loops=True):
    """Build normalized gene-gene adjacency for GCN.

    Asymmetric: MI→MU edges kept, MU→MI edges zeroed.
    Self-loops added for residual-like behavior.

    A[i, j] = influence of gene j on gene i (row-normalized).
    Used as: H = A @ X (for each feature dimension).
    """
    G = ppi.shape[0]
    coexpr_bin = (coexpr >= coexpr_threshold).astype(np.float32) * coexpr
    W = ppi_weight * ppi + coexpr_weight * coexpr_bin
    np.fill_diagonal(W, 0.0)

    # Asymmetric masking: zero out MU→MI and MU→MOD edges
    A = W.copy()
    for i in range(G):
        for j in range(G):
            if A[i, j] == 0: continue
            # A[i,j] = influence of j on i
            # Block: j=MU influencing i=MI
            if mu_mask[j] and mi_mask[i]: A[i, j] = 0.0
            # Block: j=MU influencing i=MOD
            elif mu_mask[j] and mod_mask[i]: A[i, j] = 0.0

    if add_self_loops:
        np.fill_diagonal(A, 1.0)

    # Row-normalize: each gene i's incoming influences sum to 1
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    A = A / row_sums

    logger.info("Gene adjacency: %d edges (excl self-loops), asymmetric",
                np.count_nonzero(A) - (np.diag(A) > 0).sum())
    return A.astype(np.float32)


# C. GeneGCN Model

class GeneGCN(nn.Module):
    """Lightweight 2-layer GCN on gene graph for prediction correction.

    For each cell, processes the G-dimensional V0 prediction vector through
    graph convolutions on the gene-gene adjacency, learning non-linear
    corrections. Outputs a residual delta added to V0 predictions.

    Input:  (batch, G) — V0 predictions per cell
    Output: (batch, G) — corrected predictions (V0 + learned delta)
    """

    def __init__(self, n_genes, hidden=64, n_layers=2, dropout=0.1):
        super().__init__()
        self.n_genes = n_genes
        self.n_layers = n_layers

        # Lift scalar per gene → hidden dim
        self.input_proj = nn.Linear(1, hidden)

        # GCN layers
        self.gc_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.gc_layers.append(nn.Linear(hidden, hidden))

        # Project back to scalar per gene
        self.output_proj = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden)

    def forward(self, x, A):
        """
        Args:
            x: (batch, G) V0 predictions
            A: (G, G) row-normalized adjacency (on same device)
        Returns:
            corrected: (batch, G) = x + delta
        """
        # x: (B, G) → (B, G, 1)
        h = x.unsqueeze(-1)
        # Lift to hidden: (B, G, 1) → (B, G, H)
        h = F.relu(self.input_proj(h))

        # GCN layers with residual
        for gc in self.gc_layers:
            # Message passing: aggregate neighbor features
            # h: (B, G, H), A: (G, G)
            # h_agg[b, i, :] = sum_j A[i,j] * h[b, j, :]
            h_agg = torch.einsum('ij,bjf->bif', A, h)
            # Transform + residual
            h_new = gc(h_agg)
            h_new = self.layer_norm(h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            h = h + h_new  # residual connection

        # Project back: (B, G, H) → (B, G, 1) → (B, G)
        delta = self.output_proj(h).squeeze(-1)

        # Residual learning: output = V0 + delta
        corrected = x + delta
        return corrected


# D. Training

def train_gene_gnn(model, A, pred_train, gt_train, mi_mask, mu_mask, mod_mask,
                   pred_val, gt_val,
                   epochs=200, batch_size=2048, lr=1e-3, weight_decay=1e-4,
                   lambda_mu=3.0, lambda_mod=1.5, lambda_mi=0.0,
                   device="cpu", patience=20):
    """Train the GeneGCN model.

    Loss weighting:
      - MI genes: lambda_mi=0 (no loss — snap-back ensures they stay at V0)
      - MU genes: lambda_mu=3.0 (focus)
      - MOD genes: lambda_mod=1.5
    """
    model = model.to(device)
    A_t = torch.from_numpy(A).float().to(device)

    # Build gene weight vector
    G = len(mi_mask)
    gene_weights = torch.ones(G, device=device)
    gene_weights[mi_mask] = lambda_mi
    gene_weights[mu_mask] = lambda_mu
    gene_weights[mod_mask] = lambda_mod
    gene_weights = gene_weights / gene_weights.sum() * G  # normalize to mean=1

    pred_train_t = torch.from_numpy(pred_train).float()
    gt_train_t = torch.from_numpy(gt_train).float()
    pred_val_t = torch.from_numpy(pred_val).float().to(device)
    gt_val_t = torch.from_numpy(gt_val).float().to(device)

    N_train = pred_train_t.shape[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    best_val_mu_pcc = -1.0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(N_train)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, N_train, batch_size):
            idx = perm[i:i+batch_size]
            x_batch = pred_train_t[idx].to(device)
            y_batch = gt_train_t[idx].to(device)

            pred = model(x_batch, A_t)

            # Weighted MSE
            diff = (pred - y_batch) ** 2  # (B, G)
            loss = (diff * gene_weights.unsqueeze(0)).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        # Validation: compute MU gene PCC
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                # Process val in chunks to save memory
                val_preds = []
                for i in range(0, pred_val_t.shape[0], batch_size * 2):
                    chunk = pred_val_t[i:i+batch_size*2]
                    val_preds.append(model(chunk, A_t))
                val_pred = torch.cat(val_preds, dim=0)

                # Snap MI back
                val_pred[:, torch.from_numpy(mi_mask).to(device)] = pred_val_t[:, torch.from_numpy(mi_mask).to(device)]

                # Per-gene PCC for MU genes
                val_np = val_pred.cpu().numpy()
                gt_np = gt_val_t.cpu().numpy()
                pcc = _per_gene_pcc(val_np, gt_np)
                mu_pcc = float(pcc[mu_mask].mean())
                overall_pcc = float(pcc.mean())

            scheduler.step(-mu_pcc)

            if mu_pcc > best_val_mu_pcc:
                best_val_mu_pcc = mu_pcc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
                marker = " *"
            else:
                no_improve += 1
                marker = ""

            logger.info("  Epoch %3d | loss=%.4f | MU_PCC=%.4f Overall=%.4f%s",
                        epoch + 1, avg_loss, mu_pcc, overall_pcc, marker)

            if no_improve >= patience:
                logger.info("  Early stopping at epoch %d", epoch + 1)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info("Best val MU_PCC=%.4f", best_val_mu_pcc)
    return model


def predict_with_gnn(model, A, pred, mi_mask, device="cpu", batch_size=4096):
    """Run inference and snap MI genes back."""
    model = model.to(device).eval()
    A_t = torch.from_numpy(A).float().to(device)
    pred_t = torch.from_numpy(pred).float()

    outputs = []
    with torch.no_grad():
        for i in range(0, pred_t.shape[0], batch_size):
            chunk = pred_t[i:i+batch_size].to(device)
            out = model(chunk, A_t)
            # MI snap-back
            out[:, torch.from_numpy(mi_mask).to(device)] = chunk[:, torch.from_numpy(mi_mask).to(device)]
            outputs.append(out.cpu().numpy())

    return np.concatenate(outputs, axis=0)


# E. Metrics

def _per_gene_pcc(pred, target, eps=1e-8):
    p = pred - pred.mean(axis=0, keepdims=True)
    t = target - target.mean(axis=0, keepdims=True)
    num = (p * t).sum(axis=0)
    denom = np.sqrt((p**2).sum(axis=0) * (t**2).sum(axis=0) + eps)
    return np.nan_to_num(num / denom)


def stratified_report(mi, mu, mod, per_gene_metrics, baseline_metrics=None):
    report = {"overall": {}, "MI": {"n": int(mi.sum())}, "MU": {"n": int(mu.sum())},
              "Moderate": {"n": int(mod.sum())}, "per_gene": {}}
    for metric, vals in per_gene_metrics.items():
        vals = np.nan_to_num(vals)
        report["overall"][metric] = float(vals.mean())
        report["per_gene"][metric] = vals
        bl_vals = baseline_metrics.get(metric) if baseline_metrics else None
        if bl_vals is not None: bl_vals = np.nan_to_num(bl_vals)
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
    metrics = list(rep["overall"].keys())
    logger.info("  Overall: %s", "  ".join(f"{m}={rep['overall'][m]:.4f}" for m in metrics))
    for grp in ("MI", "MU", "Moderate"):
        r = rep[grp]
        n = r.get("n", 0)
        parts = [f"  {grp:>10s} (n={n:3d})"]
        for m in metrics:
            val = r.get(f"{m}_mean", float("nan"))
            dval = r.get(f"d{m}_mean", float("nan"))
            p = r.get(f"d{m}_wilcoxon_p", float("nan"))
            star = "***" if p < 0.001 else "** " if p < 0.01 else "*  " if p < 0.05 else "   "
            parts.append(f"  {m}={val:.4f} (d={dval:+.4f} {star})")
        logger.info("  ".join(parts))


# F. Main

def main():
    parser = argparse.ArgumentParser(description="Task8b GeneGCN for MU gene correction")
    parser.add_argument("--dataset", choices=["mg", "skin"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--v0-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-mu", type=float, default=3.0)
    parser.add_argument("--lambda-mod", type=float, default=1.5)
    parser.add_argument("--coexpr-threshold", type=float, default=0.3)
    parser.add_argument("--ppi-weight", type=float, default=0.5)
    args = parser.parse_args()

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # Paths
    if args.v0_dir:
        v0_dir = Path(args.v0_dir)
    else:
        v0_dir = PROJECT_ROOT / "Task7_anchorBoost" / "results" / f"{args.dataset}_seed{args.seed}" / "V0_baseline"
    assert (v0_dir / "panelB1.npy").exists(), f"V0 not found: {v0_dir}"

    if args.output_dir:
        out_root = Path(args.output_dir)
    else:
        out_root = HERE / "results_gnn" / f"{args.dataset}_seed{args.seed}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.info("Device: %s", device)

    # ── Load data ──
    v0 = load_v0_baseline(v0_dir)
    panelB1_orig = v0["panelB1"]
    panelA2_orig = v0["panelA2"]
    pcc_s1, pcc_s2 = v0["slice1_pcc_per_gene"], v0["slice2_pcc_per_gene"]
    mi_mask, mu_mask, mod_mask, mean_pcc = build_masks_from_pcc(pcc_s1, pcc_s2)
    G = len(mi_mask)

    # Task6 for ground truth + evaluation
    logger.info("Loading Task6 library...")
    t6 = load_task6_lib()
    if device != "cuda":
        t6["device"] = "cpu"
    else:
        t6["device"] = device

    if args.dataset == "mg":
        adata1, adata2 = preprocess_mg(t6)
        ppi_dir = t6["PPI_DATA_DIR"]
    else:
        adata1, adata2 = preprocess_skin(t6)
        ppi_dir = t6["PPI_DATA_DIR_SKIN"]

    gt_s1 = to_dense(adata1.X)
    gt_s2 = to_dense(adata2.X)
    var_names = adata1.var_names.tolist()

    # ── Gene graph ──
    ppi = load_ppi(ppi_dir, var_names)
    coexpr_path = out_root / "coexpression_matrix.npy"
    if coexpr_path.exists():
        coexpr = np.load(coexpr_path)
    else:
        coexpr = compute_coexpression(np.vstack([gt_s1, gt_s2]))
        np.save(coexpr_path, coexpr)

    A = build_gene_adjacency(
        ppi, coexpr, mi_mask, mu_mask, mod_mask,
        ppi_weight=args.ppi_weight, coexpr_weight=1.0 - args.ppi_weight,
        coexpr_threshold=args.coexpr_threshold, add_self_loops=True,
    )
    np.save(out_root / "gene_adjacency.npy", A)

    # ── Train/Val split: use 80% of slice1 cells for train, 20% for val ──
    N1 = gt_s1.shape[0]
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(N1)
    n_train = int(0.8 * N1)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    logger.info("Train: %d cells, Val: %d cells, Genes: %d (MI=%d MU=%d MOD=%d)",
                len(train_idx), len(val_idx), G, mi_mask.sum(), mu_mask.sum(), mod_mask.sum())

    # ── Build & train model ──
    model = GeneGCN(n_genes=G, hidden=args.hidden, n_layers=args.n_layers, dropout=0.1)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("GeneGCN: %d params, hidden=%d, layers=%d", n_params, args.hidden, args.n_layers)

    logger.info("=" * 60)
    logger.info("Training GeneGCN")
    logger.info("=" * 60)
    t0 = time.time()
    model = train_gene_gnn(
        model, A,
        pred_train=panelB1_orig[train_idx], gt_train=gt_s1[train_idx],
        mi_mask=mi_mask, mu_mask=mu_mask, mod_mask=mod_mask,
        pred_val=panelB1_orig[val_idx], gt_val=gt_s1[val_idx],
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=1e-4,
        lambda_mu=args.lambda_mu, lambda_mod=args.lambda_mod, lambda_mi=0.0,
        device=device, patience=20,
    )
    train_time = time.time() - t0
    logger.info("Training took %.1f min", train_time / 60)

    # Save model
    torch.save(model.state_dict(), out_root / "gene_gcn.pt")

    # ── Inference on both slices ──
    logger.info("=" * 60)
    logger.info("Inference")
    logger.info("=" * 60)
    panelB1_corr = predict_with_gnn(model, A, panelB1_orig, mi_mask, device=device)
    panelA2_corr = predict_with_gnn(model, A, panelA2_orig, mi_mask, device=device)

    np.save(out_root / "panelB1_gnn.npy", panelB1_corr)
    np.save(out_root / "panelA2_gnn.npy", panelA2_corr)

    # ── Evaluate ──
    logger.info("=" * 60)
    logger.info("Evaluation")
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
    baseline_pg_s1, baseline_pg_s2 = {}, {}
    for mkey, fkey in [("PCC", "pcc_per_gene"), ("SSIM", "ssim_per_gene"),
                       ("SPCC", "spcc_per_gene"), ("MoransI", "morans_per_gene")]:
        k1, k2 = f"slice1_{fkey}", f"slice2_{fkey}"
        if k1 in v0: baseline_pg_s1[mkey] = v0[k1]
        if k2 in v0: baseline_pg_s2[mkey] = v0[k2]

    def _pg(s):
        return {k: np.asarray(s[v]) for k, v in
                [("PCC", "pcc_per_gene"), ("SSIM", "ssim_per_gene"),
                 ("SPCC", "spcc_per_gene"), ("MoransI", "morans_per_gene")] if v in s}

    pg_s1, pg_s2 = _pg(s1), _pg(s2)
    rep_s1 = stratified_report(mi_mask, mu_mask, mod_mask, pg_s1, baseline_pg_s1)
    rep_s2 = stratified_report(mi_mask, mu_mask, mod_mask, pg_s2, baseline_pg_s2)
    print_report(rep_s1, "GNN slice1 vs V0")
    print_report(rep_s2, "GNN slice2 vs V0")

    # ── Save ──
    def _strip(rep):
        return {k: v for k, v in rep.items() if k != "per_gene"}

    save = {
        "dataset": args.dataset, "seed": args.seed,
        "model": {"hidden": args.hidden, "n_layers": args.n_layers,
                  "n_params": n_params, "lambda_mu": args.lambda_mu,
                  "lambda_mod": args.lambda_mod, "coexpr_threshold": args.coexpr_threshold,
                  "ppi_weight": args.ppi_weight, "epochs_trained": args.epochs},
        "v0_baseline": {
            "slice1": {k: float(v) for k, v in
                       json.load((v0_dir / "metrics.json").open()).get("slice1", {}).items()
                       if isinstance(v, (int, float))} if (v0_dir / "metrics.json").exists() else {},
            "slice2": {k: float(v) for k, v in
                       json.load((v0_dir / "metrics.json").open()).get("slice2", {}).items()
                       if isinstance(v, (int, float))} if (v0_dir / "metrics.json").exists() else {},
        },
        "gnn_corrected": {
            "slice1": {k: float(v) for k, v in s1.items() if isinstance(v, (int, float))},
            "slice2": {k: float(v) for k, v in s2.items() if isinstance(v, (int, float))},
        },
        "stratified_s1": _strip(rep_s1),
        "stratified_s2": _strip(rep_s2),
        "train_time_seconds": train_time,
    }
    with (out_root / "metrics.json").open("w") as f:
        json.dump(save, f, indent=2)

    # ── Summary ──
    logger.info("\n" + "=" * 110)
    logger.info("SUMMARY — %s seed=%d (GeneGCN)", args.dataset, args.seed)
    logger.info("=" * 110)
    logger.info("%15s | %7s %7s | %9s %9s %9s %9s",
                "", "PCC_s1", "PCC_s2", "MU_dPCC", "MU_dSPCC", "MU_dMrnI", "MU_dSSIM")
    logger.info("-" * 110)
    v0s1 = save["v0_baseline"]["slice1"]
    v0s2 = save["v0_baseline"]["slice2"]
    logger.info("%15s | %7.4f %7.4f |       —         —         —         —",
                "V0_baseline", v0s1.get("PCC", 0), v0s2.get("PCC", 0))
    for slc, st_key in [("s1", "stratified_s1"), ("s2", "stratified_s2")]:
        mu = save[st_key].get("MU", {})
        ps = save["gnn_corrected"][f"slice{'1' if slc == 's1' else '2'}"]
        logger.info("%15s | %7.4f         | %+9.4f %+9.4f %+9.4f %+9.4f",
                    f"GNN_{slc}", ps["PCC"],
                    mu.get("dPCC_mean", 0), mu.get("dSPCC_mean", 0),
                    mu.get("dMoransI_mean", 0), mu.get("dSSIM_mean", 0))

    logger.info("Done. Results → %s", out_root)


if __name__ == "__main__":
    main()
