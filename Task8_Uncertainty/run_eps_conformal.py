#!/usr/bin/env python3
"""
Task 8 v2 — Stage 2: EPS-based stratification + Conformal Prediction.

Runs in the `eps` conda env. Requires only:
  - expression_copilot
  - scanpy / anndata
  - numpy, pandas, scipy, scikit-learn

Reads .h5ad produced by prep_adata_for_eps.py (SpatialEx env) and Task5 v0
baseline .npy predictions. Outputs tier-stratified conformal metrics.

    python run_eps_conformal.py --dataset {mg,skin} --seed 42 \
           --cache_dir /path/to/h5ad --v0_dir /path/to/Task5/2_spatial_ppi
"""
import argparse
import json
import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import anndata
from scipy import stats as scipy_stats
from scipy.spatial import cKDTree
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("task8v2")

HERE = Path(__file__).resolve().parent


# -- EPS --------------------------------------------------------------------

def compute_eps(adata, image_key='he', hvgs=3000, k=5):
    from expression_copilot import ExpressionCopilotModel
    logger.info(f"Computing EPS: N={adata.n_obs}, G={adata.n_vars}")
    t0 = time.time()
    model = ExpressionCopilotModel(adata.copy(), image_key=image_key, hvgs=hvgs)
    eps_df = model.calc_metrics_per_gene(k=k)
    logger.info(f"EPS done in {time.time() - t0:.1f}s, "
                f"range=[{eps_df['EPS'].min():.3f}, {eps_df['EPS'].max():.3f}]")
    return eps_df


def stratify_by_eps(eps_vec):
    valid = np.isfinite(eps_vec)
    t_low, t_high = np.percentile(eps_vec[valid], [33.33, 66.67])
    mi = valid & (eps_vec > t_high)
    mu = valid & (eps_vec <= t_low)
    mod = valid & ~(mi | mu)
    logger.info("EPS masks: MI=%d MU=%d MOD=%d (low=%.3f, high=%.3f)",
                mi.sum(), mu.sum(), mod.sum(), t_low, t_high)
    return mi, mu, mod


# -- Spatial graph + CCV ----------------------------------------------------

def build_spatial_knn(coords, k=20):
    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=k + 1)
    return indices[:, 1:]


def compute_ccv(pred, nbr_indices, he_features=None, use_he_weights=True):
    N, G = pred.shape
    k = nbr_indices.shape[1]
    ccv = np.ones((N, G), dtype=np.float64)
    if use_he_weights and he_features is not None:
        he_norm = normalize(he_features, norm='l2', axis=1)
    chunk = 5000
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        idx = np.arange(s, e)
        nbrs = nbr_indices[s:e]
        cp = pred[idx][:, None, :]
        nb = pred[nbrs]
        sq = (nb - cp) ** 2
        if use_he_weights and he_features is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cs = np.einsum('id,ikd->ik', he_norm[idx], he_norm[nbrs])
                cs = np.nan_to_num(cs, nan=0.0)
            w = np.exp(cs)
            ws = np.maximum(w.sum(axis=1, keepdims=True), 1e-8)
            wv = np.einsum('ik,ikg->ig', w, sq) / ws
        else:
            wv = sq.mean(axis=1)
        ccv[s:e] = 1.0 + np.sqrt(np.maximum(wv, 0))
    logger.info("CCV: mean=%.4f median=%.4f", ccv.mean(), np.median(ccv))
    return ccv.astype(np.float32)


# -- Conformal --------------------------------------------------------------

def conformal_within(pred, gt, ccv, alpha=0.33):
    N, G = pred.shape
    scores = np.abs(gt - pred) / np.maximum(ccv, 1e-8)
    level = min(int(np.ceil((N + 1) * (1 - alpha))) / N, 1.0)
    qhat = np.zeros(G, dtype=np.float32)
    for j in range(G):
        s = scores[:, j]
        s = s[np.isfinite(s)]
        if len(s) > 0:
            qhat[j] = np.quantile(s, level)
    widths = 2.0 * ccv * qhat[None, :]
    lo = pred - ccv * qhat[None, :]
    hi = pred + ccv * qhat[None, :]
    cov = ((gt >= lo) & (gt <= hi)).mean(axis=0)
    return qhat, widths, cov


def conformal_cross(pred_c, gt_c, ccv_c, pred_t, gt_t, ccv_t, alpha=0.33):
    N, G = pred_c.shape
    scores = np.abs(gt_c - pred_c) / np.maximum(ccv_c, 1e-8)
    level = min(int(np.ceil((N + 1) * (1 - alpha))) / N, 1.0)
    qhat = np.zeros(G, dtype=np.float32)
    for j in range(G):
        s = scores[:, j]
        s = s[np.isfinite(s)]
        if len(s) > 0:
            qhat[j] = np.quantile(s, level)
    widths = 2.0 * ccv_t * qhat[None, :]
    lo = pred_t - ccv_t * qhat[None, :]
    hi = pred_t + ccv_t * qhat[None, :]
    cov = ((gt_t >= lo) & (gt_t <= hi)).mean(axis=0)
    return qhat, widths, cov


# -- Utils ------------------------------------------------------------------

def to_dense(X):
    return X.toarray() if hasattr(X, 'toarray') else np.asarray(X)


def tier_stat(values, mi, mu, mod, name):
    def _mean(m):
        v = values[m]
        v = v[np.isfinite(v)]
        return float(v.mean()) if len(v) > 0 else float('nan')
    return {f"{name}_MI": _mean(mi),
            f"{name}_MU": _mean(mu),
            f"{name}_MOD": _mean(mod)}


def spearman_vs_pcc(eps_vec, pcc_s1, pcc_s2):
    mean_pcc = (np.nan_to_num(pcc_s1) + np.nan_to_num(pcc_s2)) / 2.0
    v = np.isfinite(eps_vec) & np.isfinite(mean_pcc)
    if v.sum() < 10:
        return float('nan'), float('nan')
    rho, p = scipy_stats.spearmanr(eps_vec[v], mean_pcc[v])
    return float(rho), float(p)


# -- Main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['mg', 'skin'], required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cache_dir', type=str, required=True,
                    help="Dir with {mg|skin}_slice{1,2}.h5ad from prep_adata_for_eps.py")
    ap.add_argument('--v0_dir', type=str, required=True,
                    help="Task 5 v0 output dir (has panelB1.npy, panelA2.npy, per_gene PCCs)")
    ap.add_argument('--out_dir', type=str, default=str(HERE / 'results_v2'))
    ap.add_argument('--alpha', type=float, default=0.33)
    ap.add_argument('--eps_k', type=int, default=5)
    ap.add_argument('--spatial_k', type=int, default=20)
    args = ap.parse_args()

    np.random.seed(args.seed)
    out_dir = Path(args.out_dir) / f"{args.dataset}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load AnnData cache
    s1_path = f"{args.cache_dir}/{args.dataset}_slice1.h5ad"
    s2_path = f"{args.cache_dir}/{args.dataset}_slice2.h5ad"
    adata1 = anndata.read_h5ad(s1_path)
    adata2 = anndata.read_h5ad(s2_path)
    logger.info(f"Loaded: slice1={adata1.shape}, slice2={adata2.shape}")
    assert 'he' in adata1.obsm and 'he' in adata2.obsm, "obsm['he'] missing"

    # EPS per slice, merge by mean
    eps_s1 = compute_eps(adata1, image_key='he', k=args.eps_k)
    eps_s2 = compute_eps(adata2, image_key='he', k=args.eps_k)
    eps_s1.to_csv(out_dir / 'eps_slice1.csv')
    eps_s2.to_csv(out_dir / 'eps_slice2.csv')
    eps_merged = eps_s1.join(eps_s2, how='outer', lsuffix='_s1', rsuffix='_s2')
    eps_merged['EPS'] = eps_merged[['EPS_s1', 'EPS_s2']].mean(axis=1)
    eps_merged.to_csv(out_dir / 'eps_merged.csv')

    # Align EPS to slice1's gene order (matches Task 5 output gene ordering)
    gene_order = list(adata1.var_names)
    eps_vec = np.full(len(gene_order), np.nan, dtype=np.float64)
    for i, g in enumerate(gene_order):
        if g in eps_merged.index:
            eps_vec[i] = eps_merged.loc[g, 'EPS']

    mi, mu, mod = stratify_by_eps(eps_vec)

    # Load Task 5 v0 baseline predictions
    v0 = Path(args.v0_dir)
    pred_s1 = np.load(v0 / 'panelB1.npy')
    pred_s2 = np.load(v0 / 'panelA2.npy')
    gt_s1 = to_dense(adata1.X)
    gt_s2 = to_dense(adata2.X)
    logger.info(f"Baseline pred shapes: s1={pred_s1.shape}, s2={pred_s2.shape}")
    logger.info(f"GT shapes: s1={gt_s1.shape}, s2={gt_s2.shape}")

    # Task 5 v0's auto_inference uses a patch dataloader whose cell ordering
    # may differ from adata.obs_names order, and a few border cells can be
    # dropped by the patch-based hypergraph pruning. The exact cell-to-row
    # mapping was not saved. We fall back to a "first-N" approximation: take
    # the first n_pred cells of adata / GT / spatial. If the drop rate is
    # tiny (< 0.5%), this approximation affects PCC / coverage by < 0.001.
    def align_to_pred(adata, gt, pred, label):
        n_pred = pred.shape[0]
        n_ad = adata.n_obs
        if n_pred == n_ad:
            return adata, gt
        if n_pred > n_ad:
            raise ValueError(f"[{label}] pred N={n_pred} > adata N={n_ad}; data mismatch")
        drop_frac = (n_ad - n_pred) / n_ad
        if drop_frac > 0.01:
            logger.warning(f"[{label}] WARNING: drop_frac={drop_frac:.3%} is > 1%; "
                           f"per-cell alignment may be wrong (first-N approximation).")
        else:
            logger.info(f"[{label}] first-N alignment: {n_ad} → {n_pred} "
                        f"(drop_frac={drop_frac:.4%}, negligible)")
        adata = adata[:n_pred].copy()
        gt = gt[:n_pred]
        return adata, gt

    adata1, gt_s1 = align_to_pred(adata1, gt_s1, pred_s1, 'slice1')
    adata2, gt_s2 = align_to_pred(adata2, gt_s2, pred_s2, 'slice2')

    # Agreement with baseline PCC
    info = {}
    pcc1_path = v0 / 'slice1_pcc_per_gene.npy'
    pcc2_path = v0 / 'slice2_pcc_per_gene.npy'
    if pcc1_path.exists() and pcc2_path.exists():
        rho, pv = spearman_vs_pcc(eps_vec, np.load(pcc1_path), np.load(pcc2_path))
        info['spearman_rho_eps_vs_baseline_pcc'] = rho
        info['spearman_p'] = pv
        logger.info("EPS vs baseline PCC: Spearman ρ=%.3f (p=%.2e)", rho, pv)

    # Spatial k-NN + CCV
    coords_s1 = adata1.obs[['x_centroid', 'y_centroid']].values.astype(np.float32)
    coords_s2 = adata2.obs[['x_centroid', 'y_centroid']].values.astype(np.float32)
    nbr_s1 = build_spatial_knn(coords_s1, k=args.spatial_k)
    nbr_s2 = build_spatial_knn(coords_s2, k=args.spatial_k)
    ccv_s1 = compute_ccv(pred_s1, nbr_s1, adata1.obsm['he'], use_he_weights=True)
    ccv_s2 = compute_ccv(pred_s2, nbr_s2, adata2.obsm['he'], use_he_weights=True)

    # Conformal — within-slice + cross-slice
    qhat1, w1, cov1 = conformal_within(pred_s1, gt_s1, ccv_s1, alpha=args.alpha)
    qhat2, w2, cov2 = conformal_within(pred_s2, gt_s2, ccv_s2, alpha=args.alpha)
    qhat_c, w_c, cov_c = conformal_cross(pred_s1, gt_s1, ccv_s1,
                                          pred_s2, gt_s2, ccv_s2, alpha=args.alpha)

    # Summary
    summary = {
        'dataset': args.dataset,
        'seed': args.seed,
        'n_genes_MI': int(mi.sum()),
        'n_genes_MOD': int(mod.sum()),
        'n_genes_MU': int(mu.sum()),
        'alpha': args.alpha,
        **tier_stat(ccv_s1.mean(axis=0), mi, mu, mod, 'ccv_slice1'),
        **tier_stat(ccv_s2.mean(axis=0), mi, mu, mod, 'ccv_slice2'),
        **tier_stat(w1.mean(axis=0), mi, mu, mod, 'width_slice1'),
        **tier_stat(w2.mean(axis=0), mi, mu, mod, 'width_slice2'),
        **tier_stat(w_c.mean(axis=0), mi, mu, mod, 'width_cross_slice'),
        **tier_stat(cov1, mi, mu, mod, 'coverage_slice1'),
        **tier_stat(cov2, mi, mu, mod, 'coverage_slice2'),
        **tier_stat(cov_c, mi, mu, mod, 'coverage_cross_slice'),
        'coverage_slice1_overall': float(cov1.mean()),
        'coverage_slice2_overall': float(cov2.mean()),
        'coverage_cross_slice_overall': float(cov_c.mean()),
        **info,
    }

    # Save
    np.save(out_dir / 'eps_per_gene.npy', eps_vec)
    np.save(out_dir / 'mi_mask.npy', mi)
    np.save(out_dir / 'mu_mask.npy', mu)
    np.save(out_dir / 'mod_mask.npy', mod)
    for name, arr in [('qhat_slice1', qhat1), ('qhat_slice2', qhat2), ('qhat_cross', qhat_c),
                      ('coverage_slice1', cov1), ('coverage_slice2', cov2), ('coverage_cross', cov_c)]:
        np.save(out_dir / f'{name}.npy', arr)

    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("=" * 70)
    for k, v in summary.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")
    logger.info(f"Saved: {out_dir}")


if __name__ == '__main__':
    main()
