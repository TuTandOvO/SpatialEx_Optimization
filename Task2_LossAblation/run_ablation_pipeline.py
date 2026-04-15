#!/usr/bin/env python3
"""
SpatialEx 消融实验 (HPC) — 自包含脚本
==========================================
预处理只跑一次，然后自动跑所有消融变体，最后输出对比表格。

用法:
    python run_ablation_pipeline.py

每个变体约 30-60 min（500 epochs），4个变体 + baseline 共 5 轮训练。
全部跑完大约 2.5-5 小时。
"""

import numpy as np
import pandas as pd
import os
import sys
import json
import time
import logging
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from scipy import stats as scipy_stats

import SpatialEx as se
from SpatialEx.model import HGNN, Predictor_dgi
from SpatialEx.SpatialEx import SpatialEx
from SpatialEx.utils import create_optimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# =============================================
# Moran's I & SPCC 评估工具
# =============================================

def compute_morans_i_pred(true_expr, pred_expr, spatial_coords, k=7):
    """对预测表达计算 per-gene Moran's I。"""
    from sklearn.neighbors import NearestNeighbors
    import scipy.sparse as sp_sparse
    nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    nn.fit(spatial_coords)
    distances, indices = nn.kneighbors(spatial_coords)
    n = pred_expr.shape[0]
    rows, cols, vals = [], [], []
    for i in range(n):
        for j_idx in range(1, k):
            j = indices[i, j_idx]
            rows.append(i); cols.append(j); vals.append(1.0)
    adj = sp_sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    adj = adj + adj.T
    adj.data[:] = 1.0
    pred = np.asarray(pred_expr)
    if hasattr(pred, 'todense'):
        pred = np.asarray(pred.todense())
    x_bar = np.mean(pred, axis=0)
    x = pred - x_bar
    S0 = adj.sum()
    numerator = np.sum((adj @ x) * x, axis=0)
    denominator = np.sum(x ** 2, axis=0)
    morans_i = (n / S0) * (numerator / (denominator + 1e-6))
    return morans_i, float(np.nanmean(morans_i))


def compute_spcc(true_expr, pred_expr):
    """计算 per-gene Spearman's rank correlation coefficient (SPCC)。"""
    from scipy import stats as _stats
    true = np.asarray(true_expr)
    pred = np.asarray(pred_expr)
    if hasattr(true, 'todense'):
        true = np.asarray(true.todense())
    if hasattr(pred, 'todense'):
        pred = np.asarray(pred.todense())
    n_genes = true.shape[1]
    spcc = np.zeros(n_genes)
    for g in range(n_genes):
        rho, _ = _stats.spearmanr(true[:, g], pred[:, g])
        spcc[g] = rho if not np.isnan(rho) else 0.0
    return spcc, float(np.nanmean(spcc))

# =============================================
# 配置
# =============================================
device = 'cuda:1'
save_root1 = '/gpfsdata/home/renyixiang/YuanLab/data/MG_Xenium/Sample1_Rep1/Human_Breast_Cancer_Rep1/'
save_root2 = '/gpfsdata/home/renyixiang/YuanLab/data/MG_Xenium/Sample1_Rep2/Human_Breast_Cancer_Rep2/'
resolution = 64
image_encoder = 'resnet50'
num_neighbors = 7
epochs = 500

output_root = '/gpfsdata/home/renyixiang/YuanLab/data/MG_Xenium/SpatialEx_ablation/'
os.makedirs(output_root, exist_ok=True)

# ---- Skin Melanoma 数据配置（单切片 → 空间切分为两半）----
skin_root = '/gpfsdata/home/renyixiang/YuanLab/data/Skin_Xenium/Human_Skin_Melanoma_Base_FFPE/'
skin_output_root = '/gpfsdata/home/renyixiang/YuanLab/data/Skin_Xenium/SpatialEx_ablation/'
os.makedirs(skin_output_root, exist_ok=True)

# =============================================
# 消融实验定义
# =============================================
# 每个变体只改一个东西，其他保持原版 MSE
ABLATION_VARIANTS = {
    # 名称: (描述, loss_config)
    "0_baseline": (
        "原版 MSE（对照组）",
        None,  # None = 用原版 SpatialEx, 不用 ImprovedSpatialEx
    ),
    "1_cosine_only": (
        "MSE + Cosine Loss (beta=0.3) → 目标: 提升 PCC",
        {"alpha": 1.0, "beta": 0.3, "gamma": 0, "delta": 0,
         "use_gene_weights": False, "rank_tau": 1.0, "rank_max_genes": 300},
    ),
    "2_wmse_only": (
        "Weighted MSE (基因CV加权) → 目标: 提升 CMD",
        {"alpha": 1.0, "beta": 0, "gamma": 0, "delta": 0,
         "use_gene_weights": True, "rank_tau": 1.0, "rank_max_genes": 300},
    ),
    "3_smooth_only": (
        "MSE + Spatial Smoothness (delta=0.05) → 目标: 提升 CMD",
        {"alpha": 1.0, "beta": 0, "gamma": 0, "delta": 0.05,
         "use_gene_weights": False, "rank_tau": 1.0, "rank_max_genes": 300},
    ),
    "4_rank_only": (
        "MSE + Soft Rank Loss (gamma=0.05) → 目标: 提升 SSIM",
        {"alpha": 1.0, "beta": 0, "gamma": 0.05, "delta": 0,
         "use_gene_weights": False, "rank_tau": 1.0, "rank_max_genes": 300},
    ),
    "5_cosine_smooth": (
        "MSE + Cosine + Smooth (最有希望的组合)",
        {"alpha": 1.0, "beta": 0.3, "gamma": 0, "delta": 0.05,
         "use_gene_weights": True, "rank_tau": 1.0, "rank_max_genes": 300},
    ),
}


def spatial_split_adata(adata, axis=0):
    """沿空间坐标中位数将单切片分成两半（论文 Extended Data Fig. 2B 做法）。"""
    coords = adata.obsm['spatial']
    median_val = np.median(coords[:, axis])
    mask1 = coords[:, axis] <= median_val
    mask2 = coords[:, axis] > median_val
    adata1 = adata[mask1].copy()
    adata2 = adata[mask2].copy()
    print(f"[Spatial split] {adata.n_obs} cells → {adata1.n_obs} + {adata2.n_obs} (axis={axis}, median={median_val:.1f})")
    return adata1, adata2


def preprocess_skin_data():
    """预处理 Skin Melanoma 数据：读取 → 空间切分 → H&E patches → 建图。"""
    print("=" * 70)
    print("SKIN MELANOMA: Preprocessing (read → split → patches → graph)")
    print("=" * 70)
    adata = se.pp.Read_Xenium(
        skin_root + 'cell_feature_matrix.h5',
        skin_root + 'cells.csv')
    adata = se.pp.Preprocess_adata(adata)
    img, scale = se.pp.Read_HE_image(
        skin_root + 'Xenium_V1_hSkin_Melanoma_Base_FFPE_he_image.ome.tif')
    tmtx = pd.read_csv(
        skin_root + 'Xenium_V1_hSkin_Melanoma_Base_FFPE_he_imagealignment.csv',
        header=None).values
    adata = se.pp.Register_physical_to_pixel(adata, tmtx, scale=scale)
    print(f"[OK] Full skin adata: {adata.shape}")

    adata1, adata2 = spatial_split_adata(adata)

    he1, adata1 = se.pp.Tiling_HE_patches(resolution, adata1, img)
    adata1 = se.pp.Extract_HE_patches_representaion(
        he1, adata=adata1, image_encoder=image_encoder, device=device, store_key='he')
    del he1
    print(f"[OK] Skin slice 1: {adata1.shape}")

    he2, adata2 = se.pp.Tiling_HE_patches(resolution, adata2, img)
    adata2 = se.pp.Extract_HE_patches_representaion(
        he2, adata=adata2, image_encoder=image_encoder, device=device, store_key='he')
    del he2, img
    print(f"[OK] Skin slice 2: {adata2.shape}")

    g1 = se.pp.Build_hypergraph_spatial_and_HE(adata1, num_neighbors, graph_kind='spatial', return_type='csr')
    g2 = se.pp.Build_hypergraph_spatial_and_HE(adata2, num_neighbors, graph_kind='spatial', return_type='csr')
    print("[OK] Skin graphs built")
    return adata1, adata2, g1, g2


# =============================================
# Composite Loss 实现（与 run_improved_pipeline.py 相同）
# =============================================

def compute_gene_cv_weights(expression, eps=1e-8):
    if hasattr(expression, 'toarray'):
        expression = np.asarray(expression.todense())
    elif hasattr(expression, 'todense'):
        expression = np.asarray(expression.todense())
    expression = np.asarray(expression)
    mean = np.mean(expression, axis=0)
    std = np.std(expression, axis=0)
    cv = std / (np.abs(mean) + eps)
    weights = cv / (cv.mean() + eps)
    return weights.astype(np.float32)


class WeightedMSELoss(nn.Module):
    def __init__(self, gene_weights=None):
        super().__init__()
        if gene_weights is not None:
            self.register_buffer("gene_weights", gene_weights)
        else:
            self.gene_weights = None

    def forward(self, pred, target):
        diff_sq = (pred - target) ** 2
        if self.gene_weights is not None:
            diff_sq = diff_sq * self.gene_weights.unsqueeze(0)
        return diff_sq.mean()


class CosineLoss(nn.Module):
    def forward(self, pred, target):
        cos_sim = F.cosine_similarity(pred, target, dim=1)
        return 1.0 - cos_sim.mean()


class SoftRankLoss(nn.Module):
    def __init__(self, tau=1.0, max_genes=300):
        super().__init__()
        self.tau = tau
        self.max_genes = max_genes

    def _soft_rank(self, x):
        diff = x.unsqueeze(2) - x.unsqueeze(1)
        ranks = torch.sigmoid(diff / self.tau).sum(dim=2)
        return ranks

    def forward(self, pred, target):
        G = pred.shape[1]
        if G > self.max_genes:
            idx = torch.randperm(G, device=pred.device)[:self.max_genes]
            pred = pred[:, idx]
            target = target[:, idx]
        pred_ranks = self._soft_rank(pred)
        target_ranks = self._soft_rank(target)
        rank_corr = F.cosine_similarity(
            pred_ranks - pred_ranks.mean(dim=1, keepdim=True),
            target_ranks - target_ranks.mean(dim=1, keepdim=True),
            dim=1,
        )
        return 1.0 - rank_corr.mean()


class SpatialSmoothnessLoss(nn.Module):
    def forward(self, pred, graph, selection=None):
        smoothed = torch.sparse.mm(graph, pred)
        diff = pred - smoothed
        if selection is not None:
            diff = diff[selection]
        return (diff ** 2).mean()


class CompositeLoss(nn.Module):
    def __init__(self, gene_weights=None, alpha=1.0, beta=0.5, gamma=0.1,
                 delta=0.01, rank_tau=1.0, rank_max_genes=300, use_gene_weights=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        if use_gene_weights and gene_weights is not None:
            gw = torch.tensor(gene_weights, dtype=torch.float32)
        else:
            gw = None
        self.wmse = WeightedMSELoss(gw)
        self.cosine = CosineLoss()
        self.rank = SoftRankLoss(tau=rank_tau, max_genes=rank_max_genes)
        self.smooth = SpatialSmoothnessLoss()

    def forward(self, pred, target, graph=None, x_prime_raw=None, selection=None):
        loss = torch.tensor(0.0, device=pred.device)
        if self.alpha > 0:
            loss = loss + self.alpha * self.wmse(pred, target)
        if self.beta > 0:
            loss = loss + self.beta * self.cosine(pred, target)
        if self.gamma > 0:
            loss = loss + self.gamma * self.rank(pred, target)
        if self.delta > 0 and graph is not None and x_prime_raw is not None:
            loss = loss + self.delta * self.smooth(x_prime_raw, graph, selection)
        return loss


class ImprovedPredictor_spot(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers,
                 dropout=0.1, activation='prelu', composite_loss=None, agg=True):
        super().__init__()
        self.agg = agg
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LeakyReLU(0.1), nn.BatchNorm1d(hidden_dim),
        )
        self.mod = HGNN(in_dim=hidden_dim, num_hidden=hidden_dim, out_dim=hidden_dim,
                        num_layers=num_layers, dropout=0, activation=activation)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.composite_loss = composite_loss or CompositeLoss()

    def forward(self, graph, he_rep, x, agg_mtx=None, selection=None):
        he_rep = self.mlp(he_rep)
        enc = self.mod(he_rep, graph)
        x_prime = F.leaky_relu(self.linear(F.leaky_relu(enc)))
        if self.agg:
            pred = torch.sparse.mm(agg_mtx, x_prime[selection])
        else:
            pred = x_prime
        loss = self.composite_loss(pred=pred, target=x, graph=graph,
                                   x_prime_raw=x_prime, selection=selection)
        return loss, x_prime, enc

    def predict(self, graph, he_rep):
        he_rep = self.mlp(he_rep)
        enc = self.mod(he_rep, graph)
        x_prime = F.leaky_relu(self.linear(F.leaky_relu(enc)))
        return x_prime


class ImprovedModel(nn.Module):
    def __init__(self, num_layers=2, in_dim=2048, hidden_dim=512, out_dim=150,
                 composite_loss=None, device='cpu'):
        super().__init__()
        self.predictor = ImprovedPredictor_spot(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
            num_layers=num_layers, composite_loss=composite_loss)
        self.dgi_model = Predictor_dgi(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim)
        self.predictor.to(device)
        self.dgi_model.to(device)

    def forward(self, graph, he_rep, exp, agg_mtx, selection):
        loss_pre, x_prime, _ = self.predictor(graph, he_rep, exp, agg_mtx, selection)
        loss_dgi = self.dgi_model(graph, he_rep)
        return loss_pre + loss_dgi, x_prime

    def predict(self, he_representations, graph, grad=False):
        if not grad:
            with torch.no_grad():
                return self.predictor.predict(graph, he_representations)
        return self.predictor.predict(graph, he_representations)


class ImprovedSpatialEx(SpatialEx):
    def __init__(self, adata1, adata2, graph1, graph2, loss_config=None, **kwargs):
        super().__init__(adata1, adata2, graph1, graph2, **kwargs)
        loss_config = loss_config or {}
        use_gw = loss_config.pop("use_gene_weights", True)
        expr1 = np.asarray(adata1.X.todense()) if hasattr(adata1.X, 'toarray') else np.asarray(adata1.X)
        expr2 = np.asarray(adata2.X.todense()) if hasattr(adata2.X, 'toarray') else np.asarray(adata2.X)
        gw1 = compute_gene_cv_weights(expr1)
        gw2 = compute_gene_cv_weights(expr2)
        composite1 = CompositeLoss(gene_weights=gw1, use_gene_weights=use_gw, **loss_config)
        composite2 = CompositeLoss(gene_weights=gw2, use_gene_weights=use_gw, **loss_config)
        self.module_HA = ImprovedModel(
            num_layers=self.num_layers, in_dim=self.in_dim1,
            hidden_dim=self.hidden_dim, out_dim=self.out_dim1,
            composite_loss=composite1, device=self.device)
        self.module_HB = ImprovedModel(
            num_layers=self.num_layers, in_dim=self.in_dim2,
            hidden_dim=self.hidden_dim, out_dim=self.out_dim2,
            composite_loss=composite2, device=self.device)
        self.models = [self.module_HA, self.module_HB]
        self.optimizer = create_optimizer(
            kwargs.get("optimizer", "adam"), self.models, self.lr, self.weight_decay)


# =============================================
# 评估函数
# =============================================
def evaluate(adata1, adata2, panelB1, panelA2):
    """返回 dict: {slice1: {PCC, SSIM, CMD, MoransI, SPCC}, slice2: {...}}"""
    results = {}

    # Slice 1 预测
    graph_eval = se.pp.Build_graph(
        adata1.obsm['spatial'], graph_type='knn', weighted='gaussian',
        apply_normalize='row', return_type='coo')
    ssim1, ssim1_r = se.utils.Compute_metrics(adata1.X.copy(), panelB1.copy(), metric='ssim', graph=graph_eval, reduce='mean')
    pcc1, pcc1_r = se.utils.Compute_metrics(adata1.X.copy(), panelB1.copy(), metric='pcc', reduce='mean')
    cmd1, cmd1_r = se.utils.Compute_metrics(adata1.X.copy(), panelB1.copy(), metric='cmd', reduce='mean')
    morans1, morans1_r = compute_morans_i_pred(adata1.X, panelB1, adata1.obsm['spatial'])
    spcc1, spcc1_r = compute_spcc(adata1.X, panelB1)
    results['slice1'] = {
        'PCC': float(pcc1_r), 'SSIM': float(ssim1_r), 'CMD': float(cmd1_r),
        'MoransI': float(morans1_r), 'SPCC': float(spcc1_r),
        'pcc_per_gene': pcc1, 'ssim_per_gene': ssim1,
        'morans_per_gene': morans1, 'spcc_per_gene': spcc1,
    }

    # Slice 2 预测
    graph_eval = se.pp.Build_graph(
        adata2.obsm['spatial'], graph_type='knn', weighted='gaussian',
        apply_normalize='row', return_type='coo')
    ssim2, ssim2_r = se.utils.Compute_metrics(adata2.X.copy(), panelA2.copy(), metric='ssim', graph=graph_eval, reduce='mean')
    pcc2, pcc2_r = se.utils.Compute_metrics(adata2.X.copy(), panelA2.copy(), metric='pcc', reduce='mean')
    cmd2, cmd2_r = se.utils.Compute_metrics(adata2.X.copy(), panelA2.copy(), metric='cmd', reduce='mean')
    morans2, morans2_r = compute_morans_i_pred(adata2.X, panelA2, adata2.obsm['spatial'])
    spcc2, spcc2_r = compute_spcc(adata2.X, panelA2)
    results['slice2'] = {
        'PCC': float(pcc2_r), 'SSIM': float(ssim2_r), 'CMD': float(cmd2_r),
        'MoransI': float(morans2_r), 'SPCC': float(spcc2_r),
        'pcc_per_gene': pcc2, 'ssim_per_gene': ssim2,
        'morans_per_gene': morans2, 'spcc_per_gene': spcc2,
    }

    return results


# =============================================
# 1. 预处理（只跑一次！）
# =============================================
print("=" * 70)
print("STAGE 1: PREPROCESSING (only once for all ablation variants)")
print("=" * 70)

# Slice 1
file_path1 = save_root1 + 'cell_feature_matrix.h5'
obs_path1 = save_root1 + 'cells.csv'
img_path1 = save_root1 + 'Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif'
transform_mtx_path1 = save_root1 + 'Xenium_FFPE_Human_Breast_Cancer_Rep1_he_imagealignment.csv'

adata1 = se.pp.Read_Xenium(file_path1, obs_path1)
adata1 = se.pp.Preprocess_adata(adata1)
img, scale = se.pp.Read_HE_image(img_path1)
transform_mtx = pd.read_csv(transform_mtx_path1, header=None).values
adata1 = se.pp.Register_physical_to_pixel(adata1, transform_mtx, scale=scale)
he_patches, adata1 = se.pp.Tiling_HE_patches(resolution, adata1, img)
adata1 = se.pp.Extract_HE_patches_representaion(
    he_patches, adata=adata1, image_encoder=image_encoder, device=device, store_key='he')
del he_patches, img
print(f"[OK] Slice 1: {adata1.shape}")

# Slice 2
file_path2 = save_root2 + 'cell_feature_matrix.h5'
obs_path2 = save_root2 + 'cells.csv'
img_path2 = save_root2 + 'Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image.ome.tif'
transform_mtx_path2 = save_root2 + 'Xenium_FFPE_Human_Breast_Cancer_Rep2_he_imagealignment.csv'

adata2 = se.pp.Read_Xenium(file_path2, obs_path2)
adata2 = se.pp.Preprocess_adata(adata2)
img, scale = se.pp.Read_HE_image(img_path2)
transform_mtx = pd.read_csv(transform_mtx_path2, header=None).values
adata2 = se.pp.Register_physical_to_pixel(adata2, transform_mtx, scale=scale)
he_patches, adata2 = se.pp.Tiling_HE_patches(resolution, adata2, img)
adata2 = se.pp.Extract_HE_patches_representaion(
    he_patches, adata=adata2, image_encoder=image_encoder, store_key='he', device=device)
del he_patches, img
print(f"[OK] Slice 2: {adata2.shape}")

# Build graphs (reuse for all variants)
graph1 = se.pp.Build_hypergraph_spatial_and_HE(adata1, num_neighbors, graph_kind='spatial', return_type='csr')
graph2 = se.pp.Build_hypergraph_spatial_and_HE(adata2, num_neighbors, graph_kind='spatial', return_type='csr')
print("[OK] Graphs built")
print("Preprocessing complete!\n")


# =============================================
# 2. 跑所有消融变体
# =============================================
all_results = {}

for variant_name, (description, loss_config) in ABLATION_VARIANTS.items():
    print("\n" + "=" * 70)
    print(f"VARIANT: {variant_name}")
    print(f"  {description}")
    if loss_config is not None:
        active = [k for k, v in loss_config.items()
                  if k in ('beta', 'gamma', 'delta') and v > 0]
        active_str = ', '.join(active) if active else 'none (MSE only)'
        gw_str = 'ON' if loss_config.get('use_gene_weights', False) else 'OFF'
        print(f"  Active extra losses: {active_str}")
        print(f"  Gene weighting: {gw_str}")
    else:
        print("  Using original SpatialEx (pure MSE)")
    print("=" * 70)

    t_start = time.time()

    variant_dir = os.path.join(output_root, variant_name)
    os.makedirs(variant_dir, exist_ok=True)

    # --- Train ---
    if loss_config is None:
        # Baseline: 原版 SpatialEx
        trainer = se.SpatialEx(
            adata1, adata2, graph1, graph2,
            epochs=epochs, device=device)
    else:
        # Improved: 用组合损失
        trainer = ImprovedSpatialEx(
            adata1, adata2, graph1, graph2,
            epochs=epochs, device=device,
            loss_config=loss_config.copy())

    trainer.train()

    # --- Inference ---
    panelB1, panelA2 = trainer.auto_inference()

    # --- Evaluate ---
    results = evaluate(adata1, adata2, panelB1, panelA2)

    t_elapsed = time.time() - t_start

    print(f"\n  Results ({t_elapsed/60:.1f} min):")
    print(f"    Slice 1 — PCC: {results['slice1']['PCC']:.6f}  "
          f"SSIM: {results['slice1']['SSIM']:.6f}  "
          f"CMD: {results['slice1']['CMD']:.6f}  "
          f"MoransI: {results['slice1']['MoransI']:.6f}  "
          f"SPCC: {results['slice1']['SPCC']:.6f}")
    print(f"    Slice 2 — PCC: {results['slice2']['PCC']:.6f}  "
          f"SSIM: {results['slice2']['SSIM']:.6f}  "
          f"CMD: {results['slice2']['CMD']:.6f}  "
          f"MoransI: {results['slice2']['MoransI']:.6f}  "
          f"SPCC: {results['slice2']['SPCC']:.6f}")

    # Save per-variant results
    SCALAR_KEYS = ('PCC', 'SSIM', 'CMD', 'MoransI', 'SPCC')
    metrics_save = {
        'slice1': {k: v for k, v in results['slice1'].items() if k in SCALAR_KEYS},
        'slice2': {k: v for k, v in results['slice2'].items() if k in SCALAR_KEYS},
        'loss_config': loss_config,
        'description': description,
        'time_seconds': t_elapsed,
    }
    with open(os.path.join(variant_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_save, f, indent=2)

    # Save per-gene arrays for significance tests later
    for sid, s_key in [(1, 'slice1'), (2, 'slice2')]:
        for m_name in ['pcc', 'ssim', 'morans', 'spcc']:
            arr_key = f'{m_name}_per_gene'
            if arr_key in results[s_key]:
                np.save(os.path.join(variant_dir, f'slice{sid}_{m_name}_per_gene.npy'), results[s_key][arr_key])
    np.save(os.path.join(variant_dir, 'panelB1.npy'), panelB1)
    np.save(os.path.join(variant_dir, 'panelA2.npy'), panelA2)

    all_results[variant_name] = metrics_save

    # 清理 GPU 内存
    del trainer, panelB1, panelA2
    torch.cuda.empty_cache()
    gc.collect()


# =============================================
# 3. 汇总对比表
# =============================================
print("\n\n" + "=" * 100)
print("ABLATION RESULTS SUMMARY")
print("=" * 100)

# 表头
header = (f"{'Variant':<22} {'Description':<40} {'PCC_s1':>8} {'PCC_s2':>8} {'SSIM_s1':>8} {'SSIM_s2':>8} "
          f"{'CMD_s1':>8} {'CMD_s2':>8} {'MI_s1':>8} {'MI_s2':>8} {'SPCC_s1':>8} {'SPCC_s2':>8}")
print(header)
print("-" * len(header))

# Baseline 结果（用于计算 diff）
bl = all_results.get('0_baseline', {})
bl_s1 = bl.get('slice1', {})
bl_s2 = bl.get('slice2', {})

for name, res in all_results.items():
    s1 = res['slice1']
    s2 = res['slice2']
    desc = res['description'][:38]
    print(f"{name:<22} {desc:<40} "
          f"{s1['PCC']:>8.4f} {s2['PCC']:>8.4f} "
          f"{s1['SSIM']:>8.4f} {s2['SSIM']:>8.4f} "
          f"{s1['CMD']:>8.4f} {s2['CMD']:>8.4f} "
          f"{s1.get('MoransI',0):>8.4f} {s2.get('MoransI',0):>8.4f} "
          f"{s1.get('SPCC',0):>8.4f} {s2.get('SPCC',0):>8.4f}")

# Diff vs baseline
if bl_s1:
    print("\n" + "-" * len(header))
    print("DIFF vs BASELINE (positive = better for PCC/SSIM, negative = better for CMD):")
    print("-" * len(header))

    for name, res in all_results.items():
        if name == '0_baseline':
            continue
        s1 = res['slice1']
        s2 = res['slice2']

        d_pcc1 = s1['PCC'] - bl_s1['PCC']
        d_pcc2 = s2['PCC'] - bl_s2['PCC']
        d_ssim1 = s1['SSIM'] - bl_s1['SSIM']
        d_ssim2 = s2['SSIM'] - bl_s2['SSIM']
        d_cmd1 = s1['CMD'] - bl_s1['CMD']
        d_cmd2 = s2['CMD'] - bl_s2['CMD']

        # 判断方向
        def arrow(val, higher_better=True):
            if higher_better:
                return '↑' if val > 0.001 else ('↓' if val < -0.001 else '~')
            else:
                return '↑' if val < -0.001 else ('↓' if val > 0.001 else '~')

        print(f"{name:<22} "
              f"{d_pcc1:>+8.4f}{arrow(d_pcc1)} {d_pcc2:>+8.4f}{arrow(d_pcc2)} "
              f"{d_ssim1:>+8.4f}{arrow(d_ssim1)} {d_ssim2:>+8.4f}{arrow(d_ssim2)} "
              f"{d_cmd1:>+8.4f}{arrow(d_cmd1, False)} {d_cmd2:>+8.4f}{arrow(d_cmd2, False)}")

# =============================================
# 4. Wilcoxon 显著性检验（每个变体 vs baseline）
# =============================================
print("\n\n" + "=" * 100)
print("STATISTICAL SIGNIFICANCE (Wilcoxon signed-rank, two-sided)")
print("=" * 100)

from scipy import stats

bl_dir = os.path.join(output_root, '0_baseline')

for name in ABLATION_VARIANTS:
    if name == '0_baseline':
        continue

    variant_dir = os.path.join(output_root, name)
    print(f"\n{name}:")

    for slice_id in [1, 2]:
        for metric in ['pcc', 'ssim']:
            bl_path = os.path.join(bl_dir, f'slice{slice_id}_{metric}_per_gene.npy')
            im_path = os.path.join(variant_dir, f'slice{slice_id}_{metric}_per_gene.npy')

            if os.path.exists(bl_path) and os.path.exists(im_path):
                bl_vals = np.load(bl_path)
                im_vals = np.load(im_path)
                min_len = min(len(bl_vals), len(im_vals))

                # 去掉 NaN
                mask = ~(np.isnan(bl_vals[:min_len]) | np.isnan(im_vals[:min_len]))
                bl_clean = bl_vals[:min_len][mask]
                im_clean = im_vals[:min_len][mask]

                if len(bl_clean) > 10:
                    try:
                        stat, pval = stats.wilcoxon(im_clean, bl_clean)
                        mean_diff = np.mean(im_clean - bl_clean)
                        sig = '*** p<0.001' if pval < 0.001 else ('** p<0.01' if pval < 0.01 else ('* p<0.05' if pval < 0.05 else 'n.s.'))
                        print(f"  Slice {slice_id} {metric.upper()}: mean_diff={mean_diff:+.6f}, "
                              f"p={pval:.4e} {sig}")
                    except Exception as e:
                        print(f"  Slice {slice_id} {metric.upper()}: test failed ({e})")

# Save full summary
with open(os.path.join(output_root, 'ablation_summary.json'), 'w') as f:
    # Remove numpy arrays before saving
    save_results = {}
    SCALAR_KEYS2 = ('PCC', 'SSIM', 'CMD', 'MoransI', 'SPCC')
    for name, res in all_results.items():
        save_results[name] = {
            'slice1': {k: v for k, v in res['slice1'].items() if k in SCALAR_KEYS2},
            'slice2': {k: v for k, v in res['slice2'].items() if k in SCALAR_KEYS2},
            'description': res['description'],
            'loss_config': res.get('loss_config'),
            'time_seconds': res.get('time_seconds'),
        }
    json.dump(save_results, f, indent=2)

print(f"\n\n[SAVED] All results to: {output_root}")
print("Files per variant: metrics.json, panelB1.npy, panelA2.npy, *_per_gene.npy")
print("Summary: ablation_summary.json")
print("\n" + "=" * 70)
print("ABLATION COMPLETE!")
print("=" * 70)


# =============================================
# PHASE 2: Skin Melanoma Dataset（空间切分）
# =============================================
print("\n\n" + "#" * 70)
print("# PHASE 2: SKIN MELANOMA DATASET (spatial split)")
print("#" * 70)

skin_adata1, skin_adata2, skin_graph1, skin_graph2 = preprocess_skin_data()

skin_all_results = {}

for variant_name, (description, loss_config) in ABLATION_VARIANTS.items():
    print(f"\n{'=' * 70}")
    print(f"[SKIN] VARIANT: {variant_name} — {description}")
    print('=' * 70)

    t_start = time.time()
    variant_dir = os.path.join(skin_output_root, variant_name)
    os.makedirs(variant_dir, exist_ok=True)

    if loss_config is None:
        trainer = se.SpatialEx(
            skin_adata1, skin_adata2, skin_graph1, skin_graph2,
            epochs=epochs, device=device)
    else:
        trainer = ImprovedSpatialEx(
            skin_adata1, skin_adata2, skin_graph1, skin_graph2,
            epochs=epochs, device=device, loss_config=loss_config.copy())

    trainer.train()
    panelB1, panelA2 = trainer.auto_inference()
    results = evaluate(skin_adata1, skin_adata2, panelB1, panelA2)
    t_elapsed = time.time() - t_start

    print(f"  Slice 1 — PCC: {results['slice1']['PCC']:.6f}  "
          f"SSIM: {results['slice1']['SSIM']:.6f}  CMD: {results['slice1']['CMD']:.6f}  "
          f"MoransI: {results['slice1']['MoransI']:.6f}  SPCC: {results['slice1']['SPCC']:.6f}")
    print(f"  Slice 2 — PCC: {results['slice2']['PCC']:.6f}  "
          f"SSIM: {results['slice2']['SSIM']:.6f}  CMD: {results['slice2']['CMD']:.6f}  "
          f"MoransI: {results['slice2']['MoransI']:.6f}  SPCC: {results['slice2']['SPCC']:.6f}")
    print(f"  Time: {t_elapsed/60:.1f} min")

    SKIN_SCALAR_KEYS = ('PCC', 'SSIM', 'CMD', 'MoransI', 'SPCC')
    metrics_save = {
        'slice1': {k: v for k, v in results['slice1'].items() if k in SKIN_SCALAR_KEYS},
        'slice2': {k: v for k, v in results['slice2'].items() if k in SKIN_SCALAR_KEYS},
        'loss_config': loss_config, 'description': description, 'time_seconds': t_elapsed,
    }
    with open(os.path.join(variant_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_save, f, indent=2)

    for sid, s_key in [(1, 'slice1'), (2, 'slice2')]:
        for m_name in ['pcc', 'ssim', 'morans', 'spcc']:
            arr_key = f'{m_name}_per_gene'
            if arr_key in results[s_key]:
                np.save(os.path.join(variant_dir, f'slice{sid}_{m_name}_per_gene.npy'), results[s_key][arr_key])
    np.save(os.path.join(variant_dir, 'panelB1.npy'), panelB1)
    np.save(os.path.join(variant_dir, 'panelA2.npy'), panelA2)

    skin_all_results[variant_name] = metrics_save

    del trainer, panelB1, panelA2
    torch.cuda.empty_cache()
    gc.collect()

# Skin summary table
print("\n\n" + "=" * 100)
print("SKIN MELANOMA ABLATION RESULTS SUMMARY")
print("=" * 100)
header = f"{'Variant':<22} {'PCC_s1':>8} {'PCC_s2':>8} {'SSIM_s1':>8} {'SSIM_s2':>8} {'CMD_s1':>8} {'CMD_s2':>8}"
print(header)
print("-" * len(header))

skin_bl = skin_all_results.get('0_baseline', {})
for name, res in skin_all_results.items():
    s1, s2 = res['slice1'], res['slice2']
    print(f"{name:<22} {s1['PCC']:>8.4f} {s2['PCC']:>8.4f} "
          f"{s1['SSIM']:>8.4f} {s2['SSIM']:>8.4f} {s1['CMD']:>8.4f} {s2['CMD']:>8.4f}")

if skin_bl:
    print("\nDIFF vs BASELINE:")
    for name, res in skin_all_results.items():
        if name == '0_baseline':
            continue
        s1, s2 = res['slice1'], res['slice2']
        b1, b2 = skin_bl['slice1'], skin_bl['slice2']
        print(f"{name:<22} {s1['PCC']-b1['PCC']:>+8.4f} {s2['PCC']-b2['PCC']:>+8.4f} "
              f"{s1['SSIM']-b1['SSIM']:>+8.4f} {s2['SSIM']-b2['SSIM']:>+8.4f} "
              f"{s1['CMD']-b1['CMD']:>+8.4f} {s2['CMD']-b2['CMD']:>+8.4f}")

# Wilcoxon for skin
skin_bl_dir = os.path.join(skin_output_root, '0_baseline')
for name in ABLATION_VARIANTS:
    if name == '0_baseline':
        continue
    v_dir = os.path.join(skin_output_root, name)
    print(f"\n[SKIN] {name}:")
    for sid in [1, 2]:
        for m in ['pcc', 'ssim']:
            bp = os.path.join(skin_bl_dir, f'slice{sid}_{m}_per_gene.npy')
            vp = os.path.join(v_dir, f'slice{sid}_{m}_per_gene.npy')
            if os.path.exists(bp) and os.path.exists(vp):
                bv, vv = np.load(bp), np.load(vp)
                ml = min(len(bv), len(vv))
                mask = ~(np.isnan(bv[:ml]) | np.isnan(vv[:ml]))
                bc, vc = bv[:ml][mask], vv[:ml][mask]
                if len(bc) > 10:
                    try:
                        stat, pval = stats.wilcoxon(vc, bc)
                        md = np.mean(vc - bc)
                        sig = '*** p<0.001' if pval < 0.001 else '** p<0.01' if pval < 0.01 else '* p<0.05' if pval < 0.05 else 'n.s.'
                        print(f"  Slice {sid} {m.upper()}: mean_diff={md:+.6f}, p={pval:.4e} {sig}")
                    except Exception as e:
                        print(f"  Slice {sid} {m.upper()}: test failed ({e})")

with open(os.path.join(skin_output_root, 'ablation_summary.json'), 'w') as f:
    json.dump(skin_all_results, f, indent=2)

print(f"\n[SAVED] Skin results to: {skin_output_root}")
print("=" * 70)
print("ALL DATASETS COMPLETE!")
print("=" * 70)
