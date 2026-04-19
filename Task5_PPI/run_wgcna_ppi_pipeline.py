#!/usr/bin/env python3
"""
Multi-source hypergraph for SpatialEx:
  H_combined = α · H_spatial + β · H_coexpr(WGCNA) + γ · H_ppi(HumanBase)

Run download_ppi_network.py first to fetch the tissue-specific PPI matrix.
Empirical: PPI is the only source that helps; WGCNA was neutral-to-harmful
and added nothing on top of PPI.
"""

import os
import sys
import json
import logging
import time
import gc

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy import stats as scipy_stats

import SpatialEx as se
from SpatialEx.model import HGNN, Predictor_dgi, Model
from SpatialEx.SpatialEx import SpatialEx
from SpatialEx.utils import create_optimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# Moran's I & SPCC 评估工具

def compute_morans_i_pred(true_expr, pred_expr, spatial_coords, k=7):
    """对预测表达计算 per-gene Moran's I。"""
    from sklearn.neighbors import NearestNeighbors
    import scipy.sparse as sp_sparse2
    nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    nn.fit(spatial_coords)
    distances, indices = nn.kneighbors(spatial_coords)
    n = pred_expr.shape[0]
    rows, cols, vals = [], [], []
    for i in range(n):
        for j_idx in range(1, k):
            j = indices[i, j_idx]
            rows.append(i); cols.append(j); vals.append(1.0)
    adj = sp_sparse2.csr_matrix((vals, (rows, cols)), shape=(n, n))
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
    true = np.asarray(true_expr)
    pred = np.asarray(pred_expr)
    if hasattr(true, 'todense'):
        true = np.asarray(true.todense())
    if hasattr(pred, 'todense'):
        pred = np.asarray(pred.todense())
    n_genes = true.shape[1]
    spcc = np.zeros(n_genes)
    for g in range(n_genes):
        rho, _ = scipy_stats.spearmanr(true[:, g], pred[:, g])
        spcc[g] = rho if not np.isnan(rho) else 0.0
    return spcc, float(np.nanmean(spcc))

# 配置
device = 'cuda:1'
save_root1 = '/gpfsdata/home/renyixiang/YuanLab/data/MG_Xenium/Sample1_Rep1/Human_Breast_Cancer_Rep1/'
save_root2 = '/gpfsdata/home/renyixiang/YuanLab/data/MG_Xenium/Sample1_Rep2/Human_Breast_Cancer_Rep2/'
resolution = 64
image_encoder = 'resnet50'
num_neighbors = 7
epochs = 500

output_dir = '/gpfsdata/home/renyixiang/YuanLab/data/MG_Xenium/SpatialEx_results_wgcna_ppi_MG/'
OVERWRITE = False  # True = 覆盖已有结果；False = 跳过已完成的变体
os.makedirs(output_dir, exist_ok=True)

baseline_dir = '/gpfsdata/home/renyixiang/YuanLab/data/MG_Xenium/SpatialEx_results/'

# ---- Skin Melanoma 数据配置（单切片 → 空间切分为两半）----
skin_root = '/gpfsdata/home/renyixiang/YuanLab/data/Skin_Xenium/Human_Skin_Melanoma_Base_FFPE/'
output_dir_skin = '/gpfsdata/home/renyixiang/YuanLab/data/Skin_Xenium/SpatialEx_results_wgcna_ppi_skin/'
os.makedirs(output_dir_skin, exist_ok=True)

# PPI 数据路径（由 download_ppi_network.py 生成）
PPI_DATA_DIR = '/gpfsdata/home/renyixiang/YuanLab/Task6_wgcnaPPI/ppi_data_MG/'
# Skin PPI 数据路径（使用 --tissue skin 下载）
PPI_DATA_DIR_SKIN = '/gpfsdata/home/renyixiang/YuanLab/Task6_wgcnaPPI/ppi_data_skin/'

# 超图融合权重
ALPHA_SPATIAL = 1.0   # 空间超图权重（原版）
BETA_COEXPR = 0.3     # 共表达模块超图权重
GAMMA_PPI = 0.3       # PPI 超图权重

# WGCNA 参数
WGCNA_SOFT_POWER = 6        # soft thresholding 幂次（默认 6, 可调 4-12）
WGCNA_MIN_MODULE_SIZE = 10  # 最小模块大小
WGCNA_MERGE_CUT_HEIGHT = 0.25  # 合并相似模块的切割高度
COEXPR_K = 7                 # 共表达空间 k-NN 的 k

# PPI 超图参数
PPI_THRESHOLD = 0.1   # PPI 权重阈值（低于此值忽略）
PPI_K = 7             # PPI-weighted 空间 k-NN 的 k


# A. WGCNA-Lite: 基因共表达模块检测

def compute_soft_adjacency(expression: np.ndarray, power: int = 6) -> np.ndarray:
    """计算 WGCNA-style soft adjacency matrix。

    a_ij = |cor(gene_i, gene_j)|^power

    Args:
        expression: [N, G] 基因表达矩阵（行=细胞，列=基因）。
        power: soft thresholding 幂次。

    Returns:
        [G, G] soft adjacency 矩阵。
    """
    # 基因-基因 Pearson 相关（G×G）
    G = expression.shape[1]
    cor_matrix = np.corrcoef(expression.T)  # [G, G]
    cor_matrix = np.nan_to_num(cor_matrix, nan=0.0)

    # Soft thresholding: |r|^power
    adj = np.abs(cor_matrix) ** power
    np.fill_diagonal(adj, 0)

    logger.info("Soft adjacency (power=%d): mean=%.4f, max=%.4f",
                 power, adj.mean(), adj[adj > 0].max() if adj.max() > 0 else 0)
    return adj.astype(np.float32)


def compute_tom(adj: np.ndarray) -> np.ndarray:
    """计算 Topological Overlap Matrix (TOM)。

    TOM_ij = (sum_k(a_ik * a_kj) + a_ij) / (min(k_i, k_j) + 1 - a_ij)

    TOM 衡量两个基因共享邻居的程度，比 correlation 更鲁棒。

    Args:
        adj: [G, G] soft adjacency 矩阵。

    Returns:
        [G, G] TOM 矩阵。
    """
    G = adj.shape[0]
    # 节点连接度
    k = adj.sum(axis=1)  # [G]

    # L = A @ A: L_ij = sum_k(a_ik * a_kj)
    L = adj @ adj  # [G, G]

    # TOM
    k_min = np.minimum(k[:, None], k[None, :])  # [G, G]
    denominator = k_min + 1 - adj
    denominator = np.maximum(denominator, 1e-8)  # 避免除零

    tom = (L + adj) / denominator
    np.fill_diagonal(tom, 1.0)

    logger.info("TOM: mean=%.4f, median=%.4f", tom.mean(), np.median(tom))
    return tom.astype(np.float32)


def detect_modules(
    expression: np.ndarray,
    soft_power: int = 6,
    min_module_size: int = 10,
    merge_cut_height: float = 0.25,
    ppi_matrix: np.ndarray | None = None,
    ppi_weight: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """WGCNA-Lite 基因模块检测。

    流程：
      1. 计算 soft adjacency
      2. （可选）用 PPI 加权 adjacency
      3. 计算 TOM
      4. 层次聚类（1 - TOM 为距离）
      5. Dynamic tree cut → 模块分配
      6. 计算 module eigengenes

    Args:
        expression: [N, G] 基因表达矩阵。
        soft_power: soft thresholding 幂次。
        min_module_size: 最小模块大小。
        merge_cut_height: 模块合并切割高度。
        ppi_matrix: [G, G] PPI 权重矩阵（可选，用于加权）。
        ppi_weight: PPI 加权系数。

    Returns:
        module_labels: [G] 每个基因的模块标签（0=unassigned）
        module_eigengenes: [N, M] 模块 eigengene 矩阵（M=模块数）
        info: dict 包含模块统计信息
    """
    N, G = expression.shape
    logger.info("WGCNA module detection: %d cells, %d genes, power=%d",
                 N, G, soft_power)

    # Step 1: Soft adjacency
    adj = compute_soft_adjacency(expression, power=soft_power)

    # Step 2: PPI weighting（可选）
    if ppi_matrix is not None:
        # PPI-informed adjacency: adj_ppi = adj * (1 + ppi_weight * ppi)
        # PPI 高的基因对在 adjacency 中被放大
        ppi_norm = ppi_matrix / (ppi_matrix.max() + 1e-8)
        adj = adj * (1.0 + ppi_weight * ppi_norm)
        logger.info("Applied PPI weighting (ppi_weight=%.2f)", ppi_weight)

    # Step 3: TOM
    tom = compute_tom(adj)

    # Step 4: Hierarchical clustering on 1 - TOM distance
    dist = 1.0 - tom
    np.fill_diagonal(dist, 0)

    # 转为 condensed distance matrix
    condensed = dist[np.triu_indices(G, k=1)]
    Z = linkage(condensed, method='average')

    # Step 5: Cut tree → 模块
    # 使用固定 cut height（简化版 dynamic tree cut）
    labels = fcluster(Z, t=merge_cut_height, criterion='distance')

    # 过滤小模块 → 标记为 0 (unassigned)
    unique_labels, counts = np.unique(labels, return_counts=True)
    for lbl, cnt in zip(unique_labels, counts):
        if cnt < min_module_size:
            labels[labels == lbl] = 0

    # 重新编号
    unique_nonzero = sorted(set(labels) - {0})
    label_map = {old: new for new, old in enumerate(unique_nonzero, start=1)}
    label_map[0] = 0
    module_labels = np.array([label_map[l] for l in labels], dtype=np.int32)

    n_modules = len(unique_nonzero)
    logger.info("Detected %d modules (min_size=%d)", n_modules, min_module_size)
    for m in range(1, n_modules + 1):
        n_genes = np.sum(module_labels == m)
        logger.info("  Module %d: %d genes", m, n_genes)

    # Step 6: Module eigengenes（每个模块的第一主成分）
    module_eigengenes = np.zeros((N, n_modules), dtype=np.float32)
    for m in range(1, n_modules + 1):
        gene_idx = np.where(module_labels == m)[0]
        if len(gene_idx) < 2:
            continue
        module_expr = expression[:, gene_idx]  # [N, n_genes_in_module]
        # 标准化
        module_expr_std = (module_expr - module_expr.mean(axis=0)) / (module_expr.std(axis=0) + 1e-8)
        # PCA → 第一主成分
        pca = PCA(n_components=1)
        eigengene = pca.fit_transform(module_expr_std).ravel()  # [N]
        module_eigengenes[:, m - 1] = eigengene

    info = {
        'n_modules': n_modules,
        'module_sizes': {int(m): int(np.sum(module_labels == m))
                         for m in range(1, n_modules + 1)},
        'n_unassigned': int(np.sum(module_labels == 0)),
    }

    return module_labels, module_eigengenes, info


# B. 多源超图构建

def build_knn_graph(features: np.ndarray, k: int = 7,
                     weighted: str = 'binary') -> sp.csr_matrix:
    """从特征空间构建 k-NN 图（CSR 格式）。

    Args:
        features: [N, d] 特征矩阵。
        k: 近邻数。
        weighted: 'binary' 或 'gaussian'。

    Returns:
        [N, N] CSR 邻接矩阵。
    """
    N = features.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(features)
    distances, indices = nbrs.kneighbors(features)
    distances = distances[:, 1:]  # 排除自身
    indices = indices[:, 1:]

    row_idx = np.repeat(np.arange(N), k)
    col_idx = indices.ravel()

    if weighted == 'gaussian':
        sigma = np.median(distances)
        data = np.exp(-distances.ravel() ** 2 / (2 * sigma ** 2))
    else:
        data = np.ones(len(row_idx), dtype=np.float32)

    adj = sp.csr_matrix((data.astype(np.float32), (row_idx, col_idx)), shape=(N, N))
    # 对称化
    adj = (adj + adj.T) / 2
    return adj


def build_ppi_weighted_knn(
    expression: np.ndarray,
    spatial_coords: np.ndarray,
    ppi_matrix: np.ndarray,
    k: int = 7,
    ppi_threshold: float = 0.1,
) -> sp.csr_matrix:
    """构建 PPI 加权的 k-NN 图。

    使用 PPI 权重对基因表达做加权变换，然后在变换后的特征空间做 k-NN。

    原理：
      1. 对 PPI 矩阵做特征分解，取前 d 个主成分作为"PPI embedding"
      2. 将基因表达投影到 PPI embedding 空间：X_ppi = X @ V_ppi
         → 在 PPI 强连接的基因方向上放大信号
      3. 拼接空间坐标和 PPI 表达 embedding → 联合 k-NN

    Args:
        expression: [N, G] 基因表达。
        spatial_coords: [N, 2] 空间坐标。
        ppi_matrix: [G, G] PPI 权重矩阵。
        k: k-NN 的 k。
        ppi_threshold: 过滤低权重 PPI edges 的阈值。

    Returns:
        [N, N] CSR 邻接矩阵。
    """
    G = expression.shape[1]

    # 过滤低权重
    ppi_filtered = ppi_matrix.copy()
    ppi_filtered[ppi_filtered < ppi_threshold] = 0

    # PPI 矩阵特征分解 → 取前 d 个成分
    n_components = min(50, G // 3)
    eigenvalues, eigenvectors = np.linalg.eigh(ppi_filtered)
    # 取最大的 n_components 个特征向量
    idx = np.argsort(eigenvalues)[::-1][:n_components]
    V_ppi = eigenvectors[:, idx]  # [G, d]

    # 投影到 PPI 空间
    X_ppi = expression @ V_ppi  # [N, d]

    # 标准化
    X_ppi = (X_ppi - X_ppi.mean(axis=0)) / (X_ppi.std(axis=0) + 1e-8)

    # 拼接空间坐标（标准化）
    coords_norm = (spatial_coords - spatial_coords.mean(axis=0)) / (spatial_coords.std(axis=0) + 1e-8)
    combined = np.hstack([coords_norm * 0.5, X_ppi])  # 空间坐标权重较小

    logger.info("PPI-weighted features: %d dims (spatial=2 + PPI=%d)", combined.shape[1], n_components)

    return build_knn_graph(combined, k=k, weighted='gaussian')


def build_multi_source_hypergraph(
    adata,
    spatial_graph: sp.csr_matrix,
    module_eigengenes: np.ndarray,
    ppi_matrix: np.ndarray | None,
    alpha: float = 1.0,
    beta: float = 0.3,
    gamma: float = 0.3,
    coexpr_k: int = 7,
    ppi_k: int = 7,
    ppi_threshold: float = 0.1,
) -> sp.csr_matrix:
    """构建多源融合超图。

    H_combined = α·H_spatial + β·H_coexpr + γ·H_ppi
    归一化后用于 HGNN 卷积（与原版超图格式兼容）。

    Args:
        adata: AnnData 对象（含 spatial 坐标和表达）。
        spatial_graph: 原版空间 k-NN 超图 [N, N] CSR。
        module_eigengenes: [N, M] 模块 eigengene 矩阵。
        ppi_matrix: [G, G] PPI 权重矩阵（None 时跳过 PPI 层）。
        alpha: 空间超图权重。
        beta: 共表达超图权重。
        gamma: PPI 超图权重。
        coexpr_k: 共表达 k-NN 的 k。
        ppi_k: PPI k-NN 的 k。
        ppi_threshold: PPI 权重过滤阈值。

    Returns:
        [N, N] CSR 多源融合超图矩阵。
    """
    N = adata.n_obs
    logger.info("Building multi-source hypergraph for %d cells...", N)

    # Layer 1: Spatial（原版，已传入）
    H_spatial = spatial_graph.astype(np.float32)
    # 行归一化
    row_sums = np.array(H_spatial.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    H_spatial = sp.diags(1.0 / row_sums).dot(H_spatial)
    logger.info("  Spatial graph: %d edges", H_spatial.nnz)

    # Layer 2: Co-expression module k-NN
    if module_eigengenes.shape[1] > 0 and beta > 0:
        H_coexpr = build_knn_graph(module_eigengenes, k=coexpr_k, weighted='gaussian')
        row_sums = np.array(H_coexpr.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        H_coexpr = sp.diags(1.0 / row_sums).dot(H_coexpr)
        logger.info("  Co-expression graph: %d edges", H_coexpr.nnz)
    else:
        H_coexpr = sp.csr_matrix((N, N), dtype=np.float32)
        logger.info("  Co-expression graph: SKIPPED (no modules or beta=0)")

    # Layer 3: PPI-weighted k-NN
    expression = (np.asarray(adata.X.todense())
                  if hasattr(adata.X, 'toarray') else np.asarray(adata.X))
    coords = adata.obsm['spatial']

    if ppi_matrix is not None and gamma > 0 and ppi_matrix.max() > 0:
        H_ppi = build_ppi_weighted_knn(
            expression, coords, ppi_matrix,
            k=ppi_k, ppi_threshold=ppi_threshold,
        )
        row_sums = np.array(H_ppi.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        H_ppi = sp.diags(1.0 / row_sums).dot(H_ppi)
        logger.info("  PPI-weighted graph: %d edges", H_ppi.nnz)
    else:
        H_ppi = sp.csr_matrix((N, N), dtype=np.float32)
        logger.info("  PPI graph: SKIPPED (no PPI data or gamma=0)")

    # Combine
    H_combined = alpha * H_spatial + beta * H_coexpr + gamma * H_ppi

    # 最终行归一化
    row_sums = np.array(H_combined.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    H_combined = sp.diags(1.0 / row_sums).dot(H_combined)

    logger.info("  Combined graph: %d edges, weights=(%.1f, %.1f, %.1f)",
                 H_combined.nnz, alpha, beta, gamma)

    return H_combined.tocsr()


# C. MultiSourceSpatialEx Trainer

class MultiSourceSpatialEx(SpatialEx):
    """使用多源超图（spatial + co-expression + PPI）的 SpatialEx。

    仅替换超图构建方式，模型架构（HGNN + DGI + MLP）保持原版不变。
    这使得改进的效果完全归因于超图质量的提升。

    Args:
        ppi_matrix: [G, G] PPI 权重矩阵（None 时退化为 spatial + coexpr）。
        alpha, beta, gamma: 三层超图融合权重。
        soft_power: WGCNA soft thresholding 幂次。
        min_module_size: WGCNA 最小模块大小。
    """

    def __init__(self, adata1, adata2, graph1, graph2,
                 ppi_matrix: np.ndarray | None = None,
                 alpha: float = 1.0,
                 beta: float = 0.3,
                 gamma: float = 0.3,
                 soft_power: int = 6,
                 min_module_size: int = 10,
                 merge_cut_height: float = 0.25,
                 coexpr_k: int = 7,
                 ppi_k: int = 7,
                 ppi_threshold: float = 0.1,
                 **kwargs):
        # 先调用父类 __init__（会用原版 graph 建 dataloader）
        # 我们之后会用多源图替换 dataloader
        super().__init__(adata1, adata2, graph1, graph2, **kwargs)

        # WGCNA 模块检测
        expr1 = (np.asarray(adata1.X.todense())
                 if hasattr(adata1.X, 'toarray') else np.asarray(adata1.X))
        expr2 = (np.asarray(adata2.X.todense())
                 if hasattr(adata2.X, 'toarray') else np.asarray(adata2.X))

        logger.info("=== WGCNA Module Detection (Slice 1) ===")
        labels1, eigengenes1, info1 = detect_modules(
            expr1, soft_power=soft_power,
            min_module_size=min_module_size,
            merge_cut_height=merge_cut_height,
            ppi_matrix=ppi_matrix,
            ppi_weight=gamma,
        )

        logger.info("=== WGCNA Module Detection (Slice 2) ===")
        labels2, eigengenes2, info2 = detect_modules(
            expr2, soft_power=soft_power,
            min_module_size=min_module_size,
            merge_cut_height=merge_cut_height,
            ppi_matrix=ppi_matrix,
            ppi_weight=gamma,
        )

        # 构建多源超图
        logger.info("=== Multi-source Hypergraph (Slice 1) ===")
        multi_graph1 = build_multi_source_hypergraph(
            adata1, graph1, eigengenes1, ppi_matrix,
            alpha=alpha, beta=beta, gamma=gamma,
            coexpr_k=coexpr_k, ppi_k=ppi_k, ppi_threshold=ppi_threshold,
        )

        logger.info("=== Multi-source Hypergraph (Slice 2) ===")
        multi_graph2 = build_multi_source_hypergraph(
            adata2, graph2, eigengenes2, ppi_matrix,
            alpha=alpha, beta=beta, gamma=gamma,
            coexpr_k=coexpr_k, ppi_k=ppi_k, ppi_threshold=ppi_threshold,
        )

        # 用多源超图重建 dataloader（替换父类的 dataloader）
        from SpatialEx import preprocess as pp
        self.slice1_dataloader = pp.Build_dataloader(
            adata1, graph=multi_graph1, graph_norm='hpnn',
            feat_norm=False, prune=[self.prune, self.prune], drop_last=False,
        )
        self.slice2_dataloader = pp.Build_dataloader(
            adata2, graph=multi_graph2, graph_norm='hpnn',
            feat_norm=False, prune=[self.prune, self.prune], drop_last=False,
        )

        # 保存模块信息
        self.wgcna_info = {'slice1': info1, 'slice2': info2}
        self.module_labels = {'slice1': labels1, 'slice2': labels2}

        logger.info("MultiSourceSpatialEx initialized | alpha=%.1f beta=%.1f gamma=%.1f",
                     alpha, beta, gamma)


# 评估函数

def evaluate(adata1, adata2, panelB1, panelA2):
    """计算两个 slice 的 PCC/SSIM/CMD/MoransI/SPCC 指标。"""
    results = {}
    g1 = se.pp.Build_graph(
        adata1.obsm['spatial'], graph_type='knn', weighted='gaussian',
        apply_normalize='row', return_type='coo'
    )
    ssim1, ssim1_r = se.utils.Compute_metrics(
        adata1.X.copy(), panelB1.copy(), metric='ssim', graph=g1, reduce='mean'
    )
    pcc1, pcc1_r = se.utils.Compute_metrics(
        adata1.X.copy(), panelB1.copy(), metric='pcc', reduce='mean'
    )
    cmd1, cmd1_r = se.utils.Compute_metrics(
        adata1.X.copy(), panelB1.copy(), metric='cmd', reduce='mean'
    )
    morans1, morans1_r = compute_morans_i_pred(adata1.X, panelB1, adata1.obsm['spatial'])
    spcc1, spcc1_r = compute_spcc(adata1.X, panelB1)
    results['slice1'] = {
        'PCC': float(pcc1_r), 'SSIM': float(ssim1_r), 'CMD': float(cmd1_r),
        'MoransI': float(morans1_r), 'SPCC': float(spcc1_r),
        'pcc_per_gene': pcc1, 'ssim_per_gene': ssim1,
        'morans_per_gene': morans1, 'spcc_per_gene': spcc1,
    }

    g2 = se.pp.Build_graph(
        adata2.obsm['spatial'], graph_type='knn', weighted='gaussian',
        apply_normalize='row', return_type='coo'
    )
    ssim2, ssim2_r = se.utils.Compute_metrics(
        adata2.X.copy(), panelA2.copy(), metric='ssim', graph=g2, reduce='mean'
    )
    pcc2, pcc2_r = se.utils.Compute_metrics(
        adata2.X.copy(), panelA2.copy(), metric='pcc', reduce='mean'
    )
    cmd2, cmd2_r = se.utils.Compute_metrics(
        adata2.X.copy(), panelA2.copy(), metric='cmd', reduce='mean'
    )
    morans2, morans2_r = compute_morans_i_pred(adata2.X, panelA2, adata2.obsm['spatial'])
    spcc2, spcc2_r = compute_spcc(adata2.X, panelA2)
    results['slice2'] = {
        'PCC': float(pcc2_r), 'SSIM': float(ssim2_r), 'CMD': float(cmd2_r),
        'MoransI': float(morans2_r), 'SPCC': float(spcc2_r),
        'pcc_per_gene': pcc2, 'ssim_per_gene': ssim2,
        'morans_per_gene': morans2, 'spcc_per_gene': spcc2,
    }
    return results


def spatial_split_adata(adata, axis=0):
    """沿空间坐标中位数将单切片分成两半（论文 Extended Data Fig. 2B 做法）。"""
    coords = adata.obsm['spatial']
    median_val = np.median(coords[:, axis])
    mask1 = coords[:, axis] <= median_val
    mask2 = coords[:, axis] > median_val
    adata1 = adata[mask1].copy()
    adata2 = adata[mask2].copy()
    print(f"[Spatial split] {adata.n_obs} cells -> {adata1.n_obs} + {adata2.n_obs} (axis={axis}, median={median_val:.1f})")
    return adata1, adata2


def preprocess_skin_data():
    """预处理 Skin Melanoma 数据：读取 → 空间切分 → H&E patches → 建图。"""
    print("=" * 60)
    print("SKIN MELANOMA: Preprocessing")
    print("=" * 60)
    adata = se.pp.Read_Xenium(
        skin_root + 'cell_feature_matrix.h5', skin_root + 'cells.csv')
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


# 1. 数据读取与预处理
print("=" * 70)
print("TASK 6: WGCNA + Tissue-Specific PPI Hypergraph")
print("=" * 70)

print("\nStage 1: Preprocessing Slice 1")
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
    he_patches, adata=adata1, image_encoder=image_encoder, device=device, store_key='he'
)
del he_patches, img
print(f'[OK] Slice 1: {adata1.shape}')

print("\nStage 2: Preprocessing Slice 2")
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
    he_patches, adata=adata2, image_encoder=image_encoder, store_key='he', device=device
)
del he_patches, img
print(f'[OK] Slice 2: {adata2.shape}')

# 2. 加载 PPI 数据
print("\nStage 3: Loading PPI data")

ppi_matrix = None
ppi_matrix_path = os.path.join(PPI_DATA_DIR, 'ppi_matrix.npy')

if os.path.exists(ppi_matrix_path):
    ppi_matrix = np.load(ppi_matrix_path)
    ppi_gene_names = np.load(os.path.join(PPI_DATA_DIR, 'ppi_gene_names.npy'),
                              allow_pickle=True)

    # 验证基因对齐
    our_genes = set(adata1.var_names)
    ppi_genes = set(ppi_gene_names)
    overlap = our_genes & ppi_genes
    logger.info("PPI matrix: %dx%d, overlap with adata genes: %d / %d",
                 ppi_matrix.shape[0], ppi_matrix.shape[1],
                 len(overlap), len(our_genes))

    # 如果基因顺序不一致，需要重排
    if list(ppi_gene_names) != list(adata1.var_names):
        logger.info("Reindexing PPI matrix to match adata gene order...")
        ppi_gene_to_idx = {g: i for i, g in enumerate(ppi_gene_names)}
        G = adata1.n_vars
        ppi_reindexed = np.zeros((G, G), dtype=np.float32)
        for i, g1 in enumerate(adata1.var_names):
            for j, g2 in enumerate(adata1.var_names):
                if g1 in ppi_gene_to_idx and g2 in ppi_gene_to_idx:
                    pi = ppi_gene_to_idx[g1]
                    pj = ppi_gene_to_idx[g2]
                    ppi_reindexed[i, j] = ppi_matrix[pi, pj]
        ppi_matrix = ppi_reindexed

    n_nonzero = np.count_nonzero(ppi_matrix)
    logger.info("PPI matrix loaded: %d nonzero edges", n_nonzero // 2)
else:
    logger.warning("PPI data not found at %s", PPI_DATA_DIR)
    logger.warning("Running WITHOUT PPI layer. Run download_ppi_network.py first.")
    logger.warning("Will still use spatial + WGCNA co-expression layers.")

# 3. 建图 + 训练
print("\nStage 4: Building spatial graphs (baseline)")
graph1 = se.pp.Build_hypergraph_spatial_and_HE(
    adata1, num_neighbors, graph_kind='spatial', return_type='csr'
)
graph2 = se.pp.Build_hypergraph_spatial_and_HE(
    adata2, num_neighbors, graph_kind='spatial', return_type='csr'
)

# --- 消融实验：对比多种超图配置 ---
ABLATION_CONFIGS = {
    '0_baseline': {
        'alpha': 1.0, 'beta': 0.0, 'gamma': 0.0,
        'description': '原版 Spatial-only hypergraph',
        'use_original': True,
    },
    '1_spatial_coexpr': {
        'alpha': 1.0, 'beta': 0.3, 'gamma': 0.0,
        'description': 'Spatial + WGCNA co-expression',
        'use_original': False,
    },
    '2_spatial_ppi': {
        'alpha': 1.0, 'beta': 0.0, 'gamma': 0.3,
        'description': 'Spatial + PPI-weighted',
        'use_original': False,
    },
    '3_all_three': {
        'alpha': ALPHA_SPATIAL, 'beta': BETA_COEXPR, 'gamma': GAMMA_PPI,
        'description': 'Spatial + WGCNA + PPI (full model)',
        'use_original': False,
    },
}

# 如果没有 PPI 数据，跳过含 PPI 的变体
if ppi_matrix is None:
    ABLATION_CONFIGS = {k: v for k, v in ABLATION_CONFIGS.items()
                        if v.get('gamma', 0) == 0 or v.get('use_original', False)}
    logger.warning("Skipping PPI variants (no PPI data)")

all_results = {}

for config_name, config in ABLATION_CONFIGS.items():
    print(f"\n{'='*70}")
    print(f"Config: {config_name} — {config['description']}")
    print(f"  alpha={config['alpha']}, beta={config['beta']}, gamma={config['gamma']}")
    print('=' * 70)

    config_dir = os.path.join(output_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)

    # Overwrite 检查
    if not OVERWRITE and os.path.exists(os.path.join(config_dir, 'metrics.json')):
        logger.info("Skipping %s (already exists, OVERWRITE=False)", config_name)
        with open(os.path.join(config_dir, 'metrics.json')) as f:
            all_results[config_name] = json.load(f)
        continue

    t_start = time.time()

    if config.get('use_original', False):
        # 原版 baseline
        trainer = se.SpatialEx(
            adata1, adata2, graph1, graph2,
            epochs=epochs, device=device,
        )
    else:
        trainer = MultiSourceSpatialEx(
            adata1, adata2, graph1, graph2,
            ppi_matrix=ppi_matrix,
            alpha=config['alpha'],
            beta=config['beta'],
            gamma=config['gamma'],
            soft_power=WGCNA_SOFT_POWER,
            min_module_size=WGCNA_MIN_MODULE_SIZE,
            merge_cut_height=WGCNA_MERGE_CUT_HEIGHT,
            coexpr_k=COEXPR_K,
            ppi_k=PPI_K,
            ppi_threshold=PPI_THRESHOLD,
            epochs=epochs, device=device,
        )
        # 保存模块信息
        if hasattr(trainer, 'wgcna_info'):
            with open(os.path.join(config_dir, 'wgcna_info.json'), 'w') as f:
                json.dump(trainer.wgcna_info, f, indent=2)
            np.save(os.path.join(config_dir, 'module_labels_s1.npy'),
                    trainer.module_labels['slice1'])
            np.save(os.path.join(config_dir, 'module_labels_s2.npy'),
                    trainer.module_labels['slice2'])

    trainer.train()
    panelB1, panelA2 = trainer.auto_inference()
    results = evaluate(adata1, adata2, panelB1, panelA2)
    t_elapsed = time.time() - t_start

    print(f"  Slice 1 — PCC: {results['slice1']['PCC']:.6f}  "
          f"SSIM: {results['slice1']['SSIM']:.6f}  "
          f"CMD: {results['slice1']['CMD']:.6f}")
    print(f"  Slice 2 — PCC: {results['slice2']['PCC']:.6f}  "
          f"SSIM: {results['slice2']['SSIM']:.6f}  "
          f"CMD: {results['slice2']['CMD']:.6f}")
    print(f"  Time: {t_elapsed/60:.1f} min")

    np.save(os.path.join(config_dir, 'slice1_pcc_per_gene.npy'), results['slice1']['pcc_per_gene'])
    np.save(os.path.join(config_dir, 'slice1_ssim_per_gene.npy'), results['slice1']['ssim_per_gene'])
    np.save(os.path.join(config_dir, 'slice2_pcc_per_gene.npy'), results['slice2']['pcc_per_gene'])
    np.save(os.path.join(config_dir, 'slice2_ssim_per_gene.npy'), results['slice2']['ssim_per_gene'])
    np.save(os.path.join(config_dir, 'panelB1.npy'), panelB1)
    np.save(os.path.join(config_dir, 'panelA2.npy'), panelA2)

    save_res = {
        'slice1': {k: v for k, v in results['slice1'].items() if k in ('PCC', 'SSIM', 'CMD', 'MoransI', 'SPCC')},
        'slice2': {k: v for k, v in results['slice2'].items() if k in ('PCC', 'SSIM', 'CMD', 'MoransI', 'SPCC')},
        'config': {k: v for k, v in config.items() if k != 'use_original'},
        'time_seconds': t_elapsed,
    }
    with open(os.path.join(config_dir, 'metrics.json'), 'w') as f:
        json.dump(save_res, f, indent=2)

    all_results[config_name] = save_res

    del trainer, panelB1, panelA2
    torch.cuda.empty_cache()
    gc.collect()

# 4. 汇总
print("\n\n" + "=" * 100)
print("WGCNA + PPI HYPERGRAPH ABLATION SUMMARY")
print("=" * 100)

header = (f"{'Config':<22} {'Description':<35} "
          f"{'PCC_s1':>8} {'PCC_s2':>8} {'SSIM_s1':>8} {'SSIM_s2':>8} {'CMD_s1':>8} {'CMD_s2':>8}")
print(header)
print("-" * len(header))

bl = all_results.get('0_baseline', {})

for name, res in all_results.items():
    s1, s2 = res['slice1'], res['slice2']
    desc = res['config']['description'][:33]
    print(f"{name:<22} {desc:<35} "
          f"{s1['PCC']:>8.4f} {s2['PCC']:>8.4f} "
          f"{s1['SSIM']:>8.4f} {s2['SSIM']:>8.4f} "
          f"{s1['CMD']:>8.4f} {s2['CMD']:>8.4f}")

if bl:
    print("\nDIFF vs baseline:")
    print("-" * len(header))
    b1, b2 = bl['slice1'], bl['slice2']
    for name, res in all_results.items():
        if name == '0_baseline':
            continue
        s1, s2 = res['slice1'], res['slice2']

        def arrow(val, higher_better=True):
            if higher_better:
                return 'better' if val > 0.001 else ('worse' if val < -0.001 else '~')
            else:
                return 'better' if val < -0.001 else ('worse' if val > 0.001 else '~')

        print(f"{name:<22} "
              f"{s1['PCC']-b1['PCC']:>+8.4f}({arrow(s1['PCC']-b1['PCC'])}) "
              f"{s2['PCC']-b2['PCC']:>+8.4f}({arrow(s2['PCC']-b2['PCC'])}) "
              f"{s1['SSIM']-b1['SSIM']:>+8.4f}({arrow(s1['SSIM']-b1['SSIM'])}) "
              f"{s2['SSIM']-b2['SSIM']:>+8.4f}({arrow(s2['SSIM']-b2['SSIM'])}) "
              f"{s1['CMD']-b1['CMD']:>+8.4f}({arrow(s1['CMD']-b1['CMD'], False)}) "
              f"{s2['CMD']-b2['CMD']:>+8.4f}({arrow(s2['CMD']-b2['CMD'], False)})")

# Wilcoxon
from scipy import stats as scipy_stats
print("\n\nStatistical Significance (Wilcoxon vs baseline):")
bl_dir = os.path.join(output_dir, '0_baseline')

for config_name in ABLATION_CONFIGS:
    if config_name == '0_baseline':
        continue
    v_dir = os.path.join(output_dir, config_name)
    print(f"\n{config_name}:")
    for slice_id in [1, 2]:
        for metric_name in ['pcc', 'ssim']:
            bl_path = os.path.join(bl_dir, f'slice{slice_id}_{metric_name}_per_gene.npy')
            v_path = os.path.join(v_dir, f'slice{slice_id}_{metric_name}_per_gene.npy')
            if os.path.exists(bl_path) and os.path.exists(v_path):
                bl_vals = np.load(bl_path)
                v_vals = np.load(v_path)
                min_len = min(len(bl_vals), len(v_vals))
                mask = ~(np.isnan(bl_vals[:min_len]) | np.isnan(v_vals[:min_len]))
                bl_c, v_c = bl_vals[:min_len][mask], v_vals[:min_len][mask]
                if len(bl_c) > 10:
                    try:
                        stat, pval = scipy_stats.wilcoxon(v_c, bl_c)
                        mean_diff = np.mean(v_c - bl_c)
                        sig = ('*** p<0.001' if pval < 0.001 else
                               '** p<0.01' if pval < 0.01 else
                               '* p<0.05' if pval < 0.05 else 'n.s.')
                        print(f"  Slice {slice_id} {metric_name.upper()}: "
                              f"mean_diff={mean_diff:+.6f}, p={pval:.4e} {sig}")
                    except Exception as e:
                        print(f"  Slice {slice_id} {metric_name.upper()}: test failed ({e})")

# Save summary
with open(os.path.join(output_dir, 'ablation_summary.json'), 'w') as f:
    json.dump(all_results, f, indent=2)

# Save best config (3_all_three) as top-level metrics
best_key = '3_all_three' if '3_all_three' in all_results else '1_spatial_coexpr'
if best_key in all_results:
    best = all_results[best_key]
    metrics_top = {
        'slice1_prediction': best['slice1'],
        'slice2_prediction': best['slice2'],
        'config': best['config'],
    }
    with open(os.path.join(output_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics_top, f, indent=2)

print(f"\n[SAVED] All results to: {output_dir}")
print("=" * 70)
print("Breast Cancer DONE!")
print("=" * 70)


# PHASE 2: Skin Melanoma Dataset（完整消融，与 MG 一致）
print("\n\n" + "#" * 70)
print("# PHASE 2: SKIN MELANOMA DATASET (spatial split, full ablation)")
print("#" * 70)

skin_adata1, skin_adata2, skin_graph1, skin_graph2 = preprocess_skin_data()

# 加载 skin PPI 矩阵
skin_ppi_path = os.path.join(PPI_DATA_DIR_SKIN, 'ppi_matrix.npy')
if os.path.exists(skin_ppi_path):
    skin_ppi_matrix = np.load(skin_ppi_path)
    logger.info("[SKIN] Loaded PPI matrix: %s", skin_ppi_matrix.shape)
else:
    skin_ppi_matrix = None
    logger.warning("[SKIN] PPI matrix not found at %s, running without PPI", skin_ppi_path)

# 复用 MG 的消融配置
SKIN_ABLATION_CONFIGS = dict(ABLATION_CONFIGS)
if skin_ppi_matrix is None:
    SKIN_ABLATION_CONFIGS = {k: v for k, v in SKIN_ABLATION_CONFIGS.items()
                              if v.get('gamma', 0) == 0 or v.get('use_original', False)}
    logger.warning("[SKIN] Skipping PPI variants (no PPI data)")

SCALAR_KEYS = ('PCC', 'SSIM', 'CMD', 'MoransI', 'SPCC')
skin_all_results = {}

for config_name, config in SKIN_ABLATION_CONFIGS.items():
    print(f"\n{'='*70}")
    print(f"[SKIN] Config: {config_name} — {config['description']}")
    print(f"  alpha={config['alpha']}, beta={config['beta']}, gamma={config['gamma']}")
    print('=' * 70)

    config_dir = os.path.join(output_dir_skin, config_name)
    os.makedirs(config_dir, exist_ok=True)

    # Overwrite 检查
    if not OVERWRITE and os.path.exists(os.path.join(config_dir, 'metrics.json')):
        logger.info("[SKIN] Skipping %s (already exists, OVERWRITE=False)", config_name)
        with open(os.path.join(config_dir, 'metrics.json')) as f:
            skin_all_results[config_name] = json.load(f)
        continue

    t_start = time.time()

    if config.get('use_original', False):
        skin_trainer = se.SpatialEx(
            skin_adata1, skin_adata2, skin_graph1, skin_graph2,
            epochs=epochs, device=device,
        )
    else:
        skin_trainer = MultiSourceSpatialEx(
            skin_adata1, skin_adata2, skin_graph1, skin_graph2,
            ppi_matrix=skin_ppi_matrix,
            alpha=config['alpha'],
            beta=config['beta'],
            gamma=config['gamma'],
            soft_power=WGCNA_SOFT_POWER,
            min_module_size=WGCNA_MIN_MODULE_SIZE,
            merge_cut_height=WGCNA_MERGE_CUT_HEIGHT,
            coexpr_k=COEXPR_K,
            ppi_k=PPI_K,
            ppi_threshold=PPI_THRESHOLD,
            epochs=epochs, device=device,
        )
        if hasattr(skin_trainer, 'wgcna_info'):
            with open(os.path.join(config_dir, 'wgcna_info.json'), 'w') as f:
                json.dump(skin_trainer.wgcna_info, f, indent=2)
            np.save(os.path.join(config_dir, 'module_labels_s1.npy'),
                    skin_trainer.module_labels['slice1'])
            np.save(os.path.join(config_dir, 'module_labels_s2.npy'),
                    skin_trainer.module_labels['slice2'])

    skin_trainer.train()
    panelB1_skin, panelA2_skin = skin_trainer.auto_inference()
    skin_results = evaluate(skin_adata1, skin_adata2, panelB1_skin, panelA2_skin)
    t_elapsed = time.time() - t_start

    s1, s2 = skin_results['slice1'], skin_results['slice2']
    print(f"  Slice 1 — PCC: {s1['PCC']:.6f}  SSIM: {s1['SSIM']:.6f}  CMD: {s1['CMD']:.6f}")
    print(f"  Slice 2 — PCC: {s2['PCC']:.6f}  SSIM: {s2['SSIM']:.6f}  CMD: {s2['CMD']:.6f}")
    print(f"  Time: {t_elapsed/60:.1f} min")

    # 保存 per-gene 数组
    for slice_id, s_key in [(1, 'slice1'), (2, 'slice2')]:
        for m_name in ['pcc', 'ssim', 'morans', 'spcc']:
            arr_key = f'{m_name}_per_gene'
            if arr_key in skin_results[s_key]:
                np.save(os.path.join(config_dir, f'slice{slice_id}_{m_name}_per_gene.npy'),
                        skin_results[s_key][arr_key])
    np.save(os.path.join(config_dir, 'panelB1.npy'), panelB1_skin)
    np.save(os.path.join(config_dir, 'panelA2.npy'), panelA2_skin)

    save_res = {
        'slice1': {k: v for k, v in s1.items() if k in SCALAR_KEYS},
        'slice2': {k: v for k, v in s2.items() if k in SCALAR_KEYS},
        'config': {k: v for k, v in config.items() if k != 'use_original'},
        'time_seconds': t_elapsed,
    }
    with open(os.path.join(config_dir, 'metrics.json'), 'w') as f:
        json.dump(save_res, f, indent=2)

    skin_all_results[config_name] = save_res

    del skin_trainer, panelB1_skin, panelA2_skin
    torch.cuda.empty_cache()
    gc.collect()

# Skin 汇总表
print("\n\n" + "=" * 100)
print("[SKIN] WGCNA + PPI HYPERGRAPH ABLATION SUMMARY")
print("=" * 100)

header = (f"{'Config':<22} {'Description':<35} "
          f"{'PCC_s1':>8} {'PCC_s2':>8} {'SSIM_s1':>8} {'SSIM_s2':>8} {'CMD_s1':>8} {'CMD_s2':>8}")
print(header)
print("-" * len(header))

skin_bl = skin_all_results.get('0_baseline', {})
for name, res in skin_all_results.items():
    s1, s2 = res['slice1'], res['slice2']
    desc = res.get('config', {}).get('description', name)[:33]
    print(f"{name:<22} {desc:<35} "
          f"{s1['PCC']:>8.4f} {s2['PCC']:>8.4f} "
          f"{s1['SSIM']:>8.4f} {s2['SSIM']:>8.4f} "
          f"{s1['CMD']:>8.4f} {s2['CMD']:>8.4f}")

if skin_bl:
    print("\n[SKIN] DIFF vs baseline:")
    print("-" * len(header))
    b1, b2 = skin_bl['slice1'], skin_bl['slice2']
    for name, res in skin_all_results.items():
        if name == '0_baseline':
            continue
        s1, s2 = res['slice1'], res['slice2']
        print(f"{name:<22} "
              f"{s1['PCC']-b1['PCC']:>+8.4f} {s2['PCC']-b2['PCC']:>+8.4f} "
              f"{s1['SSIM']-b1['SSIM']:>+8.4f} {s2['SSIM']-b2['SSIM']:>+8.4f} "
              f"{s1['CMD']-b1['CMD']:>+8.4f} {s2['CMD']-b2['CMD']:>+8.4f}")

# Skin Wilcoxon
print("\n\n[SKIN] Statistical Significance (Wilcoxon vs baseline):")
skin_bl_dir = os.path.join(output_dir_skin, '0_baseline')
for config_name in SKIN_ABLATION_CONFIGS:
    if config_name == '0_baseline':
        continue
    v_dir = os.path.join(output_dir_skin, config_name)
    print(f"\n{config_name}:")
    for slice_id in [1, 2]:
        for metric_name in ['pcc', 'ssim']:
            bl_path = os.path.join(skin_bl_dir, f'slice{slice_id}_{metric_name}_per_gene.npy')
            v_path = os.path.join(v_dir, f'slice{slice_id}_{metric_name}_per_gene.npy')
            if os.path.exists(bl_path) and os.path.exists(v_path):
                bl_vals = np.load(bl_path)
                v_vals = np.load(v_path)
                min_len = min(len(bl_vals), len(v_vals))
                mask = ~(np.isnan(bl_vals[:min_len]) | np.isnan(v_vals[:min_len]))
                bl_c, v_c = bl_vals[:min_len][mask], v_vals[:min_len][mask]
                if len(bl_c) > 10:
                    try:
                        stat, pval = scipy_stats.wilcoxon(v_c, bl_c)
                        mean_diff = np.mean(v_c - bl_c)
                        sig = ('*** p<0.001' if pval < 0.001 else
                               '** p<0.01' if pval < 0.01 else
                               '* p<0.05' if pval < 0.05 else 'n.s.')
                        print(f"  Slice {slice_id} {metric_name.upper()}: "
                              f"mean_diff={mean_diff:+.6f}, p={pval:.4e} {sig}")
                    except Exception as e:
                        print(f"  Slice {slice_id} {metric_name.upper()}: test failed ({e})")

with open(os.path.join(output_dir_skin, 'ablation_summary.json'), 'w') as f:
    json.dump(skin_all_results, f, indent=2)

best_key = '3_all_three' if '3_all_three' in skin_all_results else '1_spatial_coexpr'
if best_key in skin_all_results:
    best = skin_all_results[best_key]
    metrics_top = {
        'slice1_prediction': best['slice1'],
        'slice2_prediction': best['slice2'],
        'config': best.get('config', {}),
    }
    with open(os.path.join(output_dir_skin, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics_top, f, indent=2)

print(f'\n[SAVED] Skin results to: {output_dir_skin}')
print("=" * 70)
print("ALL DATASETS COMPLETE!")
print("=" * 70)
