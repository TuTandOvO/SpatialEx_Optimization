#!/usr/bin/env python3
"""
SpatialEx — Feature Selection Pipeline (HPC)
==============================================
改进点：在 SpatialEx 框架中引入两种特征选择机制：

  (A) 空间变异基因（SVG）选择 — 聚焦训练于空间表达模式最显著的基因
  (B) 注意力门控（Attention Gate） — 对输入 H&E 特征进行可学习的维度级门控

**A. SVG 选择（Spatially Variable Gene Selection）**
  动机：
    原版用所有基因训练预测头，但许多基因（尤其是高表达管家基因）空间模式平淡，
    占用了模型容量，还可能因主导 MSE 而掩盖空间标志性基因的信号。

  实现：
    1. 用 Moran's I 量化每个基因的空间自相关性（越高 = 空间模式越显著）。
    2. 选取 Moran's I 最高的 top-k 基因子集，只在这些基因上计算预测损失。
    3. 推理时仍预测全部基因，确保评估公平。

  预期提升：
    - SVG 子集上的 PCC 和 SSIM 显著提升（更专注地学习空间分布）
    - 全局平均 PCC/SSIM 轻微提升（减少噪声基因干扰）

**B. 注意力门控（Attention Gate on H&E Features）**
  动机：
    UNI/ResNet50 提取的特征向量维度高（1024/2048），其中部分维度对基因预测
    可能噪声大。引入可学习的 sigmoid 门，按维度选择性地抑制噪声特征。

  实现：
    在预测头 MLP 之前，添加：
      gate = sigmoid(Linear(he_rep))      # [N, in_dim]，逐维度门权重
      he_rep_gated = he_rep * gate        # 逐元素相乘
    然后送入原版 MLP + HGNN 流程。

  预期提升：
    - 减少 H&E 特征噪声 → PCC 稳定提升
    - 注意力权重可视化揭示哪些特征维度对基因预测最重要（可解释性）

用法：
    python run_feature_select_pipeline.py

    通过顶部的 METHOD 变量切换实验：
      METHOD = 'svg'      → 仅 SVG 选择
      METHOD = 'attn'     → 仅注意力门控
      METHOD = 'combined' → 两者结合
      METHOD = 'ablation' → 跑全部变体
"""

import os
import json
import logging
import time
import gc

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

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

output_dir = '/gpfsdata/home/renyixiang/YuanLab/data/MG_Xenium/SpatialEx_results_feature_select_MG/'
OVERWRITE = False  # True = 覆盖已有结果；False = 跳过已完成的变体
os.makedirs(output_dir, exist_ok=True)

baseline_dir = '/gpfsdata/home/renyixiang/YuanLab/data/MG_Xenium/SpatialEx_results/'

# ---- Skin Melanoma 数据配置（单切片 → 空间切分为两半）----
skin_root = '/gpfsdata/home/renyixiang/YuanLab/data/Skin_Xenium/Human_Skin_Melanoma_Base_FFPE/'
output_dir_skin = '/gpfsdata/home/renyixiang/YuanLab/data/Skin_Xenium/SpatialEx_results_feature_select_skin/'
os.makedirs(output_dir_skin, exist_ok=True)

# 实验模式
METHOD = 'ablation'   # 'svg' | 'attn' | 'combined' | 'ablation'

# SVG 参数
SVG_TOP_K = 150       # 取 Moran's I 最高的 K 个基因用于训练损失
                      # 推理时仍预测全部基因（out_dim = adata.n_vars）

# Attention Gate 参数
ATTN_GATE_HIDDEN = 256  # 门控 MLP 的隐藏层大小（小于 in_dim 以减少参数量）
ATTN_GATE_DROPOUT = 0.1


# =============================================
# A. Moran's I 计算与 SVG 选择
# =============================================

def compute_morans_i(expression: np.ndarray, spatial_coords: np.ndarray,
                     k: int = 7) -> np.ndarray:
    """计算每个基因的 Moran's I 空间自相关系数。

    Moran's I ∈ [-1, 1]：
      +1 = 强正空间自相关（相邻细胞表达相似）
       0 = 随机分布
      -1 = 强负空间自相关（相邻细胞表达互斥）

    高 Moran's I 的基因具有显著的空间表达模式，是 SVG 选择的首选目标。

    Args:
        expression: [N, G] 基因表达矩阵（已规范化）。
        spatial_coords: [N, 2] 空间坐标。
        k: 近邻数量，用于构建空间权重矩阵。

    Returns:
        [G] float32 数组，每个基因的 Moran's I 值。
    """
    if sp.issparse(expression):
        expression = np.asarray(expression.todense())
    else:
        expression = np.asarray(expression)

    N, G = expression.shape

    # 构建 k-NN 空间权重矩阵（行归一化）
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='euclidean').fit(spatial_coords)
    _, indices = nbrs.kneighbors(spatial_coords)
    indices = indices[:, 1:]  # 排除自身

    row_idx = np.repeat(np.arange(N), k)
    col_idx = indices.ravel()
    W = sp.csr_matrix(
        (np.ones(len(row_idx), dtype=np.float32), (row_idx, col_idx)),
        shape=(N, N)
    )
    # 行归一化（空间权重矩阵标准化）
    row_sums = np.array(W.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    W_norm = sp.diags(1.0 / row_sums).dot(W)

    # 向量化 Moran's I 计算
    # I_g = (N / S0) * (X_g - mu_g)' W (X_g - mu_g) / ||X_g - mu_g||^2
    # 对于行归一化 W，S0 = N（每行和为1）
    morans = np.zeros(G, dtype=np.float32)
    S0 = float(N)

    for g in range(G):
        x = expression[:, g].astype(np.float64)
        mu = x.mean()
        z = x - mu
        norm_sq = np.dot(z, z)
        if norm_sq < 1e-12:
            morans[g] = 0.0
            continue
        Wz = W_norm.dot(z)
        morans[g] = float((N / S0) * np.dot(z, Wz) / norm_sq)

    logger.info(
        "Moran's I: min=%.4f, max=%.4f, mean=%.4f, top10: %s",
        morans.min(), morans.max(), morans.mean(),
        np.sort(morans)[::-1][:10].round(4).tolist()
    )
    return morans


def select_svg_indices(morans_i: np.ndarray, top_k: int) -> np.ndarray:
    """返回 Moran's I 最高的 top_k 个基因的索引。

    Args:
        morans_i: [G] 每个基因的 Moran's I 值。
        top_k: 选取的基因数量。

    Returns:
        [top_k] int64 索引数组（按 Moran's I 降序排列）。
    """
    top_k = min(top_k, len(morans_i))
    indices = np.argsort(morans_i)[::-1][:top_k]
    logger.info("Selected %d SVGs (Moran's I range: %.4f to %.4f)",
                top_k, morans_i[indices[-1]], morans_i[indices[0]])
    return indices.astype(np.int64)


# =============================================
# A. SVG 预测头：只对 SVG 子集计算损失
# =============================================

class SVGPredictor_spot(nn.Module):
    """基于空间变异基因（SVG）选择的预测头。

    架构与原版完全相同，仅在前向传播时：
      - 用全部基因预测（out_dim = G）
      - 只对 SVG 子集计算损失（损失 = MSE on SVGs only）

    这样训练时专注 SVG，推理时仍输出完整基因谱。

    Args:
        in_dim: 输入维度。
        hidden_dim: 隐藏层维度。
        out_dim: 输出维度（全部基因数 G）。
        num_layers: HGNN 层数。
        svg_indices: [K] SVG 基因索引，None 时等效原版（全基因损失）。
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int, activation: str = 'prelu',
                 svg_indices: np.ndarray | None = None):
        super().__init__()
        self.agg = True
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
        )
        self.mod = HGNN(
            in_dim=hidden_dim, num_hidden=hidden_dim, out_dim=hidden_dim,
            num_layers=num_layers, dropout=0, activation=activation,
        )
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.criterion = nn.MSELoss()

        if svg_indices is not None:
            self.register_buffer(
                'svg_indices',
                torch.tensor(svg_indices, dtype=torch.long)
            )
        else:
            self.svg_indices = None

    def forward(self, graph, he_rep, x, agg_mtx=None, selection=None):
        he_rep = self.mlp(he_rep)
        enc = self.mod(he_rep, graph)
        x_prime = F.leaky_relu(self.linear(F.leaky_relu(enc)))  # [N, G]

        if self.agg:
            pred_full = torch.sparse.mm(agg_mtx, x_prime[selection])  # [n_agg, G]
        else:
            pred_full = x_prime

        # 只对 SVG 子集计算损失
        if self.svg_indices is not None:
            pred_svg = pred_full[:, self.svg_indices]   # [n_agg, K]
            true_svg = x[:, self.svg_indices]            # [n_agg, K]
            loss = self.criterion(pred_svg, true_svg)
        else:
            loss = self.criterion(pred_full, x)

        return loss, x_prime, enc

    def predict(self, graph, he_rep):
        he_rep = self.mlp(he_rep)
        enc = self.mod(he_rep, graph)
        return F.leaky_relu(self.linear(F.leaky_relu(enc)))


# =============================================
# B. Attention Gate 预测头
# =============================================

class AttentionGatePredictor_spot(nn.Module):
    """带注意力门控的预测头。

    在标准 MLP 输入之前，通过一个小型 sigmoid 门学习对 H&E 特征维度加权：
      gate = sigmoid(W2 * ReLU(W1 * he_rep))   # [N, in_dim]
      he_rep_gated = he_rep * gate              # 逐元素相乘

    这使模型能够学习忽略对基因预测噪声大的特征维度，专注于有用的维度。

    Args:
        in_dim: 输入特征维度（H&E embedding 维度）。
        hidden_dim: 主干 HGNN 隐藏维度。
        out_dim: 输出维度（基因数）。
        num_layers: HGNN 层数。
        gate_hidden_dim: 注意力门控 MLP 隐藏层维度。
        gate_dropout: 门控 dropout 比例。
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int, activation: str = 'prelu',
                 gate_hidden_dim: int = 256, gate_dropout: float = 0.1):
        super().__init__()
        self.agg = True

        # Attention Gate: 学习对 in_dim 维特征进行门控
        # 使用比 in_dim 小的 hidden 以减少过拟合
        self.attention_gate = nn.Sequential(
            nn.Linear(in_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(gate_dropout),
            nn.Linear(gate_hidden_dim, in_dim),
            nn.Sigmoid(),
        )

        # 主干（与原版相同）
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
        )
        self.mod = HGNN(
            in_dim=hidden_dim, num_hidden=hidden_dim, out_dim=hidden_dim,
            num_layers=num_layers, dropout=0, activation=activation,
        )
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.criterion = nn.MSELoss()

    def forward(self, graph, he_rep, x, agg_mtx=None, selection=None):
        # 注意力门控
        gate = self.attention_gate(he_rep)  # [N, in_dim]
        he_rep_gated = he_rep * gate         # [N, in_dim]

        # 原版流程
        he_rep_gated = self.mlp(he_rep_gated)
        enc = self.mod(he_rep_gated, graph)
        x_prime = F.leaky_relu(self.linear(F.leaky_relu(enc)))

        if self.agg:
            pred = torch.sparse.mm(agg_mtx, x_prime[selection])
        else:
            pred = x_prime
        loss = self.criterion(pred, x)
        return loss, x_prime, enc

    def predict(self, graph, he_rep):
        gate = self.attention_gate(he_rep)
        he_rep_gated = he_rep * gate
        he_rep_gated = self.mlp(he_rep_gated)
        enc = self.mod(he_rep_gated, graph)
        return F.leaky_relu(self.linear(F.leaky_relu(enc)))

    def get_gate_weights(self, he_rep: torch.Tensor) -> np.ndarray:
        """返回平均注意力权重（用于可视化）。

        Args:
            he_rep: [N, in_dim] H&E 特征张量。

        Returns:
            [in_dim] 平均门控权重。
        """
        with torch.no_grad():
            gate = self.attention_gate(he_rep)
            return gate.mean(dim=0).cpu().numpy()


# =============================================
# C. Combined: SVG + Attention Gate
# =============================================

class CombinedPredictor_spot(nn.Module):
    """结合 SVG 损失选择 + Attention Gate 的预测头。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int, activation: str = 'prelu',
                 svg_indices: np.ndarray | None = None,
                 gate_hidden_dim: int = 256, gate_dropout: float = 0.1):
        super().__init__()
        self.agg = True

        self.attention_gate = nn.Sequential(
            nn.Linear(in_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(gate_dropout),
            nn.Linear(gate_hidden_dim, in_dim),
            nn.Sigmoid(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
        )
        self.mod = HGNN(
            in_dim=hidden_dim, num_hidden=hidden_dim, out_dim=hidden_dim,
            num_layers=num_layers, dropout=0, activation=activation,
        )
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.criterion = nn.MSELoss()

        if svg_indices is not None:
            self.register_buffer(
                'svg_indices', torch.tensor(svg_indices, dtype=torch.long)
            )
        else:
            self.svg_indices = None

    def forward(self, graph, he_rep, x, agg_mtx=None, selection=None):
        gate = self.attention_gate(he_rep)
        he_rep_gated = he_rep * gate
        he_rep_gated = self.mlp(he_rep_gated)
        enc = self.mod(he_rep_gated, graph)
        x_prime = F.leaky_relu(self.linear(F.leaky_relu(enc)))

        if self.agg:
            pred_full = torch.sparse.mm(agg_mtx, x_prime[selection])
        else:
            pred_full = x_prime

        if self.svg_indices is not None:
            pred_svg = pred_full[:, self.svg_indices]
            true_svg = x[:, self.svg_indices]
            loss = self.criterion(pred_svg, true_svg)
        else:
            loss = self.criterion(pred_full, x)

        return loss, x_prime, enc

    def predict(self, graph, he_rep):
        gate = self.attention_gate(he_rep)
        he_rep_gated = he_rep * gate
        he_rep_gated = self.mlp(he_rep_gated)
        enc = self.mod(he_rep_gated, graph)
        return F.leaky_relu(self.linear(F.leaky_relu(enc)))


# =============================================
# 通用 Model wrapper
# =============================================

class FeatureSelectModel(nn.Module):
    """通用 Model wrapper：接受任意 Predictor_spot 子类 + 原版 DGI。"""

    def __init__(self, predictor: nn.Module, in_dim: int, hidden_dim: int,
                 out_dim: int, device: str = 'cpu'):
        super().__init__()
        self.predictor = predictor
        self.dgi_model = Predictor_dgi(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim
        )
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


# =============================================
# FeatureSelectSpatialEx — 通用包装 Trainer
# =============================================

class FeatureSelectSpatialEx(SpatialEx):
    """通用 FeatureSelect Trainer：接受 method 参数决定使用哪种特征选择。

    Args:
        method: 'svg' | 'attn' | 'combined'
        svg_top_k: SVG 方法中选取的基因数量。
        gate_hidden_dim: Attention Gate 隐藏维度。
        gate_dropout: Attention Gate dropout。
    """

    def __init__(self, adata1, adata2, graph1, graph2,
                 method: str = 'attn',
                 svg_top_k: int = 150,
                 gate_hidden_dim: int = 256,
                 gate_dropout: float = 0.1,
                 **kwargs):
        # 过滤掉自定义参数，避免传入 SpatialEx.__init__()
        spatialex_kwargs = {k: v for k, v in kwargs.items()
                           if k not in ('attn_gate_hidden', 'attn_gate_dropout',
                                        'svg_top_k_override', 'num_neighbors')}
        super().__init__(adata1, adata2, graph1, graph2, **spatialex_kwargs)

        self.method = method
        svg_idx1 = svg_idx2 = None

        if method in ('svg', 'combined'):
            expr1 = (np.asarray(adata1.X.todense())
                     if hasattr(adata1.X, 'toarray') else np.asarray(adata1.X))
            expr2 = (np.asarray(adata2.X.todense())
                     if hasattr(adata2.X, 'toarray') else np.asarray(adata2.X))
            coords1 = adata1.obsm['spatial']
            coords2 = adata2.obsm['spatial']

            logger.info("Computing Moran's I for slice 1 (%d cells, %d genes)...",
                        expr1.shape[0], expr1.shape[1])
            mi1 = compute_morans_i(expr1, coords1, k=kwargs.get('num_neighbors', 7))
            svg_idx1 = select_svg_indices(mi1, top_k=min(svg_top_k, expr1.shape[1]))
            np.save(os.path.join(output_dir, 'slice1_morans_i.npy'), mi1)
            np.save(os.path.join(output_dir, 'slice1_svg_indices.npy'), svg_idx1)

            logger.info("Computing Moran's I for slice 2 (%d cells, %d genes)...",
                        expr2.shape[0], expr2.shape[1])
            mi2 = compute_morans_i(expr2, coords2, k=kwargs.get('num_neighbors', 7))
            svg_idx2 = select_svg_indices(mi2, top_k=min(svg_top_k, expr2.shape[1]))
            np.save(os.path.join(output_dir, 'slice2_morans_i.npy'), mi2)
            np.save(os.path.join(output_dir, 'slice2_svg_indices.npy'), svg_idx2)

        def _make_predictor(in_dim, hidden_dim, out_dim, svg_idx):
            if method == 'svg':
                return SVGPredictor_spot(
                    in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                    num_layers=self.num_layers, svg_indices=svg_idx,
                )
            elif method == 'attn':
                return AttentionGatePredictor_spot(
                    in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                    num_layers=self.num_layers,
                    gate_hidden_dim=gate_hidden_dim, gate_dropout=gate_dropout,
                )
            elif method == 'combined':
                return CombinedPredictor_spot(
                    in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
                    num_layers=self.num_layers, svg_indices=svg_idx,
                    gate_hidden_dim=gate_hidden_dim, gate_dropout=gate_dropout,
                )
            else:
                raise ValueError(f"Unknown method: {method}")

        pred_A = _make_predictor(self.in_dim1, self.hidden_dim, self.out_dim1, svg_idx1)
        pred_B = _make_predictor(self.in_dim2, self.hidden_dim, self.out_dim2, svg_idx2)

        self.module_HA = FeatureSelectModel(
            pred_A, self.in_dim1, self.hidden_dim, self.out_dim1, self.device
        )
        self.module_HB = FeatureSelectModel(
            pred_B, self.in_dim2, self.hidden_dim, self.out_dim2, self.device
        )
        self.models = [self.module_HA, self.module_HB]
        self.optimizer = create_optimizer(
            kwargs.get('optimizer', 'adam'), self.models, self.lr, self.weight_decay
        )
        logger.info("FeatureSelectSpatialEx initialized | method=%s", method)


# =============================================
# 评估函数
# =============================================

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


# =============================================
# 1. 数据读取与预处理
# =============================================
print("=" * 60)
print("Stage 1: Preprocessing Slice 1")
print("=" * 60)

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

print("=" * 60)
print("Stage 2: Preprocessing Slice 2")
print("=" * 60)

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

graph1 = se.pp.Build_hypergraph_spatial_and_HE(
    adata1, num_neighbors, graph_kind='spatial', return_type='csr'
)
graph2 = se.pp.Build_hypergraph_spatial_and_HE(
    adata2, num_neighbors, graph_kind='spatial', return_type='csr'
)
print('[OK] Hypergraphs built\n')

# =============================================
# 2. 运行实验（单模式或消融）
# =============================================

ABLATION_METHODS = {
    '0_baseline': None,
    '1_svg': 'svg',
    '2_attn': 'attn',
    '3_combined': 'combined',
}

if METHOD != 'ablation':
    # ---- 单一方法 ----
    print("=" * 60)
    print(f"Stage 3: Training with method='{METHOD}'")
    print("=" * 60)

    t_start = time.time()
    trainer = FeatureSelectSpatialEx(
        adata1, adata2, graph1, graph2,
        method=METHOD,
        epochs=epochs, device=device,
        svg_top_k=SVG_TOP_K,
        gate_hidden_dim=ATTN_GATE_HIDDEN,
        gate_dropout=ATTN_GATE_DROPOUT,
        num_neighbors=num_neighbors,
    )
    trainer.train()
    logger.info("Training time: %.1f min", (time.time() - t_start) / 60)

    panelB1, panelA2 = trainer.auto_inference()
    results = evaluate(adata1, adata2, panelB1, panelA2)

    print(f'[{METHOD}] Slice 1 — PCC: {results["slice1"]["PCC"]:.6f}  '
          f'SSIM: {results["slice1"]["SSIM"]:.6f}  '
          f'CMD: {results["slice1"]["CMD"]:.6f}')
    print(f'[{METHOD}] Slice 2 — PCC: {results["slice2"]["PCC"]:.6f}  '
          f'SSIM: {results["slice2"]["SSIM"]:.6f}  '
          f'CMD: {results["slice2"]["CMD"]:.6f}')

    np.save(os.path.join(output_dir, 'slice1_pcc_per_gene.npy'), results['slice1']['pcc_per_gene'])
    np.save(os.path.join(output_dir, 'slice1_ssim_per_gene.npy'), results['slice1']['ssim_per_gene'])
    np.save(os.path.join(output_dir, 'slice2_pcc_per_gene.npy'), results['slice2']['pcc_per_gene'])
    np.save(os.path.join(output_dir, 'slice2_ssim_per_gene.npy'), results['slice2']['ssim_per_gene'])

    metrics = {
        'slice1_prediction': {k: v for k, v in results['slice1'].items()
                               if k in ('PCC', 'SSIM', 'CMD', 'MoransI', 'SPCC')},
        'slice2_prediction': {k: v for k, v in results['slice2'].items()
                               if k in ('PCC', 'SSIM', 'CMD', 'MoransI', 'SPCC')},
        'method': METHOD,
        'svg_top_k': SVG_TOP_K,
        'gate_hidden_dim': ATTN_GATE_HIDDEN,
    }
    with open(os.path.join(output_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

else:
    # ---- 消融：对比四种设置 ----
    print("=" * 70)
    print("Stage 3: Feature Selection Ablation Study (4 variants)")
    print("=" * 70)

    all_results = {}

    for variant_name, method in ABLATION_METHODS.items():
        print(f"\n{'='*70}")
        print(f"Variant: {variant_name} | method={method}")
        print('=' * 70)

        variant_dir = os.path.join(output_dir, variant_name)
        os.makedirs(variant_dir, exist_ok=True)

        # Overwrite 检查
        if not OVERWRITE and os.path.exists(os.path.join(variant_dir, 'metrics.json')):
            logger.info("Skipping %s (already exists, OVERWRITE=False)", variant_name)
            with open(os.path.join(variant_dir, 'metrics.json')) as f:
                all_results[variant_name] = json.load(f)
            continue

        t_start = time.time()

        if method is None:
            trainer = se.SpatialEx(
                adata1, adata2, graph1, graph2,
                epochs=epochs, device=device
            )
        else:
            trainer = FeatureSelectSpatialEx(
                adata1, adata2, graph1, graph2,
                method=method,
                epochs=epochs, device=device,
                svg_top_k=SVG_TOP_K,
                gate_hidden_dim=ATTN_GATE_HIDDEN,
                gate_dropout=ATTN_GATE_DROPOUT,
                num_neighbors=num_neighbors,
            )

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

        np.save(os.path.join(variant_dir, 'slice1_pcc_per_gene.npy'), results['slice1']['pcc_per_gene'])
        np.save(os.path.join(variant_dir, 'slice1_ssim_per_gene.npy'), results['slice1']['ssim_per_gene'])
        np.save(os.path.join(variant_dir, 'slice2_pcc_per_gene.npy'), results['slice2']['pcc_per_gene'])
        np.save(os.path.join(variant_dir, 'slice2_ssim_per_gene.npy'), results['slice2']['ssim_per_gene'])
        np.save(os.path.join(variant_dir, 'panelB1.npy'), panelB1)
        np.save(os.path.join(variant_dir, 'panelA2.npy'), panelA2)

        save_res = {
            'slice1': {k: v for k, v in results['slice1'].items()
                       if k in ('PCC', 'SSIM', 'CMD', 'MoransI', 'SPCC')},
            'slice2': {k: v for k, v in results['slice2'].items()
                       if k in ('PCC', 'SSIM', 'CMD', 'MoransI', 'SPCC')},
            'method': method,
            'time_seconds': t_elapsed,
        }
        with open(os.path.join(variant_dir, 'metrics.json'), 'w') as f:
            json.dump(save_res, f, indent=2)

        all_results[variant_name] = save_res

        del trainer, panelB1, panelA2
        torch.cuda.empty_cache()
        gc.collect()

    # 打印对比表
    print("\n\n" + "=" * 90)
    print("FEATURE SELECTION ABLATION SUMMARY")
    print("=" * 90)
    header = (f"{'Variant':<18} {'Method':<10} {'PCC_s1':>8} {'PCC_s2':>8} "
              f"{'SSIM_s1':>8} {'SSIM_s2':>8} {'CMD_s1':>8} {'CMD_s2':>8}")
    print(header)
    print("-" * len(header))

    bl = all_results.get('0_baseline', {})
    for name, res in all_results.items():
        s1, s2 = res['slice1'], res['slice2']
        m = str(res.get('method', 'baseline'))
        print(f"{name:<18} {m:<10} "
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
            m = str(res.get('method', ''))
            print(f"{name:<18} {m:<10} "
                  f"{s1['PCC']-b1['PCC']:>+8.4f} {s2['PCC']-b2['PCC']:>+8.4f} "
                  f"{s1['SSIM']-b1['SSIM']:>+8.4f} {s2['SSIM']-b2['SSIM']:>+8.4f} "
                  f"{s1['CMD']-b1['CMD']:>+8.4f} {s2['CMD']-b2['CMD']:>+8.4f}")

    # Wilcoxon 显著性检验
    from scipy import stats as scipy_stats
    print("\n\nStatistical Significance (Wilcoxon vs baseline):")
    bl_dir = os.path.join(output_dir, '0_baseline')

    for variant_name in ABLATION_METHODS:
        if variant_name == '0_baseline':
            continue
        v_dir = os.path.join(output_dir, variant_name)
        print(f"\n{variant_name}:")
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
                            print(f"  Slice {slice_id} {metric_name.upper()}: "
                                  f"test failed ({e})")

    with open(os.path.join(output_dir, 'ablation_summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # 保存最佳方法（combined）的顶层 metrics
    if '3_combined' in all_results:
        best = all_results['3_combined']
        metrics_top = {
            'slice1_prediction': best['slice1'],
            'slice2_prediction': best['slice2'],
            'method': 'combined',
            'svg_top_k': SVG_TOP_K,
            'gate_hidden_dim': ATTN_GATE_HIDDEN,
        }
        with open(os.path.join(output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_top, f, indent=2)

# =============================================
# 与 baseline 对比
# =============================================
print("\n" + "=" * 60)
print("Stage 4: Comparison with External Baseline")
print("=" * 60)

baseline_metrics_path = os.path.join(baseline_dir, 'metrics_summary.json')
top_metrics_path = os.path.join(output_dir, 'metrics_summary.json')

if os.path.exists(baseline_metrics_path) and os.path.exists(top_metrics_path):
    from scipy import stats as scipy_stats
    with open(baseline_metrics_path) as f:
        baseline_metrics = json.load(f)
    with open(top_metrics_path) as f:
        our_metrics = json.load(f)

    best_method = our_metrics.get('method', METHOD)
    print(f"Comparing best method '{best_method}' vs external baseline:\n")

    print(f"{'Metric':<8} {'Slice':<18} {'Baseline':>10} {'Ours':>10} {'Diff':>10} {'Dir':>8}")
    print("-" * 68)
    for slice_key in ['slice1_prediction', 'slice2_prediction']:
        bl_key = slice_key.replace('_prediction', '')
        bl_key = bl_key if bl_key in baseline_metrics else slice_key
        if bl_key not in baseline_metrics:
            continue
        for metric in ['PCC', 'SSIM', 'CMD']:
            bl_val = baseline_metrics[bl_key][metric]
            our_val = our_metrics[slice_key][metric]
            diff = our_val - bl_val
            direction = ('better' if diff < 0 else 'worse') if metric == 'CMD' else (
                'better' if diff > 0 else 'worse'
            )
            print(f"{metric:<8} {slice_key:<18} {bl_val:>10.6f} "
                  f"{our_val:>10.6f} {diff:>+10.6f} {direction:>8}")
else:
    print(f"Warning: baseline or output metrics not found.")

print(f'\n[SAVED] All results to: {output_dir}')
print("=" * 60)
print("Breast Cancer DONE!")
print("=" * 60)


# =============================================
# PHASE 2: Skin Melanoma Dataset（完整消融，与 MG 一致）
# =============================================
print("\n\n" + "#" * 70)
print("# PHASE 2: SKIN MELANOMA DATASET (spatial split, full ablation)")
print("#" * 70)

skin_adata1, skin_adata2, skin_graph1, skin_graph2 = preprocess_skin_data()

SCALAR_KEYS = ('PCC', 'SSIM', 'CMD', 'MoransI', 'SPCC')
skin_all_results = {}

for variant_name, method in ABLATION_METHODS.items():
    print(f"\n{'='*70}")
    print(f"[SKIN] Variant: {variant_name} | method={method}")
    print('=' * 70)

    variant_dir = os.path.join(output_dir_skin, variant_name)
    os.makedirs(variant_dir, exist_ok=True)

    # Overwrite 检查
    if not OVERWRITE and os.path.exists(os.path.join(variant_dir, 'metrics.json')):
        logger.info("[SKIN] Skipping %s (already exists, OVERWRITE=False)", variant_name)
        with open(os.path.join(variant_dir, 'metrics.json')) as f:
            skin_all_results[variant_name] = json.load(f)
        continue

    t_start = time.time()

    if method is None:
        skin_trainer = se.SpatialEx(
            skin_adata1, skin_adata2, skin_graph1, skin_graph2,
            epochs=epochs, device=device,
        )
    else:
        skin_trainer = FeatureSelectSpatialEx(
            skin_adata1, skin_adata2, skin_graph1, skin_graph2,
            method=method,
            epochs=epochs, device=device,
            svg_top_k=SVG_TOP_K,
            gate_hidden_dim=ATTN_GATE_HIDDEN,
            gate_dropout=ATTN_GATE_DROPOUT,
        )

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
                np.save(os.path.join(variant_dir, f'slice{slice_id}_{m_name}_per_gene.npy'),
                        skin_results[s_key][arr_key])
    np.save(os.path.join(variant_dir, 'panelB1.npy'), panelB1_skin)
    np.save(os.path.join(variant_dir, 'panelA2.npy'), panelA2_skin)

    save_res = {
        'slice1': {k: v for k, v in s1.items() if k in SCALAR_KEYS},
        'slice2': {k: v for k, v in s2.items() if k in SCALAR_KEYS},
        'method': method,
        'time_seconds': t_elapsed,
    }
    with open(os.path.join(variant_dir, 'metrics.json'), 'w') as f:
        json.dump(save_res, f, indent=2)

    skin_all_results[variant_name] = save_res

    del skin_trainer, panelB1_skin, panelA2_skin
    torch.cuda.empty_cache()
    gc.collect()

# Skin 汇总表
print("\n\n" + "=" * 90)
print("[SKIN] FEATURE SELECTION ABLATION SUMMARY")
print("=" * 90)
header = (f"{'Variant':<18} {'Method':<10} {'PCC_s1':>8} {'PCC_s2':>8} "
          f"{'SSIM_s1':>8} {'SSIM_s2':>8} {'CMD_s1':>8} {'CMD_s2':>8}")
print(header)
print("-" * len(header))

skin_bl = skin_all_results.get('0_baseline', {})
for name, res in skin_all_results.items():
    s1, s2 = res['slice1'], res['slice2']
    m = str(res.get('method', 'baseline'))
    print(f"{name:<18} {m:<10} "
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
        m = str(res.get('method', ''))
        print(f"{name:<18} {m:<10} "
              f"{s1['PCC']-b1['PCC']:>+8.4f} {s2['PCC']-b2['PCC']:>+8.4f} "
              f"{s1['SSIM']-b1['SSIM']:>+8.4f} {s2['SSIM']-b2['SSIM']:>+8.4f} "
              f"{s1['CMD']-b1['CMD']:>+8.4f} {s2['CMD']-b2['CMD']:>+8.4f}")

# Skin Wilcoxon
print("\n\n[SKIN] Statistical Significance (Wilcoxon vs baseline):")
skin_bl_dir = os.path.join(output_dir_skin, '0_baseline')
for variant_name in ABLATION_METHODS:
    if variant_name == '0_baseline':
        continue
    v_dir = os.path.join(output_dir_skin, variant_name)
    print(f"\n{variant_name}:")
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

if '3_combined' in skin_all_results:
    best = skin_all_results['3_combined']
    metrics_top = {
        'slice1_prediction': best['slice1'],
        'slice2_prediction': best['slice2'],
        'method': 'combined',
    }
    with open(os.path.join(output_dir_skin, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics_top, f, indent=2)

print(f'\n[SAVED] Skin results to: {output_dir_skin}')
print("=" * 70)
print("ALL DATASETS COMPLETE!")
print("=" * 70)
