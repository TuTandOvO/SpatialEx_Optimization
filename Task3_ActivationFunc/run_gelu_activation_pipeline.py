#!/usr/bin/env python3
"""
SpatialEx — GELU Activation Pipeline (HPC)
============================================
改进点：将预测头中的 LeakyReLU 激活函数替换为 GELU（Gaussian Error Linear Unit）。

**动机**：
  原版 SpatialEx 的预测头（Predictor_spot）使用 LeakyReLU(0.1)：
    - MLP 层后：nn.LeakyReLU(0.1)
    - 最终输出：F.leaky_relu(self.linear(F.leaky_relu(enc)))

  GELU 是 BERT、ViT、GPT 等 Transformer 模型中广泛使用的激活函数，具有：
    - 平滑梯度（处处可微，避免 LeakyReLU 的"折点"梯度不连续）
    - 隐式正则化效果（通过高斯 CDF 门控，接近零的激活被随机抑制）
    - 与 UNI ViT 特征的兼容性更好（UNI 内部也使用 GELU）

  对于从 UNI ViT 提取的 1024 维特征，使用 GELU 可能减少训练初期的激活偏移，
  尤其是 BatchNorm 之后，因为 GELU 在 0 附近更平滑的梯度有助于稳定训练。

**实现策略**：
  1. 修改 MLP 中的 LeakyReLU → GELU
  2. 修改最终输出层的 F.leaky_relu → F.gelu
  3. HGNN 内部激活由 'prelu' 改为 'gelu'（需要 patch create_activation）
  4. DGI 保持原版不变（DGI 内部使用 PReLU，已经是参数化激活，保持稳定）

**预期提升**：
  - 更平滑的梯度流 → 在 BatchNorm 之后表现更稳定
  - 与 UNI 特征分布更兼容（UNI 内部 GELU）→ 可能减少训练初期损失震荡
  - GELU 的随机 dropout 特性 → 轻微正则化效果 → 改善泛化

用法：
    python run_gelu_activation_pipeline.py
"""

import os
import json
import logging
import time

import numpy as np
import pandas as pd
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

output_dir = '/gpfsdata/home/renyixiang/YuanLab/data/MG_Xenium/SpatialEx_results_gelu_MG/'
os.makedirs(output_dir, exist_ok=True)

baseline_dir = '/gpfsdata/home/renyixiang/YuanLab/data/MG_Xenium/SpatialEx_results/'

# ---- Skin Melanoma 数据配置（单切片 → 空间切分为两半）----
skin_root = '/gpfsdata/home/renyixiang/YuanLab/data/Skin_Xenium/Human_Skin_Melanoma_Base_FFPE/'
output_dir_skin = '/gpfsdata/home/renyixiang/YuanLab/data/Skin_Xenium/SpatialEx_results_gelu_skin/'
os.makedirs(output_dir_skin, exist_ok=True)


def spatial_split_adata(adata, axis=0):
    """沿空间坐标中位数将单切片分成两半（论文 Extended Data Fig. 2B 做法）。"""
    coords = adata.obsm['spatial']
    median_val = np.median(coords[:, axis])
    mask1 = coords[:, axis] <= median_val
    mask2 = coords[:, axis] > median_val
    adata1 = adata[mask1].copy()
    adata2 = adata[mask2].copy()
    logger.info("Spatial split: %d + %d cells (axis=%d, median=%.1f)",
                adata1.n_obs, adata2.n_obs, axis, median_val)
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

# 激活函数选择：'gelu' | 'silu' | 'mish'
# gelu: 与 Transformer 系模型最兼容
# silu: SiLU/Swish，与 gelu 相近，在某些视觉任务更优
# mish: Mish 激活，更平滑但计算稍重
ACTIVATION_TYPE = 'gelu'


# =============================================
# Patch：向 SpatialEx 的 create_activation 添加 GELU/SiLU/Mish 支持
# =============================================
# 注意：SpatialEx 的 create_activation 只支持 relu/elu/leaky_relu/prelu，
# 这里我们 monkeypatch 以支持 gelu/silu/mish。

import SpatialEx.utils as _se_utils

_original_create_activation = _se_utils.create_activation


def _patched_create_activation(name: str) -> nn.Module:
    """扩展版 create_activation，增加 GELU、SiLU、Mish 支持。"""
    name = name.lower()
    if name == 'gelu':
        return nn.GELU()
    elif name in ('silu', 'swish'):
        return nn.SiLU()
    elif name == 'mish':
        return nn.Mish()
    else:
        return _original_create_activation(name)


# 动态替换 HGNN 所依赖的 create_activation
_se_utils.create_activation = _patched_create_activation

# 同时需要替换 model 模块中的引用（因为 model.py 已经 import 了）
import SpatialEx.model as _se_model
_se_model.create_activation = _patched_create_activation

logger.info("Patched create_activation to support GELU/SiLU/Mish")


# =============================================
# GELU HGNN（直接使用 patch 后的 HGNN，传入 activation='gelu'）
# =============================================

class GELUPredictor_spot(nn.Module):
    """预测头：使用 GELU 替代所有 LeakyReLU 激活函数。

    改动：
      1. MLP 中的 nn.LeakyReLU(0.1) → nn.GELU()
      2. forward 中的 F.leaky_relu(enc) → F.gelu(enc)
      3. forward 中的最终 F.leaky_relu → F.gelu
      4. HGNN 内部激活 'prelu' → activation_type（默认 'gelu'）

    Args:
        in_dim: 输入特征维度。
        hidden_dim: 隐藏层维度。
        out_dim: 输出维度（基因数）。
        num_layers: HGNN 层数。
        activation_type: HGNN 内部激活函数名称（'gelu', 'silu', 'mish'）。
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int, activation_type: str = 'gelu'):
        super().__init__()
        self.agg = True

        # MLP: LeakyReLU → GELU
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
        )

        # HGNN: 使用 patched create_activation，传入 activation_type
        self.mod = HGNN(
            in_dim=hidden_dim, num_hidden=hidden_dim, out_dim=hidden_dim,
            num_layers=num_layers, dropout=0, activation=activation_type,
        )

        self.linear = nn.Linear(hidden_dim, out_dim)
        self.criterion = nn.MSELoss()
        self.activation_type = activation_type

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        """根据激活类型对张量应用激活。"""
        if self.activation_type == 'gelu':
            return F.gelu(x)
        elif self.activation_type in ('silu', 'swish'):
            return F.silu(x)
        elif self.activation_type == 'mish':
            return F.mish(x)
        else:
            return F.gelu(x)

    def forward(self, graph, he_rep, x, agg_mtx=None, selection=None):
        he_rep = self.mlp(he_rep)
        enc = self.mod(he_rep, graph)
        # 原版: F.leaky_relu(self.linear(F.leaky_relu(enc)))
        x_prime = self._apply_activation(self.linear(self._apply_activation(enc)))
        if self.agg:
            pred = torch.sparse.mm(agg_mtx, x_prime[selection])
        else:
            pred = x_prime
        loss = self.criterion(pred, x)
        return loss, x_prime, enc

    def predict(self, graph, he_rep):
        he_rep = self.mlp(he_rep)
        enc = self.mod(he_rep, graph)
        return self._apply_activation(self.linear(self._apply_activation(enc)))


class GELUModel(nn.Module):
    """GELU 模型：GELUPredictor_spot + 原版 Predictor_dgi。"""

    def __init__(self, num_layers=2, in_dim=2048, hidden_dim=512, out_dim=150,
                 activation_type='gelu', device='cpu'):
        super().__init__()
        self.predictor = GELUPredictor_spot(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
            num_layers=num_layers, activation_type=activation_type,
        )
        self.dgi_model = Predictor_dgi(
            in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
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


class GELUSpatialEx(SpatialEx):
    """继承原版 SpatialEx trainer，仅将预测头的激活函数改为 GELU。

    改动极为最小化：
      - 预测头激活：LeakyReLU(0.1) → GELU（MLP + 输出层）
      - HGNN 内部激活：prelu → gelu
      - DGI 保持原版（PReLU）
      - 损失函数保持原版 MSE
      - 所有超参数与 baseline 完全一致

    用法与原版完全一致：
        trainer = GELUSpatialEx(adata1, adata2, graph1, graph2, ...)
        trainer.train()
        panelB1, panelA2 = trainer.auto_inference()
    """

    def __init__(self, adata1, adata2, graph1, graph2,
                 activation_type: str = 'gelu', **kwargs):
        super().__init__(adata1, adata2, graph1, graph2, **kwargs)

        self.module_HA = GELUModel(
            num_layers=self.num_layers, in_dim=self.in_dim1,
            hidden_dim=self.hidden_dim, out_dim=self.out_dim1,
            activation_type=activation_type, device=self.device,
        )
        self.module_HB = GELUModel(
            num_layers=self.num_layers, in_dim=self.in_dim2,
            hidden_dim=self.hidden_dim, out_dim=self.out_dim2,
            activation_type=activation_type, device=self.device,
        )
        self.models = [self.module_HA, self.module_HB]
        self.optimizer = create_optimizer(
            kwargs.get('optimizer', 'adam'), self.models, self.lr, self.weight_decay
        )
        logger.info("GELUSpatialEx initialized | activation=%s", activation_type)


# =============================================
# 消融：激活函数对比（可选，设为 False 可只跑主实验）
# =============================================
RUN_ABLATION = True  # 是否对比多种激活函数

ACTIVATION_VARIANTS = {
    'leakyrelu_baseline': None,    # 原版，用作内部对照
    'gelu': 'gelu',
    'silu': 'silu',
    'mish': 'mish',
}


def run_single_activation(adata1, adata2, graph1, graph2, activation_type=None):
    """训练单个激活函数变体并返回评估结果。"""
    if activation_type is None:
        trainer = se.SpatialEx(adata1, adata2, graph1, graph2,
                                epochs=epochs, device=device)
    else:
        trainer = GELUSpatialEx(
            adata1, adata2, graph1, graph2,
            epochs=epochs, device=device,
            activation_type=activation_type,
        )
    trainer.train()
    panelB1, panelA2 = trainer.auto_inference()
    return trainer, panelB1, panelA2


def evaluate(adata1, adata2, panelB1, panelA2):
    """计算两个 slice 的评估指标（PCC, SSIM, CMD, MoransI, SPCC）。"""
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

# =============================================
# 2. 主实验：GELU 激活
# =============================================
if not RUN_ABLATION:
    print("=" * 60)
    print(f"Stage 3: Training with {ACTIVATION_TYPE.upper()} activation")
    print("=" * 60)

    t_start = time.time()
    spatialex = GELUSpatialEx(
        adata1, adata2, graph1, graph2,
        epochs=epochs, device=device,
        activation_type=ACTIVATION_TYPE,
    )
    spatialex.train()
    logger.info("Training time: %.1f min", (time.time() - t_start) / 60)

    panelB1, panelA2 = spatialex.auto_inference()
    results = evaluate(adata1, adata2, panelB1, panelA2)

    metrics = {
        'slice1_prediction': {
            'PCC': results['slice1']['PCC'],
            'SSIM': results['slice1']['SSIM'],
            'CMD': results['slice1']['CMD'],
        },
        'slice2_prediction': {
            'PCC': results['slice2']['PCC'],
            'SSIM': results['slice2']['SSIM'],
            'CMD': results['slice2']['CMD'],
        },
        'activation_type': ACTIVATION_TYPE,
    }

    np.save(os.path.join(output_dir, 'slice1_pcc_per_gene.npy'), results['slice1']['pcc_per_gene'])
    np.save(os.path.join(output_dir, 'slice1_ssim_per_gene.npy'), results['slice1']['ssim_per_gene'])
    np.save(os.path.join(output_dir, 'slice2_pcc_per_gene.npy'), results['slice2']['pcc_per_gene'])
    np.save(os.path.join(output_dir, 'slice2_ssim_per_gene.npy'), results['slice2']['ssim_per_gene'])

    print(f'[{ACTIVATION_TYPE}] Slice 1 — PCC: {metrics["slice1_prediction"]["PCC"]:.6f}  '
          f'SSIM: {metrics["slice1_prediction"]["SSIM"]:.6f}  '
          f'CMD: {metrics["slice1_prediction"]["CMD"]:.6f}')
    print(f'[{ACTIVATION_TYPE}] Slice 2 — PCC: {metrics["slice2_prediction"]["PCC"]:.6f}  '
          f'SSIM: {metrics["slice2_prediction"]["SSIM"]:.6f}  '
          f'CMD: {metrics["slice2_prediction"]["CMD"]:.6f}')

    with open(os.path.join(output_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

# =============================================
# 3. 消融：多种激活函数对比
# =============================================
else:
    print("=" * 60)
    print("Stage 3: Activation Function Ablation Study")
    print("=" * 60)

    import gc
    all_results = {}

    for variant_name, act_type in ACTIVATION_VARIANTS.items():
        print(f"\n[Activation Ablation] Variant: {variant_name}")
        variant_dir = os.path.join(output_dir, variant_name)
        os.makedirs(variant_dir, exist_ok=True)

        t_start = time.time()
        trainer, panelB1, panelA2 = run_single_activation(
            adata1, adata2, graph1, graph2, act_type
        )
        t_elapsed = time.time() - t_start
        results = evaluate(adata1, adata2, panelB1, panelA2)

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
            'activation_type': act_type,
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
    print("ACTIVATION FUNCTION ABLATION SUMMARY")
    print("=" * 90)
    header = (f"{'Activation':<22} {'PCC_s1':>8} {'PCC_s2':>8} "
              f"{'SSIM_s1':>8} {'SSIM_s2':>8} {'CMD_s1':>8} {'CMD_s2':>8}")
    print(header)
    print("-" * len(header))

    bl = all_results.get('leakyrelu_baseline', {})

    for name, res in all_results.items():
        s1, s2 = res['slice1'], res['slice2']
        print(f"{name:<22} "
              f"{s1['PCC']:>8.4f} {s2['PCC']:>8.4f} "
              f"{s1['SSIM']:>8.4f} {s2['SSIM']:>8.4f} "
              f"{s1['CMD']:>8.4f} {s2['CMD']:>8.4f}")

    if bl:
        print("\nDIFF vs LeakyReLU baseline:")
        print("-" * len(header))
        for name, res in all_results.items():
            if name == 'leakyrelu_baseline':
                continue
            s1, s2 = res['slice1'], res['slice2']
            b1, b2 = bl['slice1'], bl['slice2']
            print(f"{name:<22} "
                  f"{s1['PCC']-b1['PCC']:>+8.4f} {s2['PCC']-b2['PCC']:>+8.4f} "
                  f"{s1['SSIM']-b1['SSIM']:>+8.4f} {s2['SSIM']-b2['SSIM']:>+8.4f} "
                  f"{s1['CMD']-b1['CMD']:>+8.4f} {s2['CMD']-b2['CMD']:>+8.4f}")

    # Wilcoxon 检验（GELU vs LeakyReLU baseline）
    from scipy import stats as scipy_stats
    print("\n\nStatistical significance (Wilcoxon, GELU vs LeakyReLU):")
    bl_dir = os.path.join(output_dir, 'leakyrelu_baseline')
    gelu_dir = os.path.join(output_dir, 'gelu')
    for slice_id in [1, 2]:
        for metric_name in ['pcc', 'ssim']:
            bl_path = os.path.join(bl_dir, f'slice{slice_id}_{metric_name}_per_gene.npy')
            gelu_path = os.path.join(gelu_dir, f'slice{slice_id}_{metric_name}_per_gene.npy')
            if os.path.exists(bl_path) and os.path.exists(gelu_path):
                bl_vals = np.load(bl_path)
                gelu_vals = np.load(gelu_path)
                min_len = min(len(bl_vals), len(gelu_vals))
                mask = ~(np.isnan(bl_vals[:min_len]) | np.isnan(gelu_vals[:min_len]))
                bl_c, gelu_c = bl_vals[:min_len][mask], gelu_vals[:min_len][mask]
                if len(bl_c) > 10:
                    try:
                        stat, pval = scipy_stats.wilcoxon(gelu_c, bl_c)
                        mean_diff = np.mean(gelu_c - bl_c)
                        sig = ('*** p<0.001' if pval < 0.001 else
                               '** p<0.01' if pval < 0.01 else
                               '* p<0.05' if pval < 0.05 else 'n.s.')
                        print(f"  Slice {slice_id} {metric_name.upper()}: "
                              f"mean_diff={mean_diff:+.6f}, p={pval:.4e} {sig}")
                    except Exception as e:
                        print(f"  Slice {slice_id} {metric_name.upper()}: test failed ({e})")

    # 保存完整对比结果
    with open(os.path.join(output_dir, 'ablation_summary.json'), 'w') as f:
        save_all = {k: {kk: vv for kk, vv in v.items()
                        if kk not in ('pcc_per_gene', 'ssim_per_gene')}
                    for k, v in all_results.items()}
        json.dump(save_all, f, indent=2)

    # 同时保存主激活结果到顶层目录（方便与其他任务对比）
    if 'gelu' in all_results:
        main_res = all_results['gelu']
        metrics_top = {
            'slice1_prediction': main_res['slice1'],
            'slice2_prediction': main_res['slice2'],
            'activation_type': 'gelu',
        }
        with open(os.path.join(output_dir, 'metrics_summary.json'), 'w') as f:
            json.dump(metrics_top, f, indent=2)

print(f'\n[SAVED] All results to: {output_dir}')
print("=" * 60)
print("DONE — GELU Activation Pipeline Complete!")
print("=" * 60)


# =============================================
# PHASE 2: Skin Melanoma Dataset（空间切分）
# =============================================
print("\n\n" + "#" * 70)
print("# PHASE 2: SKIN MELANOMA DATASET (spatial split)")
print("#" * 70)

skin_adata1, skin_adata2, skin_graph1, skin_graph2 = preprocess_skin_data()

skin_all_results = {}

for variant_name, act_type in ACTIVATION_VARIANTS.items():
    print(f"\n[SKIN Activation] Variant: {variant_name}")
    variant_dir = os.path.join(output_dir_skin, variant_name)
    os.makedirs(variant_dir, exist_ok=True)

    t_start = time.time()
    trainer, panelB1, panelA2 = run_single_activation(
        skin_adata1, skin_adata2, skin_graph1, skin_graph2, act_type
    )
    t_elapsed = time.time() - t_start
    results = evaluate(skin_adata1, skin_adata2, panelB1, panelA2)

    print(f"  Slice 1 — PCC: {results['slice1']['PCC']:.6f}  "
          f"SSIM: {results['slice1']['SSIM']:.6f}  CMD: {results['slice1']['CMD']:.6f}")
    print(f"  Slice 2 — PCC: {results['slice2']['PCC']:.6f}  "
          f"SSIM: {results['slice2']['SSIM']:.6f}  CMD: {results['slice2']['CMD']:.6f}")
    print(f"  Time: {t_elapsed/60:.1f} min")

    np.save(os.path.join(variant_dir, 'slice1_pcc_per_gene.npy'), results['slice1']['pcc_per_gene'])
    np.save(os.path.join(variant_dir, 'slice1_ssim_per_gene.npy'), results['slice1']['ssim_per_gene'])
    np.save(os.path.join(variant_dir, 'slice2_pcc_per_gene.npy'), results['slice2']['pcc_per_gene'])
    np.save(os.path.join(variant_dir, 'slice2_ssim_per_gene.npy'), results['slice2']['ssim_per_gene'])
    np.save(os.path.join(variant_dir, 'panelB1.npy'), panelB1)
    np.save(os.path.join(variant_dir, 'panelA2.npy'), panelA2)

    save_res = {
        'slice1': {k: v for k, v in results['slice1'].items() if k in ('PCC', 'SSIM', 'CMD', 'MoransI', 'SPCC')},
        'slice2': {k: v for k, v in results['slice2'].items() if k in ('PCC', 'SSIM', 'CMD', 'MoransI', 'SPCC')},
        'activation_type': act_type, 'time_seconds': t_elapsed,
    }
    with open(os.path.join(variant_dir, 'metrics.json'), 'w') as f:
        json.dump(save_res, f, indent=2)
    skin_all_results[variant_name] = save_res

    del trainer, panelB1, panelA2
    torch.cuda.empty_cache()
    gc.collect()

# Summary table
print("\n\n" + "=" * 90)
print("SKIN MELANOMA ACTIVATION ABLATION SUMMARY")
print("=" * 90)
header = f"{'Activation':<22} {'PCC_s1':>8} {'PCC_s2':>8} {'SSIM_s1':>8} {'SSIM_s2':>8} {'CMD_s1':>8} {'CMD_s2':>8}"
print(header)
print("-" * len(header))
for name, res in skin_all_results.items():
    s1, s2 = res['slice1'], res['slice2']
    print(f"{name:<22} {s1['PCC']:>8.4f} {s2['PCC']:>8.4f} "
          f"{s1['SSIM']:>8.4f} {s2['SSIM']:>8.4f} {s1['CMD']:>8.4f} {s2['CMD']:>8.4f}")

with open(os.path.join(output_dir_skin, 'ablation_summary.json'), 'w') as f:
    json.dump(skin_all_results, f, indent=2)

print(f"\n[SAVED] Skin results to: {output_dir_skin}")
print("=" * 70)
print("ALL DATASETS COMPLETE!")
print("=" * 70)
