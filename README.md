<p align="center">
  <img src="https://img.shields.io/badge/SpatialEx-Optimization-blue?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJ3aGl0ZSI+PHBhdGggZD0iTTEyIDJDNi40OCAyIDIgNi40OCAyIDEyczQuNDggMTAgMTAgMTAgMTAtNC40OCAxMC0xMFMxNy41MiAyIDEyIDJ6bTAgMThjLTQuNDIgMC04LTMuNTgtOC04czMuNTgtOCA4LTggOCAzLjU4IDggOC0zLjU4IDgtOCA4eiIvPjxjaXJjbGUgY3g9IjgiIGN5PSIxMCIgcj0iMS41Ii8+PGNpcmNsZSBjeD0iMTIiIGN5PSI4IiByPSIxLjUiLz48Y2lyY2xlIGN4PSIxNiIgY3k9IjEwIiByPSIxLjUiLz48Y2lyY2xlIGN4PSIxMCIgY3k9IjE0IiByPSIxLjUiLz48Y2lyY2xlIGN4PSIxNCIgY3k9IjE0IiByPSIxLjUiLz48bGluZSB4MT0iOCIgeTE9IjEwIiB4Mj0iMTIiIHkyPSI4IiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjAuOCIvPjxsaW5lIHgxPSIxMiIgeTE9IjgiIHgyPSIxNiIgeTI9IjEwIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjAuOCIvPjxsaW5lIHgxPSI4IiB5MT0iMTAiIHgyPSIxMCIgeTI9IjE0IiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjAuOCIvPjxsaW5lIHgxPSIxNiIgeTE9IjEwIiB4Mj0iMTQiIHkyPSIxNCIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIwLjgiLz48bGluZSB4MT0iMTAiIHkxPSIxNCIgeDI9IjE0IiB5Mj0iMTQiIHN0cm9rZT0id2hpdGUiIHN0cm9rZS13aWR0aD0iMC44Ii8+PC9zdmc+" alt="SpatialEx Optimization"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Platform-10x_Xenium-green?style=flat-square" alt="Platform"/>
  <img src="https://img.shields.io/badge/Framework-PyTorch-red?style=flat-square&logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square" alt="License"/>
</p>

<p align="center">
  <em>对 <a href="https://github.com/KEAML-JLU/SpatialEx">SpatialEx</a> 的系统性优化与分析 — 一个基于 H&E 图像形态学预测空间转录组的框架。</em>
</p>

---

## 项目概览 (Overview)

SpatialEx 从 H&E 组织病理图像预测空间基因表达，主流程为：

```
H&E Image --> ResNet50 (frozen, 2048-dim) --> HGNN (k=7 hypergraph) --> DGI (contrastive) --> MLP --> Gene Expression
```

本项目沿 **4 个阶段**探索了 **8 个优化方向**，在两个 10x Xenium 数据集上评估：

| Dataset | Platform | Genes | Slices |
|---------|----------|-------|--------|
| MG (Breast Cancer, Mammary Gland) | 10x Xenium | 313 | 2 |
| Skin (Human Skin Melanoma) | 10x Xenium | 282 | 2 |

按 baseline per-gene PCC 将基因分成三组，后续所有分析都基于这个分层：

- **MI (Morphology-Informative)**: 上 1/3，模型预测得最好的一批基因
- **MOD (Moderate)**: 中间 1/3
- **MU (Morphology-Uninformative)**: 下 1/3，模型基本输出空间常数

## 项目结构 (Project Structure)

```
SpatialEx_Optimization/
├── Task1_Baseline/            # 复现 baseline + 基因分层 (MI/MOD/MU)
├── Task2_LossAblation/        # Loss 消融 (5 个变体)
├── Task3_ActivationFunc/      # GELU / SiLU / Mish 激活函数替换
├── Task4_FeatureSelect/       # SVG 选择 + Attention Gate
├── Task5_PPI/                 # PPI 网络作为外部先验融入超图
├── Task6_GenePropagation/     # MI→MU 线性扩散 (gene graph)
├── Task7_GeneGCN/             # 2 层 GCN 残差校正
├── Task8_Uncertainty/         # CCV + Conformal Prediction
└── README.md
```

---

## 结果汇总 (Results Summary)

### 阶段一：架构层调参 (Tasks 2-4)

> 所有架构层修改要么不起作用，要么直接伤害 baseline。Loss 消融和 Attention Gate 的差异都在噪声范围内；SVG 子集训练则是负优化。

| Task | Method | ΔPCC (MG) | ΔPCC (Skin) |
|------|--------|:---------:|:-----------:|
| Task 2 | Loss 消融 (cosine, weighted MSE, spatial smooth, soft rank) | ≈ 0 | ≈ 0 |
| Task 3 | GELU 激活替换 LeakyReLU | ≈ 0 | ≈ 0 |
| Task 4a | SVG 特征选择（取 Moran's I 最高的 top-K 基因做 loss） | **−0.06** | **−0.03** |
| Task 4b | H&E 特征上的 Attention Gate | ≈ 0 | ≈ 0 |
| Task 4c | SVG + Attn 组合 | **−0.07** | **−0.03** |

**解读**：SVG 在 MG (0.254 → 0.190) 和 Skin (0.166 → 0.135) 上都退化，原因是把 loss 限定在少数高方差基因后，模型为了 fit 这些基因损失了对其他基因的拟合能力，全局平均反而更差。Attention Gate 不害不益，说明 ResNet50/UNI 提取的 H&E 特征本身已经足够 task-specific，不需要再做维度筛选。Loss 消融和激活函数替换的结论是：**SpatialEx 的 baseline 在架构层已经基本调到位**。

### 阶段二：外部生物先验 (Task 5)

> 把 [HumanBase](https://humanbase.net/) 的 PPI 网络融入超图，是整个项目里**唯一**带来显著提升的方向。

| Task | Method | ΔPCC (MG) | ΔPCC (Skin) |
|------|--------|:---------:|:-----------:|
| Task 5 | PPI-augmented hypergraph | **+0.06** (~22%) | **+0.05** (~32%) |

```
MG:   Baseline PCC 0.257/0.268  -->  PPI 0.315/0.322
Skin: Baseline PCC 0.163/0.156  -->  PPI 0.223/0.202
```

**灵感来源**：原版 SpatialEx 的超图只基于空间 k-NN，隐含假设"空间近邻 = 功能近邻"，在异质性组织（乳腺癌、黑色素瘤）里这个假设不成立。我们的思路是：用 [HumanBase](https://humanbase.net/) (Flatiron Institute) 提供的组织特异性基因功能网络作为额外的边来源，让模型在预测时能用上"哪些基因功能上相关"这条独立信息。具体做法是通过 HumanBase API 拉取 mammary-gland / skin 的 G×G 功能矩阵，然后把 PPI 加权的 k-NN 作为超图的第三层（与空间 k-NN 并列）。

**解读**：这个改动提示 SpatialEx 的瓶颈不在网络架构而在**信息源**。共享通路 / 共调控的基因即使在空间上不相邻，也存在统计上的协同表达，这一类信号是 H&E 图像本身无法提供的，必须从外部知识图谱注入。

> **注**：WGCNA 共表达模块也尝试过，但单独使用时反而伤 PCC（MG slice1 -0.044），叠加在 PPI 之上也几乎没有增量价值（≤ 0.001），所以最终方案只用 PPI。

### 阶段三：MU 基因后处理 (Tasks 6-7)

> 试图把 MI 基因的预测信号通过 gene-gene graph 扩散到 MU 基因，提升幅度都在 per-slice 噪声范围内。

| Task | Method | ΔPCC (MG) | ΔPCC (Skin) |
|------|--------|:---------:|:-----------:|
| Task 6 | MI→MU 线性扩散 (PPI + Spearman, α-混合) | +0.001 | +0.002 |
| Task 7 | 2 层 GCN 残差校正 | +0.002 | +0.002 |

**灵感来源**：受 [SPRITE](https://academic.oup.com/bioinformatics/article/40/Supplement_1/i482/7700862) (Bioinformatics 2024) 启发——既然 MI 基因预测得很好，能不能通过基因-基因关联图把这些"靠谱"的信号扩散给 MU 基因？我们试了两种实现：线性扩散（无参数）和 2 层 GCN（有参数、学非线性 mapping），都在 Task 5 输出的基础上做 post-hoc 校正。

**解读**：两种方法的提升都 ≤ 0.002 PCC，落在不同 random seed 之间的波动范围内，**不能算是真正的改进**。这与阶段四的结论一致：MU 基因之所以预测不好，不是后处理能解决的。

> **重要 caveat**：Task 6 的最佳超参 (α, ppi_weight, coexpr_weight, threshold) 是在 135 个组合上 grid search、并用**测试切片自身的 overall PCC** 选出的。MG 选到 `ppi_weight=0.7`、Skin 却选到 `ppi_weight=0.3`，两者完全相反——这是典型的 grid-search-on-test 过拟合信号。报告里的数字应当作上界看，而不是稳定可迁移的提升。

### 阶段四：理解能力的边界 (Task 8)

> 不再试图改进 MU 基因预测，而是**量化哪些预测可信**。

| Metric | MI Genes | MU Genes |
|--------|:--------:|:--------:|
| CCV (Coefficient of Variation) | < 0.3 | **≈ 1.0** |
| Conformal coverage (within-slice, α=0.33) | ~67% | ~67% |
| Conformal coverage (cross-slice, Skin) | ~59% | ~57% |

**核心发现**：MU 基因的 CCV ≈ 1.0 意味着模型对这类基因输出的几乎是空间常数（每个细胞预测值几乎一样，相当于"猜均值"）。这不是模型本身有缺陷，而是 H&E 形态学**根本不编码**这些基因的空间模式——这是数据层面的、不可逾越的生物学限制。Task 8 的方法借鉴了 [TISSUE](https://www.nature.com/articles/s41592-024-02084-1) (Nature Methods 2024)，但不依赖 ground truth，用 H&E 特征余弦相似度做邻居加权，因此可以在推理阶段直接使用。

> **重要 caveat**：within-slice 的 conformal coverage 精确命中名义 67%（因为校准集与测试集来自同一切片）；但当校准集换成另一切片后，coverage 跨基因层全部下降约 8 个百分点——也就是说 conformal 的覆盖率保证**不能跨切片自动迁移**，部署时需要每张切片单独重校准。

---

## 各任务说明 (Task Details)

### Task 1: Baseline 复现
- **Script**: `baseline.ipynb`
- **目的**: 复现 SpatialEx baseline 结果；按 per-gene PCC 把基因分成 MI/MOD/MU 三层

### Task 2: Loss 消融
- **Script**: `run_ablation_pipeline.py`
- **变体**: Baseline MSE, Cosine similarity, Weighted MSE, Spatial smoothness, Soft rank
- **结果**: `SpatialEx_ablation_MG/`, `SpatialEx_ablation_skin/`

### Task 3: 激活函数替换
- **Script**: `run_gelu_activation_pipeline.py`
- **方法**: 把 predictor head 的 LeakyReLU(0.1) 换成 GELU；HGNN 内部的 `prelu` 通过 monkey-patch `create_activation` 换成 GELU/SiLU/Mish
- **结果**: `SpatialEx_results_gelu_MG/`, `SpatialEx_results_gelu_skin/`

### Task 4: 特征选择
- **Script**: `run_feature_select_pipeline.py`
- **方法**: 空间变异基因 (SVG, top-Moran's I) 选择 + H&E 特征上的 sigmoid attention gate
- **结果**: `SpatialEx_results_feature_select_MG/`, `SpatialEx_results_feature_select_skin/`

### Task 5: PPI 网络先验
- **Scripts**: `run_wgcna_ppi_pipeline.py`, `download_ppi_network.py`, `analyze_stratified_improvement.py`
- **方法**: 把 HumanBase 的组织特异性 PPI 边加入空间超图
- **结果**: `SpatialEx_results_wgcna_ppi_MG/`, `SpatialEx_results_wgcna_ppi_skin/`
- **数据**: `ppi_data_MG/`, `ppi_data_skin/`

### Task 6: 基因传播 (线性扩散)
- **Script**: `run_propagation.py`
- **方法**: 在基因关联图 (PPI + Spearman 共表达) 上把 MI 基因预测线性扩散到 MU 基因
- **结果**: `results/`

### Task 7: Gene GCN
- **Script**: `run_gene_gnn.py`
- **方法**: 2 层 GCN 在基因图上学一个残差 delta，对 MU/MOD 基因做校正；MI 基因 delta mask 为 0
- **结果**: `results/`

### Task 8: 不确定性量化
- **Script**: `run_uncertainty.py`
- **方法**: CCV 分析 + split conformal prediction 给出每个 (cell, gene) 的置信区间
- **结果**: `results/`

---

## 关键结论 (Key Conclusions)

1. **架构层调整不起作用**。Loss 重加权 (Task 2)、激活函数替换 (Task 3)、Attention Gate (Task 4b) 的差异都在噪声范围内。SVG 子集训练 (Task 4a) 反而显著伤害模型 (MG -0.06, Skin -0.03 PCC)——所以"特征选择"这种听起来正面的操作不能默认有效，需要实测。

2. **唯一真正起作用的是外部生物先验**。把组织特异性 PPI (HumanBase) 加进超图后，整体 PCC 提升 +0.05~0.06（MG ≈22%，Skin ≈33%）。WGCNA 共表达单独用反而伤指标，叠加在 PPI 之上也几乎无增量价值，最终方案只保留 PPI。

3. **MU 基因不可能通过后处理拯救**。线性扩散 (Task 6) 和 2 层 GCN (Task 7) 给出的 ΔPCC 都 ≤ 0.002，且 Task 6 的"最佳"超参是在测试切片上 grid search 选出的——属于 grid-search-on-test 过拟合，不能算稳定提升。

4. **瓶颈是生物学的，不是计算的**。Task 8 的 CCV ≈ 1.0 说明模型对 MU 基因输出近似空间常数，这是输入信号本身缺失的必然结果——H&E 形态学就是不编码这些基因的空间模式。Within-slice 的 conformal coverage 完美命中名义水平；但 cross-slice 下降 ~8 个点，部署时需要逐切片重校准。

---

## 未来方向 (Future Directions)

**1. 用 ZINB head 适配稀疏 count 数据 (e.g. Stereo-seq)。** SpatialEx 当前的 MSE 回归隐含连续 Gaussian 假设，在 Xenium 这种小靶向 panel 上没问题，但对 **Stereo-seq** 这种全转录组、零膨胀、过离散的 count 数据就不合适了。把 MLP head 换成 ZINB 输出（每个基因输出 μ, θ, π 三个参数）能原生处理 dropout 和 mean-variance relationship。Hist2ST (Briefings in Bioinformatics 2022) 已经在 ST 数据上验证了 GNN+ZINB 的可行性；至于是否需要 zero-inflation 还是普通 NB 就够（参见 Sarkar & Stephens, *Genome Biology* 2022），值得逐平台实测。

**2. PPI 与空间信号的因果解耦 (Celcomen 风格)。** Task 5 只是把 PPI 边加进超图，模型没法分辨某个基因表达是被**邻居细胞**驱动 (inter-cellular, CCE) 还是被**细胞自身状态**驱动 (intra-cellular, SCE)。[Celcomen](https://arxiv.org/abs/2409.05804) (ICLR 2025) 把基因调控拆成这两条因果通道。把 PPI 当作独立的因果通道而不是和空间边混在一起，应该能给出更可解释、更有效的整合。

**3. 用 Flow Matching / Diffusion 做生成式预测。** 同样的 H&E patch 可以对应不同的表达状态——这是个一对多 mapping，点回归只能学到条件期望，无法捕捉这种 stochasticity，这也部分解释了 MU 基因为什么 collapse 到空间常数。[STFlow](https://openreview.net/forum?id=Ossg1IbHDT) (ICML 2025 Spotlight) 用 flow matching 对整张切片建模联合分布；[Stem](https://openreview.net/forum?id=FtjLUHyZAO) (ICLR 2025) 用 conditional DDPM 做表达分布生成。这两种范式直接攻击"variance collapse"问题，比换 head 工程量大但更有原理性。

**4. 用 Mixture-of-Experts decoder 处理 MI/MOD/MU 异质性。** 项目阶段一到四的所有结果都指向同一个事实：MI 和 MU 基因需要完全不同的归纳偏置，而 SpatialEx 用的是单一共享 predictor。借鉴 [scGPT-spatial](https://www.biorxiv.org/content/10.1101/2025.02.05.636714v1) (bioRxiv 2025)，可以让一个学到的 router 把每个基因路由到偏 morphology 的专家（处理 MI）或偏 context/PPI 的专家（处理 MU），不需要硬性的 tier 边界。

---

## 参考文献 (References)

| # | Paper | 与本项目的关系 |
|---|-------|-----------|
| 1 | Yuan, Z. et al. "High-Parameter Spatial Multi-Omics through Histology-Anchored Integration." *Nature Methods* (2025). [[paper]](https://www.nature.com/articles/s41592-025-02926-6) [[code]](https://github.com/KEAML-JLU/SpatialEx) | 基础框架 |
| 2 | Greene, C.S. et al. "Understanding multicellular function and disease with human tissue-specific networks." *Nature Genetics* (2015). [[HumanBase]](https://humanbase.net/) | 组织特异性 PPI 网络 (Task 5) |
| 3 | Langfelder, P. & Horvath, S. "WGCNA: an R package for weighted correlation network analysis." *BMC Bioinformatics* (2008). | 共表达网络 (Task 5) |
| 4 | Velickovic, P. et al. "Deep Graph Infomax." *ICLR* (2019). | 对比学习模块 |
| 5 | Gao, Y. et al. "HGNN+: General Hypergraph Neural Networks." *IEEE TPAMI* (2022). | 超图卷积 backbone |
| 6 | Li, Y. et al. "TISSUE: uncertainty-calibrated prediction of single-cell spatial transcriptomics improves downstream analyses." *Nature Methods* (2024). [[paper]](https://www.nature.com/articles/s41592-024-02084-1) | 不确定性量化 (Task 8) |
| 7 | Liu, Y. et al. "SPRITE: improving spatial gene expression imputation with gene and cell networks." *Bioinformatics* (2024). [[paper]](https://academic.oup.com/bioinformatics/article/40/Supplement_1/i482/7700862) | 基因传播灵感 (Tasks 6-7) |
| 8 | Hendrycks, D. & Gimpel, K. "Gaussian Error Linear Units (GELUs)." *arXiv* (2016). | 激活函数 (Task 3) |
| 9 | Megas, S. et al. "Celcomen: a generative model for disentangling intra- and inter-cellular gene regulation." *ICLR* (2025) / *Nature Communications* (2026). [[paper]](https://arxiv.org/abs/2409.05804) | 因果 PPI 建模 (Future Direction 2) |
| 10 | Zeng, Z. et al. "STFlow: scalable generation of spatial transcriptomics via whole-slide flow matching." *ICML* (2025, Spotlight). [[paper]](https://openreview.net/forum?id=Ossg1IbHDT) | 生成式预测 (Future Direction 3) |
| 11 | Bao, F. et al. "Stem: diffusion generative modeling for spatial gene expression inference." *ICLR* (2025). [[paper]](https://openreview.net/forum?id=FtjLUHyZAO) | Diffusion 预测 (Future Direction 3) |
| 12 | Cui, H. et al. "scGPT-spatial: continual pretraining for spatial transcriptomics." *bioRxiv* (2025). [[paper]](https://www.biorxiv.org/content/10.1101/2025.02.05.636714v1) | MoE 基因路由 (Future Direction 4) |
| 13 | Zeng, Y. et al. "Hist2ST: whole-slide image based spatial transcriptomics prediction." *Briefings in Bioinformatics* (2022). | GNN + ZINB 框架 (Future Direction 1) |

---

## 运行环境 (Environment)

- Python 3.10
- PyTorch
- SpatialEx (源码安装: https://github.com/KEAML-JLU/SpatialEx)
- 10x Xenium 数据集 (MG: Breast Cancer Rep1/Rep2; Skin: Human Skin Melanoma FFPE)
