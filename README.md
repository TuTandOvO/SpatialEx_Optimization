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
  <em>Systematic optimization and analysis of <a href="https://github.com/KEAML-JLU/SpatialEx">SpatialEx</a> — a spatial transcriptomics gene expression prediction framework based on H&E image morphology.</em>
</p>

---

## Overview

SpatialEx predicts spatial gene expression from H&E histology images:

```
H&E Image --> ResNet50 (frozen, 2048-dim) --> HGNN (k=7 hypergraph) --> DGI (contrastive) --> MLP --> Gene Expression
```

This project explores **8 optimization directions** across **4 phases**, evaluated on two 10x Xenium datasets:

| Dataset | Platform | Genes | Slices |
|---------|----------|-------|--------|
| MG (Mouse Brain) | 10x Xenium | 313 | 2 |
| Skin (Human Skin) | 10x Xenium | 282 | 2 |

Genes are stratified into three tiers by baseline per-gene PCC:
- **MI (Morphology-Informative)**: top 1/3 — model predicts well
- **MOD (Moderate)**: middle 1/3
- **MU (Morphology-Uninformative)**: bottom 1/3 — model outputs near-constant predictions

## Project Structure

```
SpatialEx_Optimization/
├── Task1_Baseline/            # Baseline reproduction and gene stratification
├── Task2_LossAblation/        # Loss function ablation (5 variants)
├── Task3_ActivationFunc/      # GELU activation replacement
├── Task4_FeatureSelect/       # SVG selection + attention gate
├── Task5_PPI/                 # PPI network external priors
├── Task6_GenePropagation/     # MI->MU linear diffusion
├── Task7_GeneGCN/             # 2-layer GCN residual propagation
├── Task8_Uncertainty/         # CCV + conformal prediction
└── README.md
```

---

## Results Summary

### Phase I: Architecture Tuning (Tasks 2-4)

> All architecture-level modifications yield negligible improvement, indicating the baseline is already well-tuned.

| Task | Method | ΔPCC (MG) | ΔPCC (Skin) |
|------|--------|:---------:|:-----------:|
| Task 2 | Loss ablation (cosine, weighted MSE, spatial, soft rank) | ≈ 0 | ≈ 0 |
| Task 3 | GELU activation | ≈ 0 | ≈ 0 |
| Task 4 | SVG feature selection + attention gate | ≈ 0 | ≈ 0 |

### Phase II: External Priors (Task 5)

> Incorporating PPI network from [HumanBase](https://humanbase.net/) as external biological prior provides the only meaningful improvement.

| Task | Method | ΔPCC (MG) | ΔPCC (Skin) |
|------|--------|:---------:|:-----------:|
| Task 5 | PPI-augmented hypergraph | **+0.06** (~22%) | **+0.05** (~32%) |

```
MG:   Baseline PCC 0.257/0.268  -->  PPI-only PCC 0.315/0.322
Skin: Baseline PCC 0.163/0.156  -->  PPI-only PCC 0.223/0.202
```

**Inspiration**: Traditional spatial transcriptomics models construct graphs based solely on spatial proximity. We hypothesized that gene-gene interaction priors could provide complementary biological information. Using [HumanBase](https://humanbase.net/) (Flatiron Institute), we constructed lightweight, tissue-specific PPI networks via their API, then integrated PPI edges into SpatialEx's hypergraph as an additional information source. This allows the model to leverage known protein-protein interactions when predicting gene expression — particularly beneficial for genes whose expression is co-regulated through shared pathways rather than spatial co-localization.

Note: WGCNA co-expression priors were also tested but proved ineffective or harmful.

### Phase III: MU Gene Correction (Tasks 6-7)

> Post-hoc attempts to improve MU gene predictions yield negligible gains, even with correct implementations.

| Task | Method | ΔPCC (MG) | ΔPCC (Skin) |
|------|--------|:---------:|:-----------:|
| Task 6 | MI->MU linear diffusion | +0.001 | +0.001 |
| Task 7 | 2-layer GCN residual propagation | +0.003 | +0.003 |

**Inspiration**: Since MI genes are well-predicted, we attempted to propagate their predictions to MU genes through gene correlation graphs — similar in spirit to [SPRITE](https://academic.oup.com/bioinformatics/article/40/Supplement_1/i482/7700862) (Bioinformatics, 2024). Both linear diffusion and learnable GCN approaches were tested.

### Phase IV: Understanding the Limit (Task 8)

> Uncertainty quantification reveals the fundamental reason behind MU gene prediction failure.

| Metric | MI Genes | MU Genes |
|--------|:--------:|:--------:|
| CCV (Coefficient of Variation) | < 0.3 | **≈ 1.0** |
| Conformal coverage (within-slice) | ~67% | ~67% |

**Key finding**: CCV ≈ 1.0 for MU genes means the model outputs spatially constant predictions — essentially predicting the mean everywhere. This is not a model deficiency but a **fundamental biological limitation**: MU genes' expression patterns are not encoded in tissue morphology visible in H&E images. Uncertainty quantification approach adapted from [TISSUE](https://www.nature.com/articles/s41592-024-02084-1) (Nature Methods, 2024).

---

## Task Details

### Task 1: Baseline Reproduction
- **Script**: `baseline.ipynb`
- **Purpose**: Reproduce SpatialEx baseline results; stratify genes into MI/MOD/MU tiers

### Task 2: Loss Function Ablation
- **Script**: `run_ablation_pipeline.py`
- **Variants**: Baseline MSE, Cosine similarity, Weighted MSE, Spatial smoothness, Soft rank
- **Results**: `SpatialEx_ablation_MG/`, `SpatialEx_ablation_skin/`

### Task 3: Activation Function
- **Script**: `run_gelu_activation_pipeline.py`
- **Method**: Replace PReLU with GELU in HGNN layers
- **Results**: `SpatialEx_results_gelu_MG/`, `SpatialEx_results_gelu_skin/`

### Task 4: Feature Selection
- **Script**: `run_feature_select_pipeline.py`
- **Method**: Spatially Variable Gene (SVG) selection + attention gate on H&E features
- **Results**: `SpatialEx_results_feature_select_MG/`, `SpatialEx_results_feature_select_skin/`

### Task 5: PPI Network Priors
- **Scripts**: `run_wgcna_ppi_pipeline.py`, `download_ppi_network.py`, `analyze_stratified_improvement.py`
- **Method**: Augment spatial hypergraph with tissue-specific PPI edges from HumanBase
- **Results**: `SpatialEx_results_wgcna_ppi_MG/`, `SpatialEx_results_wgcna_ppi_skin/`
- **Data**: `ppi_data_MG/`, `ppi_data_skin/`

### Task 6: Gene Propagation (Linear Diffusion)
- **Script**: `run_propagation.py`
- **Method**: Propagate MI gene predictions to MU genes via linear diffusion on gene correlation graph
- **Results**: `results/`

### Task 7: Gene GCN (Graph Convolution)
- **Script**: `run_gene_gnn.py`
- **Method**: 2-layer GCN with residual connections for MI->MU gene prediction refinement
- **Results**: `results/`

### Task 8: Uncertainty Quantification
- **Script**: `run_uncertainty.py`
- **Method**: CCV analysis + split conformal prediction for calibrated confidence intervals
- **Results**: `results/`

---

## Key Conclusions

1. **Architecture is saturated**: Loss functions, activations, and feature selection do not improve SpatialEx — the baseline is already near-optimal for the H&E -> gene expression task.

2. **External biological priors help**: PPI network integration is the only modification that meaningfully improves prediction (ΔPCC +0.05~0.06), suggesting gene-gene interaction knowledge complements image morphology.

3. **MU genes are fundamentally limited**: Post-hoc correction cannot recover MU gene predictions because the model outputs spatially constant values (CCV ≈ 1.0) — there is no spatial signal to correct.

4. **The limitation is biological, not computational**: MU genes' expression is not encoded in H&E-visible tissue morphology. Uncertainty quantification provides calibrated confidence intervals as a practical alternative.

---

## References

| # | Paper | Relevance |
|---|-------|-----------|
| 1 | Yuan, Z. et al. "High-Parameter Spatial Multi-Omics through Histology-Anchored Integration." *Nature Methods* (2025). [[paper]](https://www.nature.com/articles/s41592-025-02926-6) [[code]](https://github.com/KEAML-JLU/SpatialEx) | Base framework |
| 2 | Greene, C.S. et al. "Understanding multicellular function and disease with human tissue-specific networks." *Nature Genetics* (2015). [[HumanBase]](https://humanbase.net/) | Tissue-specific PPI networks (Task 5) |
| 3 | Langfelder, P. & Horvath, S. "WGCNA: an R package for weighted correlation network analysis." *BMC Bioinformatics* (2008). | Co-expression network (Task 5) |
| 4 | Velickovic, P. et al. "Deep Graph Infomax." *ICLR* (2019). | Contrastive learning module |
| 5 | Gao, Y. et al. "HGNN+: General Hypergraph Neural Networks." *IEEE TPAMI* (2022). | Hypergraph convolution backbone |
| 6 | Li, Y. et al. "TISSUE: uncertainty-calibrated prediction of single-cell spatial transcriptomics improves downstream analyses." *Nature Methods* (2024). [[paper]](https://www.nature.com/articles/s41592-024-02084-1) | Uncertainty quantification (Task 8) |
| 7 | Liu, Y. et al. "SPRITE: improving spatial gene expression imputation with gene and cell networks." *Bioinformatics* (2024). [[paper]](https://academic.oup.com/bioinformatics/article/40/Supplement_1/i482/7700862) | Gene propagation inspiration (Tasks 6-7) |
| 8 | Hendrycks, D. & Gimpel, K. "Gaussian Error Linear Units (GELUs)." *arXiv* (2016). | Activation function (Task 3) |

---

## Environment

- Python 3.10
- PyTorch
- SpatialEx (custom install)
- 10x Xenium datasets (MG, Skin)
