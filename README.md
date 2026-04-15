# SpatialEx Optimization

Systematic optimization and analysis of [SpatialEx](https://github.com/TuTandOvO/SpatialEx), a spatial transcriptomics gene expression prediction framework based on H&E image morphology.

## Overview

SpatialEx predicts spatial gene expression from H&E histology images using a pipeline of:
**H&E Image -> ResNet50 (frozen, 2048-dim) -> HGNN (k=7 hypergraph convolution) -> DGI (contrastive learning) -> MLP Predictor -> Gene Expression**

This project explores 8 optimization directions across 4 phases, evaluated on two Xenium datasets:
- **MG (Mouse Brain)**: 313 genes
- **Skin (Human Skin)**: 313 genes

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

## Datasets

| Dataset | Platform | Genes | Slices |
|---------|----------|-------|--------|
| MG (Mouse Brain) | 10x Xenium | 313 | 2 |
| Skin (Human Skin) | 10x Xenium | 313 | 2 |

Genes are stratified into three tiers by baseline per-gene PCC:
- **MI (Morphology-Informative)**: top 1/3 PCC — model predicts well
- **MOD (Moderate)**: middle 1/3
- **MU (Morphology-Uninformative)**: bottom 1/3 — model outputs near-constant predictions

## Results Summary

### Phase I: Architecture Tuning (Tasks 2-4)

All architecture-level modifications yield negligible improvement (ΔPCC ≈ 0), indicating the baseline is already well-tuned for its architecture class.

| Task | Method | ΔPCC (MG) | ΔPCC (Skin) |
|------|--------|-----------|-------------|
| Task 2 | Loss ablation (cosine, weighted MSE, spatial, soft rank) | ≈ 0 | ≈ 0 |
| Task 3 | GELU activation | ≈ 0 | ≈ 0 |
| Task 4 | SVG feature selection + attention gate | ≈ 0 | ≈ 0 |

### Phase II: External Priors (Task 5)

Incorporating PPI (Protein-Protein Interaction) network as external biological prior provides the only meaningful improvement.

| Task | Method | ΔPCC (MG) | ΔPCC (Skin) |
|------|--------|-----------|-------------|
| Task 5 | PPI-augmented hypergraph | **+0.06** (~22% relative) | **+0.05** (~32% relative) |

Baseline (MG): PCC 0.257/0.268 -> PPI-only: PCC 0.315/0.322
Baseline (Skin): PCC 0.163/0.156 -> PPI-only: PCC 0.223/0.202

Note: WGCNA co-expression priors were also tested but proved ineffective or harmful.

### Phase III: MU Gene Correction (Tasks 6-7)

Post-hoc attempts to improve MU gene predictions yield negligible gains, even with correct implementations.

| Task | Method | ΔPCC (MG) | ΔPCC (Skin) |
|------|--------|-----------|-------------|
| Task 6 | MI->MU linear diffusion | +0.001 | +0.001 |
| Task 7 | 2-layer GCN residual propagation | +0.003 | +0.003 |

### Phase IV: Understanding the Limit (Task 8)

Uncertainty quantification reveals the fundamental reason behind MU gene prediction failure.

| Metric | MI Genes | MU Genes |
|--------|----------|----------|
| CCV (Coefficient of Variation of predictions) | < 0.3 | **≈ 1.0** |
| Conformal coverage (within-slice) | ~67% | ~67% |

**Key finding**: CCV ≈ 1.0 for MU genes means the model outputs spatially constant predictions — essentially predicting the mean everywhere. This is not a model deficiency but a fundamental limitation: MU genes' expression patterns are not encoded in tissue morphology visible in H&E images.

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
- **Scripts**: `run_wgcna_ppi_pipeline.py`, `run_wgcna_ppi_pipeline_v2.py`, `download_ppi_network.py`, `analyze_stratified_improvement.py`
- **Method**: Augment spatial hypergraph with PPI (HumanBase) gene-gene interaction edges
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

## Key Conclusions

1. **Architecture is saturated**: Loss functions, activations, and feature selection do not improve SpatialEx — the baseline architecture is already near-optimal for the H&E -> gene expression task.

2. **External biological priors help**: PPI network integration (Task 5) is the only modification that meaningfully improves prediction (ΔPCC +0.05~0.06), suggesting that gene-gene interaction knowledge provides information complementary to image morphology.

3. **MU genes are fundamentally limited**: Post-hoc correction (Tasks 6-7) cannot recover MU gene predictions because the model outputs spatially constant values (CCV ≈ 1.0) — there is no spatial signal to correct.

4. **The limitation is biological, not computational**: MU genes' expression is not encoded in H&E-visible tissue morphology. Uncertainty quantification (Task 8) provides calibrated confidence intervals as a practical alternative to prediction improvement.

## Environment

- Python 3.10
- PyTorch
- SpatialEx (custom install)
- 10x Xenium datasets (MG, Skin)
