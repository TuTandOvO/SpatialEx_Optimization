# Task1 Check: Run Baseline SpatialEx on HPC

## Goal

Task 1 is to run the original `SpatialEx` baseline end to end on the HPC path before touching the composite-loss improvements.

This folder stores:

- a concrete runbook for the baseline pipeline
- raw-data and processed-data validation scripts
- a source-analysis script for the original `SpatialEx` codebase

## Current Reality

Your current HPC raw data tree is organized by modality/domain:

- `/gpfsdata/home/renyixiang/YuanLab/data/MG_Xenium`
- `/gpfsdata/home/renyixiang/YuanLab/data/Skin_Xenium`
- `/gpfsdata/home/renyixiang/YuanLab/data/MG_Xenium/Visum`

But the current project scripts in `.claude/worktrees/youthful-cori/` assume a flatter layout and already preprocessed outputs under:

- `/gpfsdata/home/renyixiang/YuanLab/data/processed/<sample>/adata_final.h5ad`
- `/gpfsdata/home/renyixiang/YuanLab/data/processed/<sample>/spatial_graph.npz`

So Task 1 is not "train immediately". The correct order is:

1. verify the HPC environment and raw-input layout
2. decide which two Xenium slices form the baseline pair
3. generate preprocessing outputs
4. verify `adata_final.h5ad` and `spatial_graph.npz`
5. run baseline `SpatialEx`

## Recommended Task 1 Steps

### Step 1. Verify raw HPC layout

Run:

```bash
python Task1_Check/scripts/check_hpc_raw_data.py \
  --data-root /gpfsdata/home/renyixiang/YuanLab/data
```

This checks:

- which raw sample folders exist
- whether each folder contains the expected `outs.zip`
- whether H&E images exist
- whether alignment CSV exists or is missing
- whether there are folder-name / file-name mismatches worth cleaning up first

### Step 2. Confirm the baseline pair

Before preprocessing, decide which two Xenium slices will be the baseline dataset.

For the breast-cancer path, the likely candidates are the two biologically matched slices corresponding to:

- one `Rep1`
- one `Rep2`

Do not assume the current folder names are already correct. Your raw tree currently mixes names like:

- `Sample1_Rep2`
- `Sample2`
- `Sample2_Rep1`

That should be resolved first, because later preprocessing and dataset registration will depend on stable sample identifiers.

### Step 3. Inspect the original SpatialEx source path

Run:

```bash
python Task1_Check/scripts/analyze_spatialex_source.py \
  --spatialex-root /Users/renyixiang/Desktop/SpatialEx
```

This confirms the baseline trainer expectations:

- `SpatialEx` takes two `AnnData` slices and two graphs
- each `AnnData` must contain `obsm["he"]`
- training builds dataloaders from the provided graphs
- baseline inference uses `auto_inference()`

### Step 4. Produce preprocessing outputs

The training scripts need, per sample:

- `adata_preprocessed.h5ad`
- `he_patches.pt`
- `adata_final.h5ad`
- `spatial_graph.npz`

The intended two-stage preprocessing is:

1. local or CPU stage:
   - raw Xenium zip -> expression matrix
   - H&E image load and registration
   - patch tiling
2. HPC GPU stage:
   - image encoder extracts `obsm["he"]`
   - graph build
   - final `.h5ad` save

Task1_Check now contains its own preprocessing scripts:

- [preprocess_local.py](/Users/renyixiang/Desktop/Yuan_miniproj/Task1_Check/scripts/preprocess_local.py)
- [preprocess_hpc.py](/Users/renyixiang/Desktop/Yuan_miniproj/Task1_Check/scripts/preprocess_hpc.py)

These support both:

- nested MG samples, e.g. `MG_Xenium/Sample1_Rep1/`
- flat skin raw files under `Skin_Xenium/`

For skin, preprocessing one sample is valid. Training baseline `SpatialEx` is still blocked until you have a second compatible skin slice.

Example commands:

```bash
python Task1_Check/scripts/preprocess_local.py \
  --data-root /gpfsdata/home/renyixiang/YuanLab/data \
  --output-root /gpfsdata/home/renyixiang/YuanLab/data/processed \
  --samples hSkin_Melanoma_Base
```

```bash
python Task1_Check/scripts/preprocess_hpc.py \
  --data-root /gpfsdata/home/renyixiang/YuanLab/data \
  --processed-dir /gpfsdata/home/renyixiang/YuanLab/data/processed \
  --samples hSkin_Melanoma_Base \
  --device cuda
```

### Step 5. Verify processed outputs

After preprocessing, run:

```bash
python Task1_Check/scripts/check_processed_outputs.py \
  --processed-root /gpfsdata/home/renyixiang/YuanLab/data/processed \
  --samples <sample_a> <sample_b>
```

This checks both preprocessing stages:

- Stage 1 outputs:
  - `adata_preprocessed.h5ad`
  - `he_patches.pt`
- Stage 2 outputs:
  - `adata_final.h5ad`
  - `spatial_graph.npz`

### Step 6. Run baseline SpatialEx

Only after Step 5 passes, the baseline training command makes sense.

The baseline script in the Claude worktree is:

```bash
python .claude/worktrees/youthful-cori/scripts/run_baseline.py \
  --config .claude/worktrees/youthful-cori/configs/baseline.yaml \
  --dataset Xenium_Human_Breast_Cancer
```

But this will only work after:

- the dataset registry points to the correct processed sample names
- the required `.h5ad` and graph files exist
- `SpatialEx` is importable in the active environment

## What "Baseline SpatialEx Ready" Means

You are ready to move beyond Task 1 only if all of the following are true:

- `SpatialEx` imports successfully
- two target Xenium slices are selected and named consistently
- both samples have `adata_final.h5ad`
- both samples have `spatial_graph.npz`
- a baseline training run finishes
- predictions and metrics are saved under a results directory

## Scripts in This Folder

- `scripts/check_hpc_raw_data.py`
  - validates the current raw-data tree
- `scripts/check_processed_outputs.py`
  - validates preprocessing outputs before training
- `scripts/analyze_spatialex_source.py`
  - summarizes the original `SpatialEx` baseline entry points and assumptions

## Practical Notes

- `Skin_Xenium` can be preprocessed as a single sample even if it cannot yet be used for two-slice baseline training.
- `MG_Xenium/Visum` is a separate modality path and should not be mixed into the first Xenium baseline run.
- Task 1 should focus on one clean Xenium pair first.
