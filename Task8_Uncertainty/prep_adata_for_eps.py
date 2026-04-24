#!/usr/bin/env python3
"""
Task 8 v2 — Stage 1: prep AnnData with H&E features for EPS.

Runs in the SpatialEx env. For each dataset (mg / skin), reads Xenium + H&E,
extracts ResNet50 patch features, and writes an .h5ad with:
  adata.X = log1p-normalized expression
  adata.obsm['he'] = H&E image features (2048-dim)
  adata.obsm['spatial'] = (x, y) in μm

These .h5ad files are consumed by run_eps_conformal.py (in the eps env).
"""
import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

import SpatialEx as se

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prep_eps")

# Hard-coded HPC paths (same as Task 5)
MG_ROOT1 = '/gpfsdata/home/renyixiang/YuanLab/data/MG_Xenium/Sample1_Rep1/Human_Breast_Cancer_Rep1/'
MG_ROOT2 = '/gpfsdata/home/renyixiang/YuanLab/data/MG_Xenium/Sample1_Rep2/Human_Breast_Cancer_Rep2/'
MG_HE1 = MG_ROOT1 + 'Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif'
MG_HE1_ALIGN = MG_ROOT1 + 'Xenium_FFPE_Human_Breast_Cancer_Rep1_he_imagealignment.csv'
MG_HE2 = MG_ROOT2 + 'Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image.ome.tif'
MG_HE2_ALIGN = MG_ROOT2 + 'Xenium_FFPE_Human_Breast_Cancer_Rep2_he_imagealignment.csv'
SKIN_ROOT = '/gpfsdata/home/renyixiang/YuanLab/data/Skin_Xenium/Human_Skin_Melanoma_Base_FFPE/'


def _find_in(root, pat_list):
    import glob
    for pat in pat_list:
        m = glob.glob(os.path.join(root, pat))
        if m:
            return m[0]
    raise FileNotFoundError(f"None of {pat_list} in {root}")


# Skin H&E — find by pattern instead of hard-coding full name
SKIN_HE = None
SKIN_HE_ALIGN = None
try:
    SKIN_HE = _find_in(SKIN_ROOT, ['*he_image.ome.tif', '*HE*.ome.tif', '*he*.ome.tif'])
    SKIN_HE_ALIGN = _find_in(SKIN_ROOT, ['*he_imagealignment.csv', '*HE*alignment*.csv', '*he*alignment*.csv'])
    logger.info(f"Skin H&E resolved: {SKIN_HE}")
    logger.info(f"Skin H&E align resolved: {SKIN_HE_ALIGN}")
except FileNotFoundError as e:
    logger.warning(f"Skin H&E not found; skin prep will fail: {e}")

RESOLUTION = 30
IMAGE_ENCODER = 'resnet50'


def prep_slice(root, he_path, align_path, out_path):
    if os.path.exists(out_path):
        logger.info(f"Skip (exists): {out_path}")
        return
    logger.info(f"Reading {root}")
    adata = se.pp.Read_Xenium(root + 'cell_feature_matrix.h5', root + 'cells.csv')
    adata = se.pp.Preprocess_adata(adata)

    img, scale = se.pp.Read_HE_image(he_path)
    tm = pd.read_csv(align_path, header=None).values
    adata = se.pp.Register_physical_to_pixel(adata, tm, scale=scale)

    he_patches, adata = se.pp.Tiling_HE_patches(RESOLUTION, adata, img)
    adata = se.pp.Extract_HE_patches_representaion(
        he_patches, adata=adata, image_encoder=IMAGE_ENCODER,
        device=None, store_key='he'
    )

    if 'spatial' not in adata.obsm:
        adata.obsm['spatial'] = adata.obs[['x_centroid', 'y_centroid']].values.astype('float32')

    # strip heavy fields to keep file small
    for k in list(adata.uns.keys()):
        adata.uns.pop(k, None)

    adata.write_h5ad(out_path, compression='gzip')
    logger.info(f"Wrote {out_path}: X={adata.shape}, he={adata.obsm['he'].shape}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['mg', 'skin'], required=True)
    ap.add_argument('--out_dir', type=str, default='/gpfsdata/home/renyixiang/YuanLab/Task8_EPS_cache/')
    args = ap.parse_args()

    out_dir = args.out_dir  # keep for logs
    os.makedirs(out_dir, exist_ok=True)

    if args.dataset == 'mg':
        prep_slice(MG_ROOT1, MG_HE1, MG_HE1_ALIGN, os.path.join(out_dir, 'mg_slice1.h5ad'))
        prep_slice(MG_ROOT2, MG_HE2, MG_HE2_ALIGN, os.path.join(out_dir, 'mg_slice2.h5ad'))
    else:
        if SKIN_HE is None or SKIN_HE_ALIGN is None:
            raise FileNotFoundError(
                f"Skin H&E image / alignment not found under {SKIN_ROOT}. "
                f"Files present: {sorted(os.listdir(SKIN_ROOT))[:20]}"
            )
        out_full = os.path.join(out_dir, 'skin_full.h5ad')
        if not os.path.exists(out_full):
            prep_slice(SKIN_ROOT, SKIN_HE, SKIN_HE_ALIGN, out_full)
        import anndata as ad
        full = ad.read_h5ad(out_full)
        x_mid = float(np.median(full.obs['x_centroid']))
        s1 = full[full.obs['x_centroid'] < x_mid].copy()
        s2 = full[full.obs['x_centroid'] >= x_mid].copy()
        s1.write_h5ad(os.path.join(out_dir, 'skin_slice1.h5ad'), compression='gzip')
        s2.write_h5ad(os.path.join(out_dir, 'skin_slice2.h5ad'), compression='gzip')
        logger.info(f"Skin split: s1={s1.shape}, s2={s2.shape}")


if __name__ == '__main__':
    main()
