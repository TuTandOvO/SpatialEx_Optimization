#!/usr/bin/env python3
"""
Stratify genes by baseline PCC into Informative / Moderate / Uninformative tiers
and report whether the PPI-augmented hypergraph improves them differently.

Reads per-gene metrics CSVs in figures/, writes stratified tables back there.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# 配置
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')

DATASETS = {
    'MG': {
        'slice1': os.path.join(FIG_DIR, 'metrics_MG_slice1.csv'),
        'slice2': os.path.join(FIG_DIR, 'metrics_MG_slice2.csv'),
        'name': 'Breast Cancer (MG)',
    },
    'Skin': {
        'slice1': os.path.join(FIG_DIR, 'metrics_skin_slice1.csv'),
        'slice2': os.path.join(FIG_DIR, 'metrics_skin_slice2.csv'),
        'name': 'Skin Melanoma',
    },
}

# 分层阈值：基于 baseline PCC 的三分位数（自动计算）
# 也可手动设定: STRAT_THRESHOLDS = (0.15, 0.30)
STRAT_THRESHOLDS = None  # None = 使用 tercile 自动分层

# 要分析的指标及其方向（higher_better）
METRICS_TO_ANALYZE = {
    'PCC':   {'delta': 'dPCC',   'bl': 'PCC_bl',   'ppi': 'PCC_ppi',   'higher_better': True},
    'SPCC':  {'delta': 'dSPCC',  'bl': 'SPCC_bl',  'ppi': 'SPCC_ppi',  'higher_better': True},
    'SSIM':  {'delta': 'dSSIM',  'bl': 'SSIM_bl',  'ppi': 'SSIM_ppi',  'higher_better': True},
    'R2':    {'delta': 'dR2',    'bl': 'R2_bl',     'ppi': 'R2_ppi',    'higher_better': True},
    'CMD':   {'delta': 'dCMD',   'bl': 'CMD_bl',    'ppi': 'CMD_ppi',   'higher_better': False},
    'MAE':   {'delta': 'dMAE',   'bl': 'MAE_bl',    'ppi': 'MAE_ppi',   'higher_better': False},
    'RMSE':  {'delta': 'dRMSE',  'bl': 'RMSE_bl',   'ppi': 'RMSE_ppi',  'higher_better': False},
    'ABC':   {'delta': 'dABC',   'bl': 'ABC_bl',    'ppi': 'ABC_ppi',   'higher_better': False},
}


def classify_genes(df, thresholds=None):
    """根据 baseline PCC 将基因分为三组。

    Args:
        df: DataFrame with at least 'PCC_bl' column.
        thresholds: (low, high) tuple. None = use terciles.

    Returns:
        df with added 'morph_group' column.
    """
    pcc_bl = df['PCC_bl'].values

    if thresholds is None:
        t_low = np.percentile(pcc_bl, 33.3)
        t_high = np.percentile(pcc_bl, 66.7)
    else:
        t_low, t_high = thresholds

    conditions = [
        pcc_bl <= t_low,
        (pcc_bl > t_low) & (pcc_bl <= t_high),
        pcc_bl > t_high,
    ]
    labels = ['Uninformative', 'Moderate', 'Informative']
    df = df.copy()
    df['morph_group'] = np.select(conditions, labels, default='Moderate')
    df['morph_group'] = pd.Categorical(df['morph_group'],
                                        categories=labels, ordered=True)

    return df, t_low, t_high


def compute_group_stats(df, metric_info, group_col='morph_group'):
    """计算每组基因在某个指标上的改善统计。"""
    rows = []
    for group_name, group_df in df.groupby(group_col, observed=True):
        n = len(group_df)
        delta_col = metric_info['delta']
        bl_col = metric_info['bl']
        ppi_col = metric_info['ppi']

        deltas = group_df[delta_col].dropna()
        bl_vals = group_df[bl_col].dropna()
        ppi_vals = group_df[ppi_col].dropna()

        # 方向感知：对于 higher_better=False 的指标，delta<0 是改善
        if metric_info['higher_better']:
            n_improved = (deltas > 0.001).sum()
            n_degraded = (deltas < -0.001).sum()
        else:
            n_improved = (deltas < -0.001).sum()
            n_degraded = (deltas > 0.001).sum()

        # Wilcoxon signed-rank test
        wilcoxon_p = np.nan
        if len(deltas) > 10:
            try:
                _, wilcoxon_p = scipy_stats.wilcoxon(deltas)
            except Exception:
                pass

        rows.append({
            'Group': group_name,
            'N_genes': n,
            'BL_mean': bl_vals.mean(),
            'BL_std': bl_vals.std(),
            'PPI_mean': ppi_vals.mean(),
            'PPI_std': ppi_vals.std(),
            'Delta_mean': deltas.mean(),
            'Delta_median': deltas.median(),
            'Delta_std': deltas.std(),
            'N_improved': int(n_improved),
            'N_degraded': int(n_degraded),
            'Pct_improved': n_improved / n * 100 if n > 0 else 0,
            'Wilcoxon_p': wilcoxon_p,
        })

    return pd.DataFrame(rows)


def compute_cross_group_correlation(df):
    """分析：baseline PCC 与 delta PCC 之间的相关性。

    如果 PPI 对 uninformative 基因改善更大，
    则 baseline PCC 与 delta PCC 应呈负相关。
    """
    pcc_bl = df['PCC_bl'].values
    dpcc = df['dPCC'].values
    mask = ~(np.isnan(pcc_bl) | np.isnan(dpcc))
    pcc_bl_clean = pcc_bl[mask]
    dpcc_clean = dpcc[mask]

    # Pearson
    r_pearson, p_pearson = scipy_stats.pearsonr(pcc_bl_clean, dpcc_clean)
    # Spearman
    r_spearman, p_spearman = scipy_stats.spearmanr(pcc_bl_clean, dpcc_clean)

    return {
        'pearson_r': r_pearson, 'pearson_p': p_pearson,
        'spearman_r': r_spearman, 'spearman_p': p_spearman,
        'n': len(pcc_bl_clean),
    }


def analyze_ppi_degree_vs_improvement(df):
    """（扩展分析）如果有 PPI degree 信息，分析 PPI 连接度与改善的关系。"""
    # 这里用 MoranI_GT 作为基因空间自相关的代理
    if 'MoranI_GT' in df.columns:
        morani = df['MoranI_GT'].values
        dpcc = df['dPCC'].values
        mask = ~(np.isnan(morani) | np.isnan(dpcc))
        if mask.sum() > 10:
            r, p = scipy_stats.spearmanr(morani[mask], dpcc[mask])
            return {'moranI_GT_vs_dPCC_spearman': r, 'p': p}
    return None


def format_significance(p):
    """格式化 p 值。"""
    if np.isnan(p):
        return 'N/A'
    elif p < 0.001:
        return f'{p:.2e} ***'
    elif p < 0.01:
        return f'{p:.4f} **'
    elif p < 0.05:
        return f'{p:.4f} *'
    else:
        return f'{p:.4f} n.s.'


# MAIN
def main():
    all_stratified_results = {}

    for dataset_key, dataset_info in DATASETS.items():
        print("\n" + "=" * 80)
        print(f"  {dataset_info['name']}")
        print("=" * 80)

        for slice_key in ['slice1', 'slice2']:
            csv_path = dataset_info[slice_key]
            if not os.path.exists(csv_path):
                print(f"  [SKIP] {csv_path} not found")
                continue

            df = pd.read_csv(csv_path)
            print(f"\n--- {dataset_key} {slice_key}: {len(df)} genes ---")

            # 1. Classify genes
            df, t_low, t_high = classify_genes(df, STRAT_THRESHOLDS)

            group_counts = df['morph_group'].value_counts().sort_index()
            print(f"\nStratification thresholds: PCC_bl <= {t_low:.4f} (Uninformative) | "
                  f"> {t_high:.4f} (Informative)")
            for g, c in group_counts.items():
                pcc_range = df.loc[df['morph_group'] == g, 'PCC_bl']
                print(f"  {g:15s}: {c:3d} genes  (PCC_bl range: [{pcc_range.min():.4f}, {pcc_range.max():.4f}])")

            # 2. Per-metric stratified analysis
            print(f"\n{'Metric':>6s} | {'Group':>14s} | {'N':>3s} | {'BL_mean':>8s} | {'PPI_mean':>8s} | "
                  f"{'Δ_mean':>8s} | {'Δ_med':>8s} | {'%Impr':>6s} | {'Wilcoxon p':>16s}")
            print("-" * 110)

            for metric_name, metric_info in METRICS_TO_ANALYZE.items():
                # Check columns exist
                if metric_info['delta'] not in df.columns:
                    continue

                stats_df = compute_group_stats(df, metric_info)
                for _, row in stats_df.iterrows():
                    sig = format_significance(row['Wilcoxon_p'])
                    print(f"{metric_name:>6s} | {row['Group']:>14s} | {row['N_genes']:3d} | "
                          f"{row['BL_mean']:8.4f} | {row['PPI_mean']:8.4f} | "
                          f"{row['Delta_mean']:+8.4f} | {row['Delta_median']:+8.4f} | "
                          f"{row['Pct_improved']:5.1f}% | {sig:>16s}")
                print()

            # 3. Cross-group correlation: baseline PCC vs delta PCC
            corr = compute_cross_group_correlation(df)
            print(f"\n[Correlation] Baseline PCC vs ΔPcc:")
            print(f"  Pearson  r = {corr['pearson_r']:+.4f}  (p = {format_significance(corr['pearson_p'])})")
            print(f"  Spearman ρ = {corr['spearman_r']:+.4f}  (p = {format_significance(corr['spearman_p'])})")
            if corr['pearson_r'] < -0.1:
                print(f"  → 负相关: PPI 对 morph-uninformative 基因改善更大 ✓")
            elif corr['pearson_r'] > 0.1:
                print(f"  → 正相关: PPI 对 morph-informative 基因改善更大")
            else:
                print(f"  → 无显著偏好性")

            # 4. Moran's I GT vs improvement
            morani_corr = analyze_ppi_degree_vs_improvement(df)
            if morani_corr:
                print(f"\n[Correlation] Moran's I (GT spatial autocorrelation) vs ΔPCC:")
                print(f"  Spearman ρ = {morani_corr['moranI_GT_vs_dPCC_spearman']:+.4f}  "
                      f"(p = {format_significance(morani_corr['p'])})")

            # 5. Key summary: improvement by group for PCC
            print(f"\n{'='*60}")
            print(f"KEY FINDING — PCC improvement by morphology group:")
            print(f"{'='*60}")
            pcc_stats = compute_group_stats(df, METRICS_TO_ANALYZE['PCC'])
            for _, row in pcc_stats.iterrows():
                bar_len = max(0, int(row['Delta_mean'] * 200))
                bar = '█' * bar_len if row['Delta_mean'] > 0 else ''
                print(f"  {row['Group']:15s}: ΔPCC = {row['Delta_mean']:+.4f}  "
                      f"({row['Pct_improved']:.0f}% improved)  {bar}")

            # Store results
            result_key = f"{dataset_key}_{slice_key}"
            all_stratified_results[result_key] = {
                'df': df,
                'thresholds': (t_low, t_high),
                'correlation': corr,
            }

            # 6. Export stratified CSV
            export_path = os.path.join(FIG_DIR, f'stratified_{dataset_key}_{slice_key}.csv')
            export_cols = ['Gene', 'morph_group', 'PCC_bl', 'PCC_ppi', 'dPCC',
                           'R2_bl', 'R2_ppi', 'dR2',
                           'SSIM_bl', 'SSIM_ppi', 'dSSIM',
                           'MAE_bl', 'MAE_ppi', 'dMAE',
                           'CMD_bl', 'CMD_ppi', 'dCMD']
            existing_cols = [c for c in export_cols if c in df.columns]
            df[existing_cols].to_csv(export_path, index=False)
            print(f"\n[SAVED] {export_path}")

    # =============================================
    # Cross-dataset summary
    # =============================================
    print("\n\n" + "#" * 80)
    print("#  CROSS-DATASET SUMMARY: Does PPI preferentially help uninformative genes?")
    print("#" * 80)

    summary_rows = []
    for key, res in all_stratified_results.items():
        df = res['df']
        corr = res['correlation']
        pcc_stats = compute_group_stats(df, METRICS_TO_ANALYZE['PCC'])

        for _, row in pcc_stats.iterrows():
            summary_rows.append({
                'Dataset': key,
                'Group': row['Group'],
                'N': row['N_genes'],
                'ΔPCC_mean': row['Delta_mean'],
                'ΔPCC_median': row['Delta_median'],
                'Pct_improved': row['Pct_improved'],
                'BL_PCC_vs_ΔPCC_r': corr['pearson_r'],
            })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False, float_format='%.4f'))

    # Final verdict
    print("\n" + "=" * 80)
    all_corrs = [res['correlation']['pearson_r'] for res in all_stratified_results.values()]
    mean_corr = np.mean(all_corrs)
    print(f"Mean correlation (BL_PCC vs ΔPCC) across all slices: {mean_corr:+.4f}")
    if mean_corr < -0.1:
        print("→ CONFIRMED: PPI+WGCNA preferentially improves morph-uninformative genes")
        print("  This supports Prof Yuan's hypothesis about PPI information transfer")
    elif mean_corr > 0.1:
        print("→ OPPOSITE: PPI+WGCNA preferentially improves morph-informative genes")
        print("  This suggests PPI reinforces existing morphology signal rather than transferring it")
    else:
        print("→ NO CLEAR PREFERENCE: PPI improvement is roughly uniform across gene categories")
    print("=" * 80)


if __name__ == '__main__':
    main()
