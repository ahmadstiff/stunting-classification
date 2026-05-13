"""
Error Analysis Module
Skripsi: Analisis Performa Algoritma Machine Learning
         untuk Klasifikasi Status Stunting pada Balita

Provides tools to analyze misclassification patterns of trained models,
including:
    - Error distribution by feature value (Umur, Tinggi Badan, Jenis Kelamin).
    - False positives vs false negatives characterization.
    - Shared errors across all models (difficult samples).
    - Per-feature misclassification summaries.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import REPORTS_PATH, FIGURES_PATH, BINARY_CLASS_NAMES, FIGURE_DPI


def build_error_dataframe(X_test, y_test, feature_names, results):
    """
    Build a DataFrame with per-sample predictions and correctness flags
    for each model, plus the original feature values.

    Parameters
    ----------
    X_test : array-like of shape (n_samples, n_features)
        Test feature matrix (after encoding).
    y_test : array-like of shape (n_samples,)
        True binary labels.
    feature_names : list[str]
        Feature column names matching X_test columns.
    results : dict[str, dict]
        Mapping model name -> evaluation result dict (containing 'y_pred').

    Returns
    -------
    df : pandas.DataFrame
        One row per test sample. Contains feature columns, 'y_true',
        and <model>_pred / <model>_correct for each model.
    """
    X_test_arr = np.asarray(X_test)
    y_test_arr = np.asarray(y_test)

    df = pd.DataFrame(X_test_arr, columns=feature_names)
    df['y_true'] = y_test_arr

    for model_name, res in results.items():
        y_pred = np.asarray(res['y_pred'])
        df[f'{model_name}_pred'] = y_pred
        df[f'{model_name}_correct'] = (y_pred == y_test_arr).astype(int)

    return df


def summarize_errors_per_model(df, results):
    """
    Summarize error counts (FP, FN, total) for each model.

    Returns
    -------
    summary_df : pandas.DataFrame
        Columns: Model, Total Errors, False Positive (prediksi Stunting, aktual
        Tidak Stunting), False Negative (prediksi Tidak Stunting, aktual
        Stunting), Error Rate (%).
    """
    rows = []
    y_true = df['y_true'].values
    n_total = len(df)

    for model_name in results.keys():
        y_pred = df[f'{model_name}_pred'].values
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        total_err = fp + fn
        rows.append({
            'Model': model_name,
            'False Positive': fp,
            'False Negative': fn,
            'Total Errors': total_err,
            'Error Rate (%)': round(total_err / n_total * 100, 4),
        })

    return pd.DataFrame(rows)


def feature_stats_by_correctness(df, feature_col, model_name):
    """
    Compute descriptive statistics of a numeric feature, split by
    correct vs. incorrect predictions from the given model.
    """
    correct_mask = df[f'{model_name}_correct'] == 1
    stats = {
        'Benar (mean ± std)': f"{df.loc[correct_mask, feature_col].mean():.2f} "
                              f"± {df.loc[correct_mask, feature_col].std():.2f}",
        'Salah (mean ± std)': f"{df.loc[~correct_mask, feature_col].mean():.2f} "
                              f"± {df.loc[~correct_mask, feature_col].std():.2f}",
        'Jumlah Benar': int(correct_mask.sum()),
        'Jumlah Salah': int((~correct_mask).sum()),
    }
    return stats


def analyze_error_feature_distribution(df, results, feature_cols, save_csv=True):
    """
    Compare feature distributions between correctly and incorrectly
    classified samples for each model.

    Parameters
    ----------
    df : pandas.DataFrame
        From build_error_dataframe.
    results : dict[str, dict]
    feature_cols : list[str]
        Numeric features to analyze (e.g., ['Umur (bulan)', 'Tinggi Badan (cm)']).
    save_csv : bool, default True.

    Returns
    -------
    combined_df : pandas.DataFrame
        Long-format summary: rows = (model, feature), with benar/salah stats.
    """
    rows = []
    for model_name in results.keys():
        for feat in feature_cols:
            if feat not in df.columns:
                continue
            stats = feature_stats_by_correctness(df, feat, model_name)
            rows.append({
                'Model': model_name,
                'Feature': feat,
                'Benar (mean ± std)': stats['Benar (mean ± std)'],
                'Salah (mean ± std)': stats['Salah (mean ± std)'],
                'Jumlah Benar': stats['Jumlah Benar'],
                'Jumlah Salah': stats['Jumlah Salah'],
            })
    out_df = pd.DataFrame(rows)

    if save_csv:
        REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        csv_path = REPORTS_PATH / 'error_feature_distribution.csv'
        out_df.to_csv(csv_path, index=False)
        print(f"Distribusi fitur pada error disimpan ke: {csv_path}")

    return out_df


def shared_errors_analysis(df, results):
    """
    Identify samples misclassified by ALL models ("hard samples") and those
    misclassified by only ONE model ("unique errors").

    Returns
    -------
    summary : dict
        Keys:
            'all_wrong_count', 'any_wrong_count',
            'all_wrong_df' (DataFrame of hard samples),
            'only_one_wrong_counts' (dict model -> count of unique errors).
    """
    correct_cols = [f'{m}_correct' for m in results.keys()]
    wrong_matrix = (1 - df[correct_cols].values)  # 1 where model is wrong

    n_wrong_per_sample = wrong_matrix.sum(axis=1)
    n_models = len(correct_cols)

    all_wrong_mask = n_wrong_per_sample == n_models
    any_wrong_mask = n_wrong_per_sample > 0

    only_one_wrong_counts = {}
    for i, m in enumerate(results.keys()):
        unique_err = (wrong_matrix[:, i] == 1) & (n_wrong_per_sample == 1)
        only_one_wrong_counts[m] = int(unique_err.sum())

    all_wrong_df = df.loc[all_wrong_mask].copy()

    return {
        'all_wrong_count': int(all_wrong_mask.sum()),
        'any_wrong_count': int(any_wrong_mask.sum()),
        'all_wrong_df': all_wrong_df,
        'only_one_wrong_counts': only_one_wrong_counts,
        'total_samples': int(len(df)),
    }


def plot_error_distribution_by_feature(
    df, results, feature_col,
    save_path=None, display_inline=True,
):
    """
    Create a boxplot comparing feature values for correctly vs. incorrectly
    classified samples, per model.
    """
    if feature_col not in df.columns:
        return

    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, results.keys()):
        correct_col = f'{model_name}_correct'
        df_plot = df[[feature_col, correct_col]].copy()
        df_plot['Status Prediksi'] = np.where(
            df_plot[correct_col] == 1, 'Benar', 'Salah'
        )
        sns.boxplot(
            data=df_plot, x='Status Prediksi', y=feature_col, ax=ax,
            palette={'Benar': '#2ecc71', 'Salah': '#e74c3c'}
        )
        ax.set_title(model_name)
        ax.set_xlabel('')

    fig.suptitle(f'Distribusi {feature_col} pada Prediksi Benar vs. Salah',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path is None:
        FIGURES_PATH.mkdir(parents=True, exist_ok=True)
        safe_feat = feature_col.replace(' ', '_').replace('(', '').replace(')', '')
        save_path = FIGURES_PATH / f'error_distribution_{safe_feat}.png'

    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"Gambar distribusi error ({feature_col}) disimpan ke: {save_path}")

    if display_inline:
        plt.show()
    else:
        plt.close(fig)


def run_error_analysis(X_test, y_test, feature_names, results,
                       save_outputs=True, display_inline=True):
    """
    Top-level driver to run the full error analysis and print a report.

    Returns
    -------
    report : dict
        Contains: 'df', 'per_model_summary', 'feature_distribution',
        'shared_errors'.
    """
    print("\n" + "=" * 70)
    print("  ANALISIS ERROR (Pola Kesalahan Prediksi)")
    print("=" * 70)

    df = build_error_dataframe(X_test, y_test, feature_names, results)

    # 1. Per-model summary
    per_model_summary = summarize_errors_per_model(df, results)
    print("\n1. Ringkasan Error per Model")
    print("-" * 70)
    print(per_model_summary.to_string(index=False))

    # 2. Feature distribution for errors
    numeric_features = [f for f in ['Umur (bulan)', 'Tinggi Badan (cm)']
                        if f in df.columns]
    feature_distribution = analyze_error_feature_distribution(
        df, results, numeric_features, save_csv=save_outputs
    )
    print("\n2. Distribusi Fitur pada Prediksi Benar vs. Salah")
    print("-" * 70)
    print(feature_distribution.to_string(index=False))

    # 3. Shared errors
    shared = shared_errors_analysis(df, results)
    print("\n3. Analisis Error Bersama (Hard Samples)")
    print("-" * 70)
    print(f"  Total sampel uji                 : {shared['total_samples']}")
    print(f"  Salah oleh setidaknya 1 model    : {shared['any_wrong_count']}")
    print(f"  Salah oleh SEMUA model           : {shared['all_wrong_count']}")
    print("  Unique errors (salah hanya di 1 model):")
    for m, n in shared['only_one_wrong_counts'].items():
        print(f"    - {m:15} : {n}")

    if shared['all_wrong_count'] > 0 and len(numeric_features) > 0:
        print("\n  Profil sampel yang salah di semua model:")
        desc = shared['all_wrong_df'][numeric_features + ['y_true']].describe()
        print(desc.to_string())

    # 4. Plots
    if save_outputs:
        for feat in numeric_features:
            plot_error_distribution_by_feature(
                df, results, feat, display_inline=display_inline
            )
        REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        per_model_summary.to_csv(
            REPORTS_PATH / 'error_summary_per_model.csv', index=False
        )

    return {
        'df': df,
        'per_model_summary': per_model_summary,
        'feature_distribution': feature_distribution,
        'shared_errors': shared,
    }


if __name__ == "__main__":
    print("Error analysis module loaded successfully!")
    print("Exposed functions: run_error_analysis, build_error_dataframe, "
          "summarize_errors_per_model, shared_errors_analysis, "
          "plot_error_distribution_by_feature")
