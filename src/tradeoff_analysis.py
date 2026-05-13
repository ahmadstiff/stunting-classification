"""
Accuracy vs. Computation-Time Trade-off Analysis
Skripsi: Analisis Performa Algoritma Machine Learning
         untuk Klasifikasi Status Stunting pada Balita

Quantifies the trade-off between predictive performance (accuracy, F1-score)
and computational cost (training time, inference time) across models.
Produces a ranking and a 2D scatter plot suitable for inclusion in BAB IV.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import REPORTS_PATH, FIGURES_PATH, FIGURE_DPI


def measure_inference_time(model, X_test, n_repeats=5):
    """
    Measure average inference time (in milliseconds per sample) over n_repeats
    passes on X_test.

    Returns
    -------
    mean_ms_per_sample : float
    total_ms : float
        Total milliseconds for predicting X_test on the last run.
    """
    X_test = np.asarray(X_test)
    n_samples = len(X_test)
    durations = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        _ = model.predict(X_test)
        durations.append(time.perf_counter() - start)

    total_seconds = float(np.mean(durations))
    total_ms = total_seconds * 1000.0
    mean_ms_per_sample = total_ms / max(n_samples, 1)
    return mean_ms_per_sample, total_ms


def build_tradeoff_table(results, models, X_test, n_repeats=5):
    """
    Build a DataFrame summarizing each model's accuracy, F1-score,
    training time, and measured inference time.

    Parameters
    ----------
    results : dict[str, dict]
        Per-model evaluation results (must contain accuracy, f1, training_time).
    models : dict[str, estimator]
        Fitted model objects (must expose .predict).
    X_test : array-like
        Test features.
    n_repeats : int
        Number of timing repeats for inference.

    Returns
    -------
    df : pandas.DataFrame
    """
    rows = []
    for model_name, res in results.items():
        model = models.get(model_name)
        if model is None:
            inf_per_sample, inf_total = np.nan, np.nan
        else:
            inf_per_sample, inf_total = measure_inference_time(
                model, X_test, n_repeats=n_repeats
            )

        rows.append({
            'Model': model_name,
            'Accuracy': round(res.get('accuracy', 0.0), 6),
            'F1-Score': round(res.get('f1', 0.0), 6),
            'Training Time (s)': round(res.get('training_time', 0.0) or 0.0, 4),
            'Inference Time Total (ms)': round(inf_total, 4)
                if not np.isnan(inf_total) else None,
            'Inference Time per Sampel (ms)': round(inf_per_sample, 6)
                if not np.isnan(inf_per_sample) else None,
        })
    return pd.DataFrame(rows)


def compute_efficiency_scores(df, accuracy_col='F1-Score',
                              time_col='Training Time (s)'):
    """
    Compute a simple efficiency score defined as:
        efficiency = accuracy_col / (1 + time_col)

    Higher means better trade-off.

    Parameters
    ----------
    df : DataFrame returned by build_tradeoff_table.
    accuracy_col, time_col : str
        Columns to use as "benefit" and "cost".
    """
    df = df.copy()
    costs = df[time_col].astype(float).fillna(0.0)
    benefits = df[accuracy_col].astype(float).fillna(0.0)
    df['Efficiency Score'] = benefits / (1.0 + costs)
    df['Efficiency Score'] = df['Efficiency Score'].round(6)
    df = df.sort_values('Efficiency Score', ascending=False).reset_index(drop=True)
    df.insert(0, 'Ranking Efisiensi', df.index + 1)
    return df


def plot_tradeoff_scatter(
    df,
    x_col='Training Time (s)',
    y_col='F1-Score',
    save_path=None,
    display_inline=True,
):
    """
    Scatter plot of accuracy vs. cost (time). Each point = one model.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {
        'Random Forest': '#1f77b4',
        'XGBoost': '#ff7f0e',
        'LightGBM': '#2ca02c',
        'CatBoost': '#d62728',
    }

    for _, row in df.iterrows():
        color = colors.get(row['Model'], '#7f7f7f')
        ax.scatter(row[x_col], row[y_col], s=240, color=color,
                   edgecolor='black', linewidth=1.2, zorder=3, label=row['Model'])
        ax.annotate(
            row['Model'],
            (row[x_col], row[y_col]),
            textcoords="offset points", xytext=(10, 6),
            fontsize=10, fontweight='bold',
        )

    ax.set_xlabel(x_col, fontsize=11)
    ax.set_ylabel(y_col, fontsize=11)
    ax.set_title(f'Trade-off: {y_col} vs. {x_col}',
                 fontsize=13, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5, zorder=1)
    ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()

    if save_path is None:
        FIGURES_PATH.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_PATH / 'tradeoff_accuracy_vs_time.png'

    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"Gambar trade-off disimpan ke: {save_path}")

    if display_inline:
        plt.show()
    else:
        plt.close(fig)


def run_tradeoff_analysis(
    results, models, X_test, display_inline=True, save_outputs=True
):
    """
    Top-level driver to compute and print the trade-off analysis.

    Returns
    -------
    tradeoff_df : DataFrame with columns including 'Efficiency Score'.
    """
    print("\n" + "=" * 70)
    print("  ANALISIS TRADE-OFF AKURASI vs. WAKTU KOMPUTASI")
    print("=" * 70)

    base_df = build_tradeoff_table(results, models, X_test, n_repeats=5)
    ranked_df = compute_efficiency_scores(
        base_df, accuracy_col='F1-Score', time_col='Training Time (s)'
    )

    print("\nTabel Trade-off (Akurasi, F1, Waktu, Efisiensi):")
    print("-" * 70)
    print(ranked_df.to_string(index=False))

    # Rekomendasi sederhana berdasarkan efisiensi
    best_row = ranked_df.iloc[0]
    print(
        f"\nModel dengan trade-off terbaik (efisiensi tertinggi): "
        f"{best_row['Model']} "
        f"(F1 = {best_row['F1-Score']:.4f}, "
        f"Training Time = {best_row['Training Time (s)']:.4f} detik, "
        f"Efficiency Score = {best_row['Efficiency Score']:.4f})."
    )

    if save_outputs:
        REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        csv_path = REPORTS_PATH / 'tradeoff_analysis.csv'
        ranked_df.to_csv(csv_path, index=False)
        print(f"Tabel trade-off disimpan ke: {csv_path}")

        plot_tradeoff_scatter(
            ranked_df, x_col='Training Time (s)', y_col='F1-Score',
            display_inline=display_inline,
        )

    return ranked_df


if __name__ == "__main__":
    print("Trade-off analysis module loaded successfully!")
    print("Exposed functions: run_tradeoff_analysis, build_tradeoff_table, "
          "compute_efficiency_scores, plot_tradeoff_scatter")
