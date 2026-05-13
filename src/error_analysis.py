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

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(str(Path(__file__).parent.parent))

from src.config import FIGURES_PATH, FIGURE_DPI, NUMERICAL_FEATURES, REPORTS_PATH
from src.logging_config import get_logger

warnings.filterwarnings("ignore")

logger = get_logger(__name__)

ArrayLike = Union[Sequence[int], np.ndarray]
ResultsDict = Mapping[str, Mapping[str, Any]]


def build_error_dataframe(
    X_test: np.ndarray,
    y_test: ArrayLike,
    feature_names: Sequence[str],
    results: ResultsDict,
) -> pd.DataFrame:
    """Build a per-sample DataFrame with predictions and correctness flags.

    Columns: one per feature (from ``feature_names``), plus ``y_true`` and
    (for each model) ``<model>_pred`` / ``<model>_correct`` (1 if correct).
    """
    X_test_arr = np.asarray(X_test)
    y_test_arr = np.asarray(y_test)

    df = pd.DataFrame(X_test_arr, columns=list(feature_names))
    df["y_true"] = y_test_arr

    for model_name, res in results.items():
        y_pred = np.asarray(res["y_pred"])
        df[f"{model_name}_pred"] = y_pred
        df[f"{model_name}_correct"] = (y_pred == y_test_arr).astype(int)

    return df


def summarize_errors_per_model(df: pd.DataFrame, results: ResultsDict) -> pd.DataFrame:
    """Summarize FP, FN, total errors, and error rate per model."""
    rows: List[Dict[str, Any]] = []
    y_true = df["y_true"].to_numpy()
    n_total = len(df)

    for model_name in results.keys():
        y_pred = df[f"{model_name}_pred"].to_numpy()
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        total_err = fp + fn
        rows.append(
            {
                "Model": model_name,
                "False Positive": fp,
                "False Negative": fn,
                "Total Errors": total_err,
                "Error Rate (%)": round(total_err / n_total * 100, 4),
            }
        )

    return pd.DataFrame(rows)


def feature_stats_by_correctness(
    df: pd.DataFrame, feature_col: str, model_name: str
) -> Dict[str, Any]:
    """Descriptive statistics of ``feature_col`` split by correct vs. incorrect predictions."""
    correct_mask = df[f"{model_name}_correct"] == 1
    return {
        "Benar (mean ± std)": (
            f"{df.loc[correct_mask, feature_col].mean():.2f} "
            f"± {df.loc[correct_mask, feature_col].std():.2f}"
        ),
        "Salah (mean ± std)": (
            f"{df.loc[~correct_mask, feature_col].mean():.2f} "
            f"± {df.loc[~correct_mask, feature_col].std():.2f}"
        ),
        "Jumlah Benar": int(correct_mask.sum()),
        "Jumlah Salah": int((~correct_mask).sum()),
    }


def analyze_error_feature_distribution(
    df: pd.DataFrame,
    results: ResultsDict,
    feature_cols: Sequence[str],
    save_csv: bool = True,
) -> pd.DataFrame:
    """Compare feature distributions between correct and incorrect predictions per model."""
    rows: List[Dict[str, Any]] = []
    for model_name in results.keys():
        for feat in feature_cols:
            if feat not in df.columns:
                continue
            stats = feature_stats_by_correctness(df, feat, model_name)
            rows.append(
                {
                    "Model": model_name,
                    "Feature": feat,
                    "Benar (mean ± std)": stats["Benar (mean ± std)"],
                    "Salah (mean ± std)": stats["Salah (mean ± std)"],
                    "Jumlah Benar": stats["Jumlah Benar"],
                    "Jumlah Salah": stats["Jumlah Salah"],
                }
            )
    out_df = pd.DataFrame(rows)

    if save_csv:
        REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        csv_path = REPORTS_PATH / "error_feature_distribution.csv"
        out_df.to_csv(csv_path, index=False)
        logger.info("Distribusi fitur pada error disimpan ke: %s", csv_path)

    return out_df


def shared_errors_analysis(df: pd.DataFrame, results: ResultsDict) -> Dict[str, Any]:
    """Identify samples misclassified by ALL models and unique-per-model errors."""
    correct_cols = [f"{m}_correct" for m in results.keys()]
    wrong_matrix = (1 - df[correct_cols].to_numpy())  # 1 where model is wrong

    n_wrong_per_sample = wrong_matrix.sum(axis=1)
    n_models = len(correct_cols)

    all_wrong_mask = n_wrong_per_sample == n_models
    any_wrong_mask = n_wrong_per_sample > 0

    only_one_wrong_counts: Dict[str, int] = {}
    for i, m in enumerate(results.keys()):
        unique_err = (wrong_matrix[:, i] == 1) & (n_wrong_per_sample == 1)
        only_one_wrong_counts[m] = int(unique_err.sum())

    all_wrong_df = df.loc[all_wrong_mask].copy()

    return {
        "all_wrong_count": int(all_wrong_mask.sum()),
        "any_wrong_count": int(any_wrong_mask.sum()),
        "all_wrong_df": all_wrong_df,
        "only_one_wrong_counts": only_one_wrong_counts,
        "total_samples": int(len(df)),
    }


def plot_error_distribution_by_feature(
    df: pd.DataFrame,
    results: ResultsDict,
    feature_col: str,
    save_path: Optional[Path] = None,
    display_inline: bool = True,
) -> Optional[Path]:
    """Boxplot comparing feature values for correctly vs. incorrectly classified samples."""
    if feature_col not in df.columns:
        return None

    n_models = len(results)
    # Use 2 columns for better readability in thesis
    n_cols = min(n_models, 2)
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 6 * n_rows), sharey=True)
    if n_models == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()
    
    # Hide unused axes
    for i in range(n_models, n_rows * n_cols):
        axes[i].axis('off')

    for ax, model_name in zip(axes, results.keys()):
        correct_col = f"{model_name}_correct"
        df_plot = df[[feature_col, correct_col]].copy()
        df_plot["Status Prediksi"] = np.where(
            df_plot[correct_col] == 1, "Benar", "Salah"
        )
        sns.boxplot(
            data=df_plot,
            x="Status Prediksi",
            y=feature_col,
            ax=ax,
            palette={"Benar": "#2ecc71", "Salah": "#e74c3c"},
        )
        ax.set_title(model_name)
        ax.set_xlabel("")

    fig.suptitle(
        f"Distribusi {feature_col} pada Prediksi Benar vs. Salah",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()

    if save_path is None:
        FIGURES_PATH.mkdir(parents=True, exist_ok=True)
        safe_feat = feature_col.replace(" ", "_").replace("(", "").replace(")", "")
        save_path = FIGURES_PATH / f"error_distribution_{safe_feat}.png"

    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    logger.info("Gambar distribusi error (%s) disimpan ke: %s", feature_col, save_path)

    if display_inline:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def run_error_analysis(
    X_test: np.ndarray,
    y_test: ArrayLike,
    feature_names: Sequence[str],
    results: ResultsDict,
    save_outputs: bool = True,
    display_inline: bool = True,
) -> Dict[str, Any]:
    """Run the full error analysis and return a structured report."""
    logger.info("=" * 70)
    logger.info("ANALISIS ERROR (Pola Kesalahan Prediksi)")
    logger.info("=" * 70)

    df = build_error_dataframe(X_test, y_test, feature_names, results)

    # 1. Per-model summary
    per_model_summary = summarize_errors_per_model(df, results)
    logger.info("1. Ringkasan Error per Model")
    logger.info("%s", per_model_summary.to_string(index=False))

    # 2. Feature distribution for errors
    numeric_features = [f for f in NUMERICAL_FEATURES if f in df.columns]
    feature_distribution = analyze_error_feature_distribution(
        df, results, numeric_features, save_csv=save_outputs
    )
    logger.info("2. Distribusi Fitur pada Prediksi Benar vs. Salah")
    logger.info("%s", feature_distribution.to_string(index=False))

    # 3. Shared errors
    shared = shared_errors_analysis(df, results)
    logger.info("3. Analisis Error Bersama (Hard Samples)")
    logger.info("  Total sampel uji                 : %d", shared["total_samples"])
    logger.info("  Salah oleh setidaknya 1 model    : %d", shared["any_wrong_count"])
    logger.info("  Salah oleh SEMUA model           : %d", shared["all_wrong_count"])
    logger.info("  Unique errors (salah hanya di 1 model):")
    for m, n in shared["only_one_wrong_counts"].items():
        logger.info("    - %-15s : %d", m, n)

    if shared["all_wrong_count"] > 0 and len(numeric_features) > 0:
        logger.info("  Profil sampel yang salah di semua model:")
        desc = shared["all_wrong_df"][numeric_features + ["y_true"]].describe()
        logger.info("%s", desc.to_string())

    # 4. Plots
    if save_outputs:
        for feat in numeric_features:
            plot_error_distribution_by_feature(
                df, results, feat, display_inline=display_inline
            )
        REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        per_model_summary.to_csv(
            REPORTS_PATH / "error_summary_per_model.csv", index=False
        )

    return {
        "df": df,
        "per_model_summary": per_model_summary,
        "feature_distribution": feature_distribution,
        "shared_errors": shared,
    }


if __name__ == "__main__":
    logger.info("Error analysis module loaded successfully!")
