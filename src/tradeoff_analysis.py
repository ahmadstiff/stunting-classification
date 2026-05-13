"""
Accuracy vs. Computation-Time Trade-off Analysis
Skripsi: Analisis Performa Algoritma Machine Learning
         untuk Klasifikasi Status Stunting pada Balita

Quantifies the trade-off between predictive performance (accuracy, F1-score)
and computational cost (training time, inference time) across models.
Produces a ranking and a 2D scatter plot suitable for inclusion in BAB IV.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.config import FIGURES_PATH, FIGURE_DPI, REPORTS_PATH
from src.logging_config import get_logger

warnings.filterwarnings("ignore")

logger = get_logger(__name__)

ResultsDict = Mapping[str, Mapping[str, Any]]
ModelsDict = Mapping[str, Any]


def measure_inference_time(
    model: Any, X_test: np.ndarray, n_repeats: int = 5
) -> Tuple[float, float]:
    """Measure average inference time on ``X_test`` over ``n_repeats`` passes.

    Returns
    -------
    mean_ms_per_sample : float
        Mean milliseconds per sample across all repeats.
    total_ms : float
        Total milliseconds to predict the full test set (mean across repeats).
    """
    X_test_arr = np.asarray(X_test)
    n_samples = len(X_test_arr)
    durations: list[float] = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        _ = model.predict(X_test_arr)
        durations.append(time.perf_counter() - start)

    total_seconds = float(np.mean(durations))
    total_ms = total_seconds * 1000.0
    mean_ms_per_sample = total_ms / max(n_samples, 1)
    return mean_ms_per_sample, total_ms


def build_tradeoff_table(
    results: ResultsDict,
    models: ModelsDict,
    X_test: np.ndarray,
    n_repeats: int = 5,
) -> pd.DataFrame:
    """Build a per-model DataFrame with accuracy, F1, training time, and inference time."""
    rows: list[dict[str, Any]] = []
    for model_name, res in results.items():
        model = models.get(model_name)
        if model is None:
            inf_per_sample, inf_total = np.nan, np.nan
        else:
            inf_per_sample, inf_total = measure_inference_time(
                model, X_test, n_repeats=n_repeats
            )

        rows.append(
            {
                "Model": model_name,
                "Accuracy": round(res.get("accuracy", 0.0), 6),
                "F1-Score": round(res.get("f1", 0.0), 6),
                "Training Time (s)": round(res.get("training_time", 0.0) or 0.0, 4),
                "Inference Time Total (ms)": (
                    round(inf_total, 4) if not np.isnan(inf_total) else None
                ),
                "Inference Time per Sampel (ms)": (
                    round(inf_per_sample, 6) if not np.isnan(inf_per_sample) else None
                ),
            }
        )
    return pd.DataFrame(rows)


def compute_efficiency_scores(
    df: pd.DataFrame,
    accuracy_col: str = "F1-Score",
    time_col: str = "Training Time (s)",
) -> pd.DataFrame:
    """Compute ``Efficiency Score = accuracy / (1 + cost)`` and rank models by it."""
    df = df.copy()
    costs = df[time_col].astype(float).fillna(0.0)
    benefits = df[accuracy_col].astype(float).fillna(0.0)
    df["Efficiency Score"] = benefits / (1.0 + costs)
    df["Efficiency Score"] = df["Efficiency Score"].round(6)
    df = df.sort_values("Efficiency Score", ascending=False).reset_index(drop=True)
    df.insert(0, "Ranking Efisiensi", df.index + 1)
    return df


def plot_tradeoff_scatter(
    df: pd.DataFrame,
    x_col: str = "Training Time (s)",
    y_col: str = "F1-Score",
    save_path: Optional[Path] = None,
    display_inline: bool = True,
) -> Path:
    """Scatter plot of accuracy vs. cost (one point per model)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {
        "Decision Tree": "#9467bd",
        "Random Forest": "#1f77b4",
        "XGBoost": "#ff7f0e",
        "LightGBM": "#2ca02c",
        "CatBoost": "#d62728",
    }

    for _, row in df.iterrows():
        color = colors.get(row["Model"], "#7f7f7f")
        ax.scatter(
            row[x_col],
            row[y_col],
            s=240,
            color=color,
            edgecolor="black",
            linewidth=1.2,
            zorder=3,
            label=row["Model"],
        )
        ax.annotate(
            row["Model"],
            (row[x_col], row[y_col]),
            textcoords="offset points",
            xytext=(10, 6),
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel(x_col, fontsize=11)
    ax.set_ylabel(y_col, fontsize=11)
    ax.set_title(f"Trade-off: {y_col} vs. {x_col}", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, zorder=1)
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()

    if save_path is None:
        FIGURES_PATH.mkdir(parents=True, exist_ok=True)
        save_path = FIGURES_PATH / "tradeoff_accuracy_vs_time.png"

    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    logger.info("Gambar trade-off disimpan ke: %s", save_path)

    if display_inline:
        plt.show()
    else:
        plt.close(fig)
    return save_path


def run_tradeoff_analysis(
    results: ResultsDict,
    models: ModelsDict,
    X_test: np.ndarray,
    display_inline: bool = True,
    save_outputs: bool = True,
) -> pd.DataFrame:
    """Run the full trade-off analysis and return a ranked DataFrame."""
    logger.info("=" * 70)
    logger.info("ANALISIS TRADE-OFF AKURASI vs. WAKTU KOMPUTASI")
    logger.info("=" * 70)

    base_df = build_tradeoff_table(results, models, X_test, n_repeats=5)
    ranked_df = compute_efficiency_scores(
        base_df, accuracy_col="F1-Score", time_col="Training Time (s)"
    )

    logger.info("Tabel Trade-off (Akurasi, F1, Waktu, Efisiensi):")
    logger.info("%s", ranked_df.to_string(index=False))

    best_row = ranked_df.iloc[0]
    logger.info(
        "Model dengan trade-off terbaik (efisiensi tertinggi): %s "
        "(F1 = %.4f, Training Time = %.4f detik, Efficiency Score = %.4f).",
        best_row["Model"],
        best_row["F1-Score"],
        best_row["Training Time (s)"],
        best_row["Efficiency Score"],
    )

    if save_outputs:
        REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        csv_path = REPORTS_PATH / "tradeoff_analysis.csv"
        ranked_df.to_csv(csv_path, index=False)
        logger.info("Tabel trade-off disimpan ke: %s", csv_path)

        plot_tradeoff_scatter(
            ranked_df,
            x_col="Training Time (s)",
            y_col="F1-Score",
            display_inline=display_inline,
        )

    return ranked_df


if __name__ == "__main__":
    logger.info("Trade-off analysis module loaded successfully!")
