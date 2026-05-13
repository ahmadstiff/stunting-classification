"""
Statistical Significance Testing Module
Skripsi: Analisis Performa Algoritma Machine Learning
         untuk Klasifikasi Status Stunting pada Balita

Implements McNemar's test for pairwise comparison of classifier performance
on the same test set. Used to determine whether differences in accuracy
between two models are statistically significant or can be attributed to chance.

Reference:
    McNemar, Q. (1947). "Note on the sampling error of the difference between
    correlated proportions or percentages". Psychometrika, 12(2), 153-157.
"""

from __future__ import annotations

import sys
import warnings
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.append(str(Path(__file__).parent.parent))

from src.config import REPORTS_PATH
from src.logging_config import get_logger

warnings.filterwarnings("ignore")

logger = get_logger(__name__)

ArrayLike = Union[Sequence[int], np.ndarray]
ResultsDict = Mapping[str, Mapping[str, Any]]


def mcnemar_contingency_table(
    y_true: ArrayLike,
    y_pred_a: ArrayLike,
    y_pred_b: ArrayLike,
) -> np.ndarray:
    """Build a 2x2 contingency table for McNemar's test.

    Table layout::

                                Model B Correct   Model B Incorrect
            Model A Correct            n00               n01
            Model A Incorrect          n10               n11

    Returns
    -------
    table : np.ndarray of shape (2, 2)
        The 2x2 contingency table ``[[n00, n01], [n10, n11]]``.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_a_arr = np.asarray(y_pred_a)
    y_pred_b_arr = np.asarray(y_pred_b)

    correct_a = y_pred_a_arr == y_true_arr
    correct_b = y_pred_b_arr == y_true_arr

    n00 = int(np.sum(correct_a & correct_b))          # both correct
    n01 = int(np.sum(correct_a & ~correct_b))         # A correct, B wrong
    n10 = int(np.sum(~correct_a & correct_b))         # A wrong, B correct
    n11 = int(np.sum(~correct_a & ~correct_b))        # both wrong

    return np.array([[n00, n01], [n10, n11]])


def mcnemar_test(
    y_true: ArrayLike,
    y_pred_a: ArrayLike,
    y_pred_b: ArrayLike,
    continuity_correction: bool = True,
) -> Dict[str, Any]:
    """Perform McNemar's test for paired nominal data.

    Parameters
    ----------
    y_true, y_pred_a, y_pred_b : array-like of shape (n_samples,)
        Ground truth labels and predictions from two models.
    continuity_correction : bool, default True
        Apply Edwards' continuity correction (recommended when
        ``b + c`` is small but > 25).

    Returns
    -------
    result : dict
        Dictionary with keys ``statistic``, ``p_value``, ``table``,
        ``b``, ``c``, ``significant_0.05``, ``significant_0.01``,
        ``interpretation``.
    """
    table = mcnemar_contingency_table(y_true, y_pred_a, y_pred_b)
    b = int(table[0, 1])  # A correct, B wrong
    c = int(table[1, 0])  # A wrong, B correct

    if (b + c) == 0:
        statistic = 0.0
        p_value = 1.0
    else:
        if continuity_correction:
            statistic = (abs(b - c) - 1.0) ** 2 / (b + c)
        else:
            statistic = (b - c) ** 2 / (b + c)
        p_value = float(1.0 - chi2.cdf(statistic, df=1))

    significant_05 = p_value < 0.05
    significant_01 = p_value < 0.01

    if b + c == 0:
        interpretation = (
            "Kedua model menghasilkan prediksi yang identik "
            "pada seluruh data uji (tidak ada perbedaan)."
        )
    elif significant_01:
        interpretation = (
            f"Perbedaan performa kedua model SIGNIFIKAN secara statistik "
            f"(p = {p_value:.4f} < 0.01)."
        )
    elif significant_05:
        interpretation = (
            f"Perbedaan performa kedua model SIGNIFIKAN secara statistik "
            f"(p = {p_value:.4f} < 0.05)."
        )
    else:
        interpretation = (
            f"Perbedaan performa kedua model TIDAK SIGNIFIKAN "
            f"(p = {p_value:.4f} >= 0.05); model dapat dianggap setara."
        )

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "table": table,
        "b": b,
        "c": c,
        "significant_0.05": bool(significant_05),
        "significant_0.01": bool(significant_01),
        "interpretation": interpretation,
    }


def run_pairwise_mcnemar(
    results: ResultsDict,
    y_test: ArrayLike,
    save_csv: bool = True,
) -> pd.DataFrame:
    """Run McNemar's test for all model pairs and return a summary DataFrame.

    Parameters
    ----------
    results : mapping of model_name -> evaluation result dict
        The value must contain a key ``'y_pred'``.
    y_test : array-like
        Ground truth labels for the test set.
    save_csv : bool, default True
        Whether to save the summary table to ``REPORTS_PATH``.
    """
    logger.info("=" * 70)
    logger.info("UJI SIGNIFIKANSI STATISTIK ANTAR MODEL (McNemar's Test)")
    logger.info("=" * 70)
    logger.info("Hipotesis:")
    logger.info("  H0 : Tidak ada perbedaan performa antara kedua model.")
    logger.info("  H1 : Terdapat perbedaan performa antara kedua model.")
    logger.info("alpha = 0.05 (Edwards continuity correction).")

    model_names = list(results.keys())
    rows: list[Dict[str, Any]] = []

    for model_a, model_b in combinations(model_names, 2):
        y_pred_a = results[model_a]["y_pred"]
        y_pred_b = results[model_b]["y_pred"]

        res = mcnemar_test(y_test, y_pred_a, y_pred_b, continuity_correction=True)

        sig_label = "signifikan" if res["significant_0.05"] else "tidak signifikan"
        logger.info("%s vs %s", model_a, model_b)
        logger.info("  b (A benar, B salah) = %d", res["b"])
        logger.info("  c (A salah, B benar) = %d", res["c"])
        logger.info("  chi-squared statistic = %.4f", res["statistic"])
        logger.info("  p-value               = %.4f", res["p_value"])
        logger.info("  keputusan             = %s pada alpha=0.05", sig_label)
        logger.info("  interpretasi          : %s", res["interpretation"])

        rows.append(
            {
                "Model A": model_a,
                "Model B": model_b,
                "b (A benar, B salah)": res["b"],
                "c (A salah, B benar)": res["c"],
                "Chi-squared": round(res["statistic"], 4),
                "p-value": round(res["p_value"], 4),
                "Signifikan (alpha=0.05)": "Ya" if res["significant_0.05"] else "Tidak",
                "Signifikan (alpha=0.01)": "Ya" if res["significant_0.01"] else "Tidak",
                "Interpretasi": res["interpretation"],
            }
        )

    summary_df = pd.DataFrame(rows)

    if save_csv:
        REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        csv_path = REPORTS_PATH / "mcnemar_pairwise_results.csv"
        summary_df.to_csv(csv_path, index=False)
        logger.info("Hasil uji McNemar disimpan ke: %s", csv_path)

    num_sig = int(summary_df["Signifikan (alpha=0.05)"].eq("Ya").sum())
    num_total = len(summary_df)
    logger.info("-" * 70)
    logger.info("Ringkasan:")
    logger.info("  Jumlah pasangan yang diuji    : %d", num_total)
    logger.info("  Signifikan pada alpha = 0.05  : %d", num_sig)
    logger.info("  Tidak signifikan              : %d", num_total - num_sig)
    if num_sig == 0:
        logger.info(
            "  Kesimpulan : Secara statistik, seluruh model tidak berbeda "
            "signifikan; perbedaan nilai metrik dapat dianggap variasi acak."
        )
    else:
        logger.info(
            "  Kesimpulan : Terdapat perbedaan performa yang signifikan pada "
            "sejumlah pasangan model; lihat tabel detail untuk keputusan per pasangan."
        )
    return summary_df


if __name__ == "__main__":
    logger.info("Statistical tests module loaded successfully!")
