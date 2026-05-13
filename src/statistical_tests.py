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

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import chi2
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import REPORTS_PATH


def mcnemar_contingency_table(y_true, y_pred_a, y_pred_b):
    """
    Build a 2x2 contingency table for McNemar's test.

    Table layout:
                            Model B Correct   Model B Incorrect
        Model A Correct            n00               n01
        Model A Incorrect          n10               n11

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels.
    y_pred_a : array-like of shape (n_samples,)
        Predictions from model A.
    y_pred_b : array-like of shape (n_samples,)
        Predictions from model B.

    Returns
    -------
    table : np.ndarray of shape (2, 2)
        The 2x2 contingency table [[n00, n01], [n10, n11]].
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    n00 = int(np.sum(correct_a & correct_b))          # both correct
    n01 = int(np.sum(correct_a & ~correct_b))         # A correct, B wrong
    n10 = int(np.sum(~correct_a & correct_b))         # A wrong, B correct
    n11 = int(np.sum(~correct_a & ~correct_b))        # both wrong

    return np.array([[n00, n01], [n10, n11]])


def mcnemar_test(y_true, y_pred_a, y_pred_b, continuity_correction=True):
    """
    Perform McNemar's test for paired nominal data.

    The test evaluates the null hypothesis that the marginal frequencies of
    disagreements between two models are equal (i.e., the two models have
    the same error rate on the same test set).

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred_a : array-like
        Predictions from model A.
    y_pred_b : array-like
        Predictions from model B.
    continuity_correction : bool, default True
        Apply Edwards' continuity correction (recommended when
        b + c is small but > 25). For very small b + c (< 25), the
        binomial exact test should be used.

    Returns
    -------
    result : dict
        Dictionary containing:
            - 'statistic': McNemar test statistic (chi-squared, 1 df).
            - 'p_value': Two-sided p-value.
            - 'table': 2x2 contingency table.
            - 'b': Count where A correct and B incorrect.
            - 'c': Count where A incorrect and B correct.
            - 'significant_0.05': True if p < 0.05.
            - 'significant_0.01': True if p < 0.01.
            - 'interpretation': Human-readable interpretation.
    """
    table = mcnemar_contingency_table(y_true, y_pred_a, y_pred_b)
    b = table[0, 1]  # A correct, B wrong
    c = table[1, 0]  # A wrong, B correct

    if (b + c) == 0:
        # Identical predictions — no disagreement at all.
        statistic = 0.0
        p_value = 1.0
    else:
        if continuity_correction:
            statistic = (abs(b - c) - 1.0) ** 2 / (b + c)
        else:
            statistic = (b - c) ** 2 / (b + c)
        # Chi-squared distribution with 1 degree of freedom.
        p_value = 1.0 - chi2.cdf(statistic, df=1)

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
        'statistic': float(statistic),
        'p_value': float(p_value),
        'table': table,
        'b': int(b),
        'c': int(c),
        'significant_0.05': bool(significant_05),
        'significant_0.01': bool(significant_01),
        'interpretation': interpretation,
    }


def run_pairwise_mcnemar(results, y_test, save_csv=True):
    """
    Run McNemar's test for all pairs of models and return a summary DataFrame.

    Parameters
    ----------
    results : dict[str, dict]
        Dictionary mapping model name to evaluation result dict
        (must contain key 'y_pred').
    y_test : array-like
        Ground truth labels for the test set.
    save_csv : bool, default True
        Whether to save the summary table to disk.

    Returns
    -------
    summary_df : pandas.DataFrame
        DataFrame with one row per model pair.
    """
    print("\n" + "=" * 70)
    print("  UJI SIGNIFIKANSI STATISTIK ANTAR MODEL (McNemar's Test)")
    print("=" * 70)
    print("""
Hipotesis:
  H0 : Tidak ada perbedaan performa antara kedua model.
  H1 : Terdapat perbedaan performa antara kedua model.
Tingkat signifikansi : alpha = 0.05 (dengan Edwards continuity correction).
""")

    model_names = list(results.keys())
    rows = []

    for model_a, model_b in combinations(model_names, 2):
        y_pred_a = results[model_a]['y_pred']
        y_pred_b = results[model_b]['y_pred']

        res = mcnemar_test(y_test, y_pred_a, y_pred_b, continuity_correction=True)

        sig_label = "signifikan" if res['significant_0.05'] else "tidak signifikan"
        print(f"\n{model_a} vs {model_b}")
        print(f"  - b (A benar, B salah) = {res['b']}")
        print(f"  - c (A salah, B benar) = {res['c']}")
        print(f"  - chi-squared statistic = {res['statistic']:.4f}")
        print(f"  - p-value               = {res['p_value']:.4f}")
        print(f"  - keputusan             = {sig_label} pada alpha=0.05")
        print(f"  - interpretasi          : {res['interpretation']}")

        rows.append({
            'Model A': model_a,
            'Model B': model_b,
            'b (A benar, B salah)': res['b'],
            'c (A salah, B benar)': res['c'],
            'Chi-squared': round(res['statistic'], 4),
            'p-value': round(res['p_value'], 4),
            'Signifikan (alpha=0.05)': 'Ya' if res['significant_0.05'] else 'Tidak',
            'Signifikan (alpha=0.01)': 'Ya' if res['significant_0.01'] else 'Tidak',
            'Interpretasi': res['interpretation'],
        })

    summary_df = pd.DataFrame(rows)

    if save_csv:
        REPORTS_PATH.mkdir(parents=True, exist_ok=True)
        csv_path = REPORTS_PATH / 'mcnemar_pairwise_results.csv'
        summary_df.to_csv(csv_path, index=False)
        print(f"\nHasil uji McNemar disimpan ke: {csv_path}")

    # Kesimpulan keseluruhan
    num_sig = int(summary_df['Signifikan (alpha=0.05)'].eq('Ya').sum())
    num_total = len(summary_df)
    print("\n" + "-" * 70)
    print("Ringkasan:")
    print(f"  Jumlah pasangan yang diuji    : {num_total}")
    print(f"  Signifikan pada alpha = 0.05  : {num_sig}")
    print(f"  Tidak signifikan              : {num_total - num_sig}")
    if num_sig == 0:
        print(
            "  Kesimpulan : Secara statistik, keempat model tidak berbeda\n"
            "               signifikan. Perbedaan nilai metrik yang diamati\n"
            "               dapat dianggap sebagai variasi acak."
        )
    else:
        print(
            "  Kesimpulan : Terdapat perbedaan performa yang signifikan\n"
            "               pada sejumlah pasangan model. Lihat tabel detail\n"
            "               untuk keputusan per pasangan."
        )
    return summary_df


if __name__ == "__main__":
    print("Statistical tests module loaded successfully!")
    print("Exposed functions: mcnemar_test, run_pairwise_mcnemar")
