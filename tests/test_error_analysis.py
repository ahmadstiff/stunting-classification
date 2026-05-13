"""Unit tests for src.error_analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.error_analysis import (
    build_error_dataframe,
    shared_errors_analysis,
    summarize_errors_per_model,
)


FEATURE_NAMES = ["Jenis Kelamin", "Umur (bulan)", "Tinggi Badan (cm)"]


def _make_dummy_data():
    """Construct a small deterministic scenario with known FP/FN counts."""
    X_test = np.array(
        [
            [0, 12, 75.0],   # idx 0
            [1, 24, 82.5],   # idx 1
            [0, 36, 90.0],   # idx 2
            [1, 48, 95.0],   # idx 3
            [0, 60, 102.0],  # idx 4
            [1, 18, 78.0],   # idx 5
        ]
    )
    y_test = np.array([0, 1, 0, 1, 0, 1])
    results = {
        # Model A: perfect
        "A": {"y_pred": np.array([0, 1, 0, 1, 0, 1])},
        # Model B: 2 errors — 1 FP (index 2: pred 1 vs actual 0) and 1 FN (index 5: pred 0 vs actual 1)
        "B": {"y_pred": np.array([0, 1, 1, 1, 0, 0])},
        # Model C: 3 errors — indices 0 (FP), 1 (FN), 4 (FP)
        "C": {"y_pred": np.array([1, 0, 0, 1, 1, 1])},
    }
    return X_test, y_test, results


class TestBuildErrorDataframe:
    def test_columns_and_rows(self):
        X, y, results = _make_dummy_data()
        df = build_error_dataframe(X, y, FEATURE_NAMES, results)

        # Feature columns preserved + y_true + pred/correct per model
        for feat in FEATURE_NAMES:
            assert feat in df.columns
        assert "y_true" in df.columns
        for m in results:
            assert f"{m}_pred" in df.columns
            assert f"{m}_correct" in df.columns
        assert len(df) == len(y)

    def test_correct_flag_values(self):
        X, y, results = _make_dummy_data()
        df = build_error_dataframe(X, y, FEATURE_NAMES, results)

        # Model A is perfect
        assert df["A_correct"].sum() == len(y)
        # Model B has 2 errors (one FP, one FN)
        assert df["B_correct"].sum() == len(y) - 2
        # Model C has 3 errors
        assert df["C_correct"].sum() == len(y) - 3


class TestSummarizeErrorsPerModel:
    def test_fp_fn_counts(self):
        X, y, results = _make_dummy_data()
        df = build_error_dataframe(X, y, FEATURE_NAMES, results)
        summary = summarize_errors_per_model(df, results)

        assert set(summary.columns) == {
            "Model",
            "False Positive",
            "False Negative",
            "Total Errors",
            "Error Rate (%)",
        }
        row_a = summary.loc[summary["Model"] == "A"].iloc[0]
        row_b = summary.loc[summary["Model"] == "B"].iloc[0]
        row_c = summary.loc[summary["Model"] == "C"].iloc[0]

        assert row_a["Total Errors"] == 0
        assert row_a["False Positive"] == 0
        assert row_a["False Negative"] == 0

        assert row_b["False Positive"] == 1
        assert row_b["False Negative"] == 1
        assert row_b["Total Errors"] == 2

        assert row_c["False Positive"] == 2
        assert row_c["False Negative"] == 1
        assert row_c["Total Errors"] == 3

    def test_error_rate_matches_fraction(self):
        X, y, results = _make_dummy_data()
        df = build_error_dataframe(X, y, FEATURE_NAMES, results)
        summary = summarize_errors_per_model(df, results)

        n_total = len(y)
        for _, row in summary.iterrows():
            expected = row["Total Errors"] / n_total * 100
            assert row["Error Rate (%)"] == round(expected, 4)


class TestSharedErrorsAnalysis:
    def test_counts(self):
        X, y, results = _make_dummy_data()
        df = build_error_dataframe(X, y, FEATURE_NAMES, results)
        shared = shared_errors_analysis(df, results)

        assert shared["total_samples"] == len(y)
        # Any wrong = union of errors from B and C (A is perfect)
        # B errors: indices {2, 5}; C errors: {0, 1, 4}; union size = 5
        assert shared["any_wrong_count"] == 5
        # No sample is wrong across all 3 models (A is perfect everywhere)
        assert shared["all_wrong_count"] == 0
        # A has 0 unique errors (perfect)
        assert shared["only_one_wrong_counts"]["A"] == 0
        # B unique errors = indices wrong only in B
        # B wrong at {2, 5}; C wrong at {0, 1, 4}; no overlap
        # so B has 2 unique errors, C has 3 unique errors
        assert shared["only_one_wrong_counts"]["B"] == 2
        assert shared["only_one_wrong_counts"]["C"] == 3

    def test_all_wrong_df_structure(self):
        """If all models share the same wrong sample, it shows up in all_wrong_df."""
        X = np.array([[0, 10, 70.0], [1, 20, 80.0]])
        y = np.array([0, 1])
        # All three models wrong on index 1, correct on index 0.
        results = {
            "A": {"y_pred": np.array([0, 0])},
            "B": {"y_pred": np.array([0, 0])},
            "C": {"y_pred": np.array([0, 0])},
        }
        df = build_error_dataframe(X, y, FEATURE_NAMES, results)
        shared = shared_errors_analysis(df, results)

        assert shared["all_wrong_count"] == 1
        assert isinstance(shared["all_wrong_df"], pd.DataFrame)
        assert len(shared["all_wrong_df"]) == 1
        # The single hard sample has y_true = 1
        assert shared["all_wrong_df"]["y_true"].iloc[0] == 1
