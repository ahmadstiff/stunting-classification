"""Unit tests for src.tradeoff_analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.tradeoff_analysis import (
    build_tradeoff_table,
    compute_efficiency_scores,
    measure_inference_time,
)


class _DummyModel:
    """Minimal stub implementing a fast `.predict` for timing tests."""

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class TestMeasureInferenceTime:
    def test_returns_positive_durations(self):
        model = _DummyModel()
        X = np.zeros((100, 3))
        per_sample_ms, total_ms = measure_inference_time(model, X, n_repeats=3)
        assert per_sample_ms > 0
        assert total_ms > 0
        # per_sample_ms * n_samples should match total_ms
        assert per_sample_ms * len(X) == pytest.approx(total_ms, rel=1e-6)

    def test_handles_empty_input(self):
        model = _DummyModel()
        X = np.zeros((0, 3))
        # mean_ms_per_sample should not divide by zero
        per_sample_ms, total_ms = measure_inference_time(model, X, n_repeats=2)
        assert total_ms >= 0
        assert per_sample_ms >= 0


class TestComputeEfficiencyScores:
    def test_efficiency_formula(self):
        df = pd.DataFrame(
            {
                "Model": ["A", "B", "C"],
                "F1-Score": [0.99, 0.98, 0.97],
                "Training Time (s)": [0.1, 1.0, 0.5],
            }
        )
        ranked = compute_efficiency_scores(df, accuracy_col="F1-Score", time_col="Training Time (s)")

        # Expected scores: 0.99 / 1.1, 0.98 / 2.0, 0.97 / 1.5
        expected = {
            "A": 0.99 / 1.1,
            "B": 0.98 / 2.0,
            "C": 0.97 / 1.5,
        }
        for _, row in ranked.iterrows():
            assert row["Efficiency Score"] == pytest.approx(
                round(expected[row["Model"]], 6), abs=1e-6
            )

    def test_ranking_descending(self):
        df = pd.DataFrame(
            {
                "Model": ["Slow", "Fast", "Mid"],
                "F1-Score": [0.99, 0.95, 0.97],
                "Training Time (s)": [10.0, 0.1, 1.0],
            }
        )
        ranked = compute_efficiency_scores(df)
        # Fast should rank 1, because 0.95 / 1.1 ≈ 0.8636 vs Slow 0.99/11 ≈ 0.09
        assert ranked.iloc[0]["Model"] == "Fast"
        assert list(ranked["Ranking Efisiensi"]) == [1, 2, 3]

    def test_nan_cost_treated_as_zero(self):
        """NaN cost shouldn't crash the calculation (filled with 0)."""
        df = pd.DataFrame(
            {
                "Model": ["A", "B"],
                "F1-Score": [0.8, 0.9],
                "Training Time (s)": [np.nan, 1.0],
            }
        )
        ranked = compute_efficiency_scores(df)
        # A: 0.8 / (1 + 0) = 0.8  ; B: 0.9 / 2 = 0.45
        # A should rank 1
        assert ranked.iloc[0]["Model"] == "A"


class TestBuildTradeoffTable:
    def test_table_contains_expected_columns(self):
        model = _DummyModel()
        X_test = np.zeros((50, 3))
        results = {
            "A": {"accuracy": 0.99, "f1": 0.99, "training_time": 0.05},
            "B": {"accuracy": 0.95, "f1": 0.94, "training_time": 0.30},
        }
        models = {"A": model, "B": model}

        df = build_tradeoff_table(results, models, X_test, n_repeats=2)
        assert set(df.columns) >= {
            "Model",
            "Accuracy",
            "F1-Score",
            "Training Time (s)",
            "Inference Time Total (ms)",
            "Inference Time per Sampel (ms)",
        }
        assert len(df) == 2

    def test_missing_model_yields_null_inference(self):
        X_test = np.zeros((50, 3))
        results = {
            "Missing": {"accuracy": 0.9, "f1": 0.9, "training_time": 0.1},
        }
        models: dict = {}  # no actual model object

        df = build_tradeoff_table(results, models, X_test, n_repeats=2)
        assert len(df) == 1
        row = df.iloc[0]
        # Inference timings should be None since the model is missing
        assert row["Inference Time Total (ms)"] is None
        assert row["Inference Time per Sampel (ms)"] is None
