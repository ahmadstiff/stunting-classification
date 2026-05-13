"""Unit tests for src.statistical_tests (McNemar's test)."""

from __future__ import annotations

import numpy as np
import pytest

from src.statistical_tests import (
    mcnemar_contingency_table,
    mcnemar_test,
    run_pairwise_mcnemar,
)


class TestMcnemarContingencyTable:
    def test_all_correct(self):
        """When both models match ground truth perfectly, only n00 should be populated."""
        y_true = np.array([0, 1, 0, 1])
        y_a = np.array([0, 1, 0, 1])
        y_b = np.array([0, 1, 0, 1])
        table = mcnemar_contingency_table(y_true, y_a, y_b)
        assert table.shape == (2, 2)
        assert table[0, 0] == 4
        assert table[0, 1] == 0
        assert table[1, 0] == 0
        assert table[1, 1] == 0

    def test_disagreement_counts(self):
        """b and c cells should match manually-counted disagreements."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_a = np.array([0, 0, 1, 0, 0, 1, 0, 1])  # wrong on index 3
        y_b = np.array([0, 0, 1, 1, 0, 1, 1, 1])  # wrong on index 6
        table = mcnemar_contingency_table(y_true, y_a, y_b)
        assert table[0, 1] == 1  # A correct, B wrong (index 6)
        assert table[1, 0] == 1  # A wrong, B correct (index 3)
        # n00 = both correct on the other 6 samples
        assert table[0, 0] == 6
        assert table[1, 1] == 0


class TestMcnemarTest:
    def test_identical_predictions(self):
        y_true = np.array([0, 1, 0, 1])
        y_same = np.array([0, 1, 0, 1])
        result = mcnemar_test(y_true, y_same, y_same)
        assert result["b"] == 0
        assert result["c"] == 0
        assert result["statistic"] == 0.0
        assert result["p_value"] == 1.0
        assert result["significant_0.05"] is False
        assert "identik" in result["interpretation"].lower()

    def test_small_disagreement_not_significant(self):
        """With only 1 vs 1 disagreements, p-value should be high."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_a = np.array([0, 0, 1, 0, 0, 1, 0, 1])
        y_b = np.array([0, 0, 1, 1, 0, 1, 1, 1])
        result = mcnemar_test(y_true, y_a, y_b)
        assert result["b"] == 1
        assert result["c"] == 1
        # With continuity correction: (|1-1|-1)^2 / 2 = 0.5
        assert result["statistic"] == pytest.approx(0.5, abs=1e-9)
        assert result["significant_0.05"] is False

    def test_large_imbalance_is_significant(self):
        """Many disagreements favouring one model should produce a significant result."""
        rng = np.random.default_rng(42)
        # 200 samples, model A perfect, model B wrong 50 times -> very significant
        y_true = rng.integers(0, 2, size=200)
        y_a = y_true.copy()
        y_b = y_true.copy()
        y_b[:50] = 1 - y_b[:50]
        result = mcnemar_test(y_true, y_a, y_b)
        assert result["significant_0.05"] is True
        assert result["significant_0.01"] is True
        assert result["p_value"] < 0.01

    def test_continuity_correction_toggle(self):
        """Without continuity correction, the statistic should be larger."""
        y_true = np.array([0] * 30 + [1] * 30)
        y_a = y_true.copy()
        y_b = y_true.copy()
        # Introduce 5 vs 15 disagreements
        y_a[:5] = 1 - y_a[:5]
        y_b[10:25] = 1 - y_b[10:25]
        with_corr = mcnemar_test(y_true, y_a, y_b, continuity_correction=True)
        without_corr = mcnemar_test(y_true, y_a, y_b, continuity_correction=False)
        assert without_corr["statistic"] > with_corr["statistic"]
        # p-value with correction should be >= without correction
        assert without_corr["p_value"] <= with_corr["p_value"]

    def test_return_keys(self):
        y_true = np.array([0, 1])
        result = mcnemar_test(y_true, [0, 0], [1, 1])
        expected_keys = {
            "statistic",
            "p_value",
            "table",
            "b",
            "c",
            "significant_0.05",
            "significant_0.01",
            "interpretation",
        }
        assert expected_keys.issubset(result.keys())


class TestRunPairwiseMcnemar:
    def test_produces_expected_pair_count(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        results = {
            "A": {"y_pred": np.array([0, 1, 0, 1, 0, 1, 0, 1])},
            "B": {"y_pred": np.array([0, 1, 0, 0, 0, 1, 0, 1])},
            "C": {"y_pred": np.array([1, 1, 0, 1, 0, 1, 0, 1])},
        }
        df = run_pairwise_mcnemar(results, y_true, save_csv=False)
        # C(3, 2) = 3 pairs
        assert len(df) == 3
        assert set(df.columns) >= {
            "Model A",
            "Model B",
            "b (A benar, B salah)",
            "c (A salah, B benar)",
            "Chi-squared",
            "p-value",
            "Signifikan (alpha=0.05)",
            "Signifikan (alpha=0.01)",
            "Interpretasi",
        }

    def test_handles_single_model_pair(self):
        y_true = np.array([0, 1, 0, 1])
        results = {
            "A": {"y_pred": np.array([0, 1, 0, 1])},
            "B": {"y_pred": np.array([0, 1, 0, 0])},
        }
        df = run_pairwise_mcnemar(results, y_true, save_csv=False)
        assert len(df) == 1
        row = df.iloc[0]
        assert row["Model A"] == "A"
        assert row["Model B"] == "B"
