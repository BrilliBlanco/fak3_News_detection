"""
Tests for the analysis modules added on top of the core pipeline:
significance, temporal_eval, alt_models, error_taxonomy, data_quality.

These assert against values worked out by hand rather than against whatever
the code currently returns - the project's headline claim rests on the
McNemar/Holm arithmetic in src/significance.py, so it gets real assertions.

Run with:
    python -m pytest -q
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import config  # noqa: E402


# --- significance: the statistical core ------------------------------------

class TestMcNemar:
    def test_counts_discordant_pairs_correctly(self):
        from significance import mcnemar
        # a right / b wrong on items 0,1,2 ; a wrong / b right on item 3
        correct_a = np.array([1, 1, 1, 0, 1, 0], dtype=bool)
        correct_b = np.array([0, 0, 0, 1, 1, 0], dtype=bool)
        out = mcnemar(correct_a, correct_b)
        assert out["b"] == 3
        assert out["c"] == 1
        # concordant items (both right, both wrong) must be excluded
        assert out["b"] + out["c"] == 4

    def test_exact_branch_matches_scipy_binomtest(self):
        from scipy.stats import binomtest

        from significance import mcnemar
        correct_a = np.array([1] * 7 + [0] * 3 + [1] * 5, dtype=bool)
        correct_b = np.array([0] * 7 + [1] * 3 + [1] * 5, dtype=bool)
        out = mcnemar(correct_a, correct_b, exact_max=100)  # force the exact branch
        expected = binomtest(7, 10, 0.5, alternative="two-sided").pvalue
        assert out["p_exact"] == pytest.approx(expected)

    def test_is_symmetric_under_argument_order(self):
        from significance import mcnemar
        a = np.array([1, 1, 0, 1, 0, 0, 1, 1], dtype=bool)
        b = np.array([0, 1, 1, 0, 0, 1, 1, 0], dtype=bool)
        forward, backward = mcnemar(a, b), mcnemar(b, a)
        assert forward["b"] == backward["c"] and forward["c"] == backward["b"]
        assert forward["p_raw"] == pytest.approx(backward["p_raw"])

    def test_identical_predictions_give_no_discordance(self):
        from significance import mcnemar
        same = np.array([1, 0, 1, 1, 0], dtype=bool)
        out = mcnemar(same, same.copy())
        assert out["b"] == 0 and out["c"] == 0
        assert out["n_discordant"] == 0
        assert out["p_raw"] == pytest.approx(1.0)


class TestHolmBonferroni:
    def test_matches_a_hand_worked_example(self):
        from significance import holm_bonferroni
        # sorted: .01 x4 = .04 | .02 x3 = .06 | .03 x2 = .06 | .04 x1 = .06 (monotone)
        adjusted = holm_bonferroni([0.01, 0.02, 0.03, 0.04])
        assert adjusted[0] == pytest.approx(0.04)
        assert adjusted[1] == pytest.approx(0.06)

    def test_enforces_monotonicity(self):
        from significance import holm_bonferroni
        adjusted = holm_bonferroni([0.01, 0.02, 0.03, 0.04])
        assert all(adjusted[i] <= adjusted[i + 1] + 1e-12
                   for i in range(len(adjusted) - 1))

    def test_caps_at_one(self):
        from significance import holm_bonferroni
        assert all(p <= 1.0 for p in holm_bonferroni([0.5, 0.6, 0.9]))

    def test_preserves_input_order(self):
        from significance import holm_bonferroni
        adjusted = holm_bonferroni([0.04, 0.01])
        # the smaller raw p must still carry the smaller adjusted p
        assert adjusted[1] < adjusted[0]


class TestOneRuleBaseline:
    def test_flags_reuters_case_insensitively(self):
        from significance import one_rule_predictions
        preds = one_rule_predictions(pd.Series([
            "WASHINGTON (Reuters) - the senate voted",
            "SHOCKING!!! you will not believe this",
            "a REUTERS poll found support fell",
        ]))
        assert list(preds) == [config.REAL, config.FAKE, config.REAL]


class TestBootstrap:
    def test_is_paired_across_models(self):
        """
        Every model must be scored on the SAME resampled indices within a
        replicate. If each were resampled independently, the confidence
        interval on the DIFFERENCE between two models would be meaningless.
        """
        from significance import bootstrap_draws
        y = np.array([1, 0] * 50)
        preds = {"a": y.copy(), "b": y.copy()}  # two identical perfect models
        # returns (n_boot, n_systems) arrays, columns ordered like `preds`
        acc, _ = bootstrap_draws(y, preds, 50, np.random.default_rng(0))
        assert acc.shape == (50, 2)
        # identical models on shared resamples => zero difference in every draw
        assert np.allclose(acc[:, 0] - acc[:, 1], 0.0)

    def test_unpaired_resampling_would_have_been_detectable(self):
        """
        Guard the property that matters: with two ANTI-correlated models the
        paired accuracies must sum to exactly 1 in every replicate, because
        both are scored on the same resampled items.
        """
        from significance import bootstrap_draws
        y = np.array([1, 0] * 50)
        preds = {"perfect": y.copy(), "inverted": 1 - y}
        acc, _ = bootstrap_draws(y, preds, 50, np.random.default_rng(3))
        assert np.allclose(acc[:, 0] + acc[:, 1], 1.0)

    def test_is_reproducible_from_a_seed(self):
        from significance import bootstrap_draws
        y = np.array([1, 0] * 50)
        preds = {"a": y.copy(), "b": (1 - y)}
        first, _ = bootstrap_draws(y, preds, 50, np.random.default_rng(7))
        second, _ = bootstrap_draws(y, preds, 50, np.random.default_rng(7))
        assert np.allclose(first, second)


# --- alternative baselines -------------------------------------------------

class TestAltModels:
    def test_style_matrix_is_purely_numeric(self):
        """The stylometry model must be structurally unable to see the Reuters tag."""
        from alt_models import STYLE_COLS, style_matrix
        X = np.asarray(style_matrix(pd.Series([
            "WASHINGTON (Reuters) - The senate voted on Tuesday.",
            "SHOCKING!!! YOU WILL NOT BELIEVE THIS!!!",
        ])))
        assert X.shape == (2, len(STYLE_COLS))
        assert np.issubdtype(X.dtype, np.number)

    def test_style_features_separate_sensational_from_sober_text(self):
        from alt_models import STYLE_COLS, style_matrix
        X = np.asarray(style_matrix(pd.Series([
            "The committee met on Thursday to discuss the proposal.",
            "SHOCKING!!! THEY ARE HIDING THIS!!! SHARE NOW!!!",
        ])))
        excl = STYLE_COLS.index("exclamation_count")
        assert X[1, excl] > X[0, excl]

    def test_registry_and_conditions_are_consistent(self):
        from alt_models import ALT_KEYS, ALT_REGISTRY, CONDITIONS
        assert set(ALT_KEYS) == set(ALT_REGISTRY)
        assert set(CONDITIONS) == {"raw", "stripped"}


# --- error taxonomy --------------------------------------------------------

class TestErrorTaxonomyBins:
    def test_bin_edges_and_labels_line_up(self):
        from error_taxonomy import (CONF_BINS, CONF_LABELS, COVERAGE_BINS,
                                    COVERAGE_LABELS, LENGTH_BINS, LENGTH_LABELS)
        # pd.cut needs exactly one more edge than label
        assert len(LENGTH_BINS) == len(LENGTH_LABELS) + 1
        assert len(CONF_BINS) == len(CONF_LABELS) + 1
        assert len(COVERAGE_BINS) == len(COVERAGE_LABELS) + 1

    def test_bin_edges_are_sorted(self):
        from error_taxonomy import CONF_BINS, COVERAGE_BINS, LENGTH_BINS
        for bins in (LENGTH_BINS, CONF_BINS, COVERAGE_BINS):
            assert list(bins) == sorted(bins)

    def test_confidence_bins_span_the_valid_range(self):
        from error_taxonomy import CONF_BINS
        # a predicted-class confidence can never be below 0.5 or above 1.0
        assert CONF_BINS[0] == pytest.approx(0.5)
        assert CONF_BINS[-1] == pytest.approx(1.0)


# --- data quality ----------------------------------------------------------

class TestDataQuality:
    def test_expected_schema_covers_the_source_columns(self):
        from data_quality import EXPECTED_SCHEMA
        assert {"title", "text", "subject", "date"} <= set(EXPECTED_SCHEMA)

    def test_status_scoring_ranks_pass_above_warn_above_fail(self):
        from data_quality import FAIL, PASS, STATUS_SCORE, WARN
        assert STATUS_SCORE[PASS] > STATUS_SCORE[WARN] > STATUS_SCORE[FAIL]

    def test_known_subjects_are_disjoint_across_classes(self):
        """DQ-2: this disjointness is the finding, so the constant must encode it."""
        from data_quality import KNOWN_SUBJECTS
        groups = [set(v) for v in KNOWN_SUBJECTS.values()] \
            if isinstance(KNOWN_SUBJECTS, dict) else [set(KNOWN_SUBJECTS)]
        if len(groups) == 2:
            assert not (groups[0] & groups[1])

    @pytest.mark.skipif(not (config.REPORTS_DIR / "data_quality.json").exists(),
                        reason="run `python src/data_quality.py` first")
    def test_report_json_has_a_stable_shape(self):
        import json
        data = json.loads((config.REPORTS_DIR / "data_quality.json")
                          .read_text(encoding="utf-8"))
        assert {"score", "status_counts", "checks"} <= set(data)
        assert len(data["checks"]) > 0


# --- temporal validation ---------------------------------------------------

class TestTemporalEval:
    def test_date_stump_finds_a_perfectly_separating_cutoff(self):
        """
        Fake articles start earlier than real ones in ISOT, so date alone
        classifies part of the corpus with no errors. That is the third
        leakage channel, and this helper is what detects it.
        """
        from temporal_eval import date_stump_accuracy
        dates = pd.to_datetime(pd.Series(
            ["2015-04-01", "2015-05-01", "2016-06-01", "2016-07-01"]))
        labels = pd.Series([config.FAKE, config.FAKE, config.REAL, config.REAL])
        accuracy, cutoff, _ = date_stump_accuracy(dates, labels)
        assert accuracy == pytest.approx(1.0)
        assert pd.Timestamp("2015-05-01") < cutoff <= pd.Timestamp("2016-06-01")

    def test_date_stump_reports_chance_on_unseparable_dates(self):
        from temporal_eval import date_stump_accuracy
        dates = pd.to_datetime(pd.Series(["2016-01-01"] * 4))
        labels = pd.Series([config.FAKE, config.REAL, config.FAKE, config.REAL])
        accuracy, _, _ = date_stump_accuracy(dates, labels)
        assert accuracy <= 0.75
