"""
Test suite for the fake news detection pipeline.

Run with:
    pytest -q

Tests that need trained models (models/*.joblib) skip themselves if you
haven't run `python src/train.py` yet, so a fresh clone still gets a green
run on the parts that don't depend on artifacts.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import config  # noqa: E402
from db import (clear, export_corrections, export_csv, fetch_predictions,  # noqa: E402
                init_db, log_prediction, record_feedback, summary_stats)
from preprocessing import (add_text_features, clean_text, load_data,  # noqa: E402
                           strip_source_boilerplate, text_stats)

MODELS_READY = (config.MODELS_DIR / config.VECTORIZER_FILE).exists()
DATA_READY = (config.DATA_DIR / "Fake.csv").exists() and (config.DATA_DIR / "True.csv").exists()

needs_models = pytest.mark.skipif(not MODELS_READY, reason="run `python src/train.py` first")
needs_data = pytest.mark.skipif(not DATA_READY, reason="run `python src/setup_data.py` first")


# --- clean_text ------------------------------------------------------------

class TestCleanText:
    def test_lowercases_and_strips_punctuation(self):
        assert clean_text("Hello, WORLD!!!") == "hello world"

    def test_removes_urls(self):
        out = clean_text("Read https://example.com/story and www.foo.org now")
        assert "http" not in out and "example" not in out and "www" not in out

    def test_removes_html(self):
        assert "div" not in clean_text("<div class='x'>breaking news</div>")

    def test_removes_digits(self):
        assert not any(c.isdigit() for c in clean_text("In 2017 the vote passed 51 to 49"))

    def test_drops_stopwords_and_short_tokens(self):
        out = clean_text("The president is in a very big house").split()
        assert "the" not in out and "is" not in out
        assert all(len(tok) > 2 for tok in out)

    @pytest.mark.parametrize("bad", ["", "   ", None, 123, "!!! ??? ...", "a b c"])
    def test_degenerate_inputs_return_empty_string(self, bad):
        assert clean_text(bad) == ""

    def test_is_deterministic(self):
        text = "Senate approves the appropriations bill on Tuesday"
        assert clean_text(text) == clean_text(text)


# --- leakage guard ---------------------------------------------------------

class TestStripBoilerplate:
    def test_removes_reuters_dateline(self):
        out = strip_source_boilerplate(
            "WASHINGTON (Reuters) - The Senate voted on Tuesday to approve the bill."
        )
        assert "Reuters" not in out
        assert "WASHINGTON" not in out
        assert "Senate voted" in out

    def test_removes_inline_agency_tag(self):
        assert "Reuters" not in strip_source_boilerplate("Officials said (Reuters) later that day.")

    def test_removes_bare_agency_word_case_insensitively(self):
        assert "reuters" not in strip_source_boilerplate("A Reuters poll found support fell.").lower()

    def test_removes_editor_signoff(self):
        out = strip_source_boilerplate("Markets fell. (Reporting by Jane Doe; Editing by John Roe)")
        assert "Jane Doe" not in out and "Markets fell" in out

    def test_removes_blog_footer(self):
        out = strip_source_boilerplate("He said it was over. Featured image via Getty Images.")
        assert "Featured image" not in out and "said it was over" in out

    def test_leaves_ordinary_prose_alone(self):
        text = "The committee met on Thursday to discuss the proposal."
        assert strip_source_boilerplate(text) == text

    @pytest.mark.parametrize("bad", ["", None, "   "])
    def test_degenerate_inputs(self, bad):
        assert strip_source_boilerplate(bad) == ""

    def test_clean_text_can_chain_the_guard(self):
        text = "LONDON (Reuters) - Prices rose sharply this quarter."
        assert "reuters" in clean_text(text, strip_boilerplate=False)
        assert "reuters" not in clean_text(text, strip_boilerplate=True)


# --- text statistics -------------------------------------------------------

class TestTextStats:
    def test_counts_words_and_markers(self):
        s = text_stats("THIS is HUGE!! Really?")
        assert s["n_words"] == 4
        assert s["exclamation_count"] == 2
        assert s["question_count"] == 1
        assert s["allcaps_words"] == 2

    def test_empty_text_gives_zeros_not_errors(self):
        s = text_stats("")
        assert s["n_words"] == 0
        assert s["avg_word_len"] == 0.0
        assert s["allcaps_ratio"] == 0.0

    def test_ratios_stay_in_range(self):
        s = text_stats("Mixed Case Text with 123 numbers and !!!")
        for key in ("allcaps_ratio", "digit_ratio", "upper_char_ratio"):
            assert 0.0 <= s[key] <= 1.0

    def test_add_text_features_adds_columns_without_dropping_rows(self):
        df = pd.DataFrame({"content": ["Hello there!", "ANOTHER ONE"]})
        out = add_text_features(df)
        assert len(out) == len(df)
        assert {"n_words", "exclamation_count", "allcaps_words"} <= set(out.columns)


# --- data loading ----------------------------------------------------------

@needs_data
class TestLoadData:
    def test_returns_expected_columns_and_labels(self):
        df = load_data()
        assert {"content", "label", "title", "body", "subject", "date"} <= set(df.columns)
        assert set(df["label"].unique()) == {config.FAKE, config.REAL}

    def test_no_empty_content_and_no_duplicates(self):
        df = load_data()
        assert (df["content"].str.len() > 0).all()
        assert not df["content"].duplicated().any()

    def test_keeping_duplicates_yields_more_rows(self):
        assert len(load_data(drop_duplicates=False)) > len(load_data(drop_duplicates=True))

    def test_missing_directory_raises_a_helpful_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="setup_data|Kaggle"):
            load_data(str(tmp_path))


# --- prediction store ------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test_predictions.db"
    init_db(path)
    return path


def _result(pred=0, conf=0.91):
    return {"prediction": pred, "label": config.LABEL_NAMES[pred],
            "confidence": conf, "proba": {0: conf if pred == 0 else 1 - conf,
                                          1: 1 - conf if pred == 0 else conf}}


class TestPredictionStore:
    def test_log_returns_increasing_ids(self, db):
        first = log_prediction("some article text", _result(), "lr", db_path=db)
        second = log_prediction("another article", _result(1), "svm", db_path=db)
        assert second > first

    def test_fetch_returns_newest_first(self, db):
        log_prediction("older", _result(), "lr", db_path=db)
        log_prediction("newer", _result(), "lr", db_path=db)
        rows = fetch_predictions(limit=10, db_path=db)
        assert rows.iloc[0]["text_preview"] == "newer"

    def test_filter_by_model(self, db):
        log_prediction("a", _result(), "lr", db_path=db)
        log_prediction("b", _result(), "svm", db_path=db)
        assert set(fetch_predictions(limit=10, model="svm", db_path=db)["model"]) == {"svm"}

    def test_summary_stats_on_empty_db(self, db):
        assert summary_stats(db)["total"] == 0

    def test_summary_stats_aggregates(self, db):
        log_prediction("a", _result(0, 0.95), "lr", source="text", db_path=db)
        log_prediction("b", _result(1, 0.60), "lr", source="url", db_path=db)
        stats = summary_stats(db)
        assert stats["total"] == 2
        assert stats["by_label"] == {"FAKE": 1, "REAL": 1}
        assert stats["low_confidence_count"] == 1
        assert stats["by_source"] == {"text": 1, "url": 1}

    def test_identical_inputs_are_detected_as_repeats(self, db):
        log_prediction("same text", _result(), "lr", db_path=db)
        log_prediction("same text", _result(), "svm", db_path=db)
        assert summary_stats(db)["repeated_inputs"] == 1

    def test_feedback_marks_correctness_and_true_label(self, db):
        row_id = log_prediction("x", _result(pred=0), "lr", db_path=db)
        assert record_feedback(row_id, true_label=1, db_path=db) is True
        row = fetch_predictions(limit=1, db_path=db).iloc[0]
        assert row["user_feedback"] == "incorrect"
        assert row["true_label"] == 1

    def test_feedback_on_unknown_id_returns_false(self, db):
        assert record_feedback(9999, correct=True, db_path=db) is False

    def test_feedback_without_a_verdict_raises(self, db):
        """Defaulting here would silently record 'incorrect' and flip true_label."""
        row_id = log_prediction("x", _result(), "lr", db_path=db)
        with pytest.raises(ValueError, match="correct=|true_label="):
            record_feedback(row_id, db_path=db)
        assert fetch_predictions(limit=1, db_path=db).iloc[0]["user_feedback"] is None

    def test_observed_accuracy_uses_reviewed_rows_only(self, db):
        good = log_prediction("a", _result(0), "lr", db_path=db)
        bad = log_prediction("b", _result(0), "lr", db_path=db)
        log_prediction("unreviewed", _result(0), "lr", db_path=db)
        record_feedback(good, correct=True, db_path=db)
        record_feedback(bad, correct=False, db_path=db)
        stats = summary_stats(db)
        assert stats["reviewed"] == 2
        assert stats["observed_accuracy"] == 0.5

    def test_exports_write_files(self, db, tmp_path):
        row_id = log_prediction("a", _result(), "lr", db_path=db)
        record_feedback(row_id, true_label=1, db_path=db)

        full = tmp_path / "all.csv"
        assert export_csv(full, db) == 1
        assert pd.read_csv(full).shape[0] == 1

        corrections = tmp_path / "corrections.csv"
        assert export_corrections(corrections, db) == 1
        assert set(pd.read_csv(corrections).columns) >= {"text", "label"}

    def test_clear_empties_the_store(self, db):
        log_prediction("a", _result(), "lr", db_path=db)
        assert clear(db) == 1
        assert summary_stats(db)["total"] == 0


# --- config integrity ------------------------------------------------------

class TestConfig:
    def test_registry_and_labels_agree(self):
        assert set(config.MODEL_KEYS) == set(config.MODEL_REGISTRY)
        assert config.LABEL_NAMES[config.FAKE] == "FAKE"
        assert config.LABEL_NAMES[config.REAL] == "REAL"
        # TARGET_NAMES must be ordered by label value for sklearn reports
        assert config.TARGET_NAMES[config.FAKE] == "fake"
        assert config.TARGET_NAMES[config.REAL] == "real"

    def test_model_path_rejects_unknown_keys(self):
        with pytest.raises(KeyError):
            config.model_path("nope")


# --- model-dependent tests -------------------------------------------------

@needs_models
class TestTrainedModels:
    @pytest.fixture(scope="class")
    def vectorizer(self):
        import joblib
        return joblib.load(config.MODELS_DIR / config.VECTORIZER_FILE)

    @pytest.mark.parametrize("key", ["nb", "svm", "lr"])
    def test_each_model_predicts_a_valid_label(self, key, vectorizer):
        import joblib
        path = config.model_path(key)
        if not path.exists():
            pytest.skip(f"{path.name} not trained")
        model = joblib.load(path)
        X = vectorizer.transform([clean_text("The Senate approved the bill on Tuesday.")])
        assert int(model.predict(X)[0]) in (config.FAKE, config.REAL)

    @pytest.mark.parametrize("key", ["nb", "svm", "lr"])
    def test_probabilities_sum_to_one(self, key, vectorizer):
        import joblib
        path = config.model_path(key)
        if not path.exists():
            pytest.skip(f"{path.name} not trained")
        model = joblib.load(path)
        X = vectorizer.transform([clean_text("Breaking news you will not believe.")])
        assert model.predict_proba(X)[0].sum() == pytest.approx(1.0)

    def test_vectorizer_and_model_feature_counts_match(self, vectorizer):
        import joblib
        from explain import get_feature_weights
        model = joblib.load(config.model_path("lr"))
        assert len(get_feature_weights(model)) == len(vectorizer.get_feature_names_out())

    def test_explanation_contributions_are_signed_consistently(self, vectorizer):
        import joblib
        from explain import explain_prediction
        model = joblib.load(config.model_path("lr"))
        result = explain_prediction(
            "BREAKING: you will not believe what the mainstream media is hiding",
            model, vectorizer, top_n=10,
        )
        contrib = result["contributions"]
        assert not contrib.empty
        # 'pushes' must agree with the sign of the contribution
        assert ((contrib["contribution"] >= 0) == (contrib["pushes"] == "REAL")).all()

    def test_sensational_text_scores_more_fake_than_wire_copy(self, vectorizer):
        import joblib
        from explain import explain_prediction
        model = joblib.load(config.model_path("lr"))
        sensational = explain_prediction(
            "SHOCKING!!! The mainstream media is HIDING this from you. Share before deleted!",
            model, vectorizer)
        sober = explain_prediction(
            "The Senate voted on Tuesday to approve the appropriations bill, "
            "according to two officials familiar with the matter.", model, vectorizer)
        assert sensational["proba"][config.FAKE] > sober["proba"][config.FAKE]

    def test_empty_input_is_reported_not_crashed(self, vectorizer):
        import joblib
        from explain import explain_prediction
        model = joblib.load(config.model_path("lr"))
        assert "error" in explain_prediction("!!! ???", model, vectorizer)
