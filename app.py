"""
Streamlit UI for the fake news detector.

Five tabs:
  Classify   - paste text / a URL / a screenshot, get a prediction *and* the
               evidence behind it, then flag whether it was right
  Batch      - upload a CSV, score every row, download the result
  Dataset    - the EDA findings, including the leakage audit
  Model      - metrics, model card, evaluation figures
  History    - everything the app has classified, with analytics and export

Run with:
    streamlit run app.py

Prerequisites:
    python src/setup_data.py    (extracts data/Fake.csv + True.csv)
    python src/train.py         (writes models/)
    python src/eda.py           (optional - fills the Dataset tab)
    python src/evaluate.py      (optional - fills the Model tab)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import json

import joblib
import pandas as pd
import streamlit as st

from config import (DB_PATH, FIGURES_DIR, LABEL_NAMES, METRICS_JSON,
                    MODEL_CARD, MODEL_KEYS, MODEL_REGISTRY, MODELS_DIR,
                    REPORTS_DIR, RUN_META, VECTORIZER_FILE, model_name,
                    model_path)
from db import (export_csv, fetch_predictions, log_prediction, record_feedback,
                summary_stats)
from explain import explain_prediction, global_top_features
from preprocessing import clean_text, text_stats

# extract.py pulls in optional scraping/OCR stacks. Import it lazily so a
# missing bs4 or Tesseract degrades the URL/Screenshot tabs instead of
# taking down the whole app.
try:
    from extract import extract_text_from_image, extract_text_from_url
    EXTRACT_ERROR = None
except ImportError as _exc:  # pragma: no cover - depends on local install
    extract_text_from_image = extract_text_from_url = None
    EXTRACT_ERROR = str(_exc)

DISPLAY_NAMES = {model_name(k): k for k in MODEL_KEYS}


# --- cached loaders --------------------------------------------------------

@st.cache_resource
def load_vectorizer():
    return joblib.load(MODELS_DIR / VECTORIZER_FILE)


@st.cache_resource
def load_model(key: str):
    return joblib.load(model_path(key))


@st.cache_data
def load_metrics():
    path = MODELS_DIR / METRICS_JSON
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@st.cache_data
def load_run_meta():
    path = MODELS_DIR / RUN_META
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def available_models() -> list[str]:
    return [k for k in MODEL_KEYS if model_path(k).exists()]


# --- classification UI -----------------------------------------------------

def show_result(text: str, model_key: str, source: str, origin: str = None):
    """Run one classification and render prediction + evidence + feedback."""
    vectorizer, model = load_vectorizer(), load_model(model_key)
    result = explain_prediction(text, model, vectorizer, top_n=12)

    if "error" in result:
        st.warning(f"{result['error']} Try a longer or different input.")
        return

    label, confidence = result["label"], result["confidence"]

    col1, col2 = st.columns([2, 1])
    with col1:
        if label == "REAL":
            st.success(f"Prediction: **{label}**")
        else:
            st.error(f"Prediction: **{label}**")
    with col2:
        st.metric("Confidence", f"{confidence * 100:.2f}%" if confidence else "n/a")

    if confidence:
        st.progress(min(float(confidence), 1.0))

    # Honest caveats, shown when they apply rather than buried in a footnote
    if confidence and confidence < 0.70:
        st.warning("Low confidence - this is close to a coin flip. Treat it as 'unclear', not as a verdict.")
    if result["n_terms_matched"] < 5:
        st.warning(
            f"Only {result['n_terms_matched']} features matched the model's vocabulary. "
            "The input is too short or too far from the training data for a meaningful call."
        )

    contrib = result.get("contributions")
    if contrib is not None and not contrib.empty:
        st.subheader("Why - the evidence behind this decision")
        st.caption(
            "Each term's effect = its TF-IDF value x the model's learned weight. "
            "These add up (with the bias) to the decision, so this is the actual "
            "arithmetic, not an approximation."
        )
        chart_df = contrib.set_index("term")["contribution"]
        st.bar_chart(chart_df, color="#3d5a80", horizontal=True)

        with st.expander("Full contribution table"):
            st.dataframe(
                contrib.style.format({"tfidf": "{:.3f}", "weight": "{:.3f}",
                                      "contribution": "{:.4f}"}),
                use_container_width=True,
            )
        c1, c2 = st.columns(2)
        c1.metric("Total evidence for REAL", f"{result['total_real_push']:.3f}")
        c2.metric("Total evidence for FAKE", f"{result['total_fake_push']:.3f}")

    with st.expander("Class probabilities"):
        if result["proba"]:
            st.write({"fake": f"{result['proba'][0] * 100:.2f}%",
                      "real": f"{result['proba'][1] * 100:.2f}%"})

    with st.expander("Writing-style signals in this text"):
        stats = text_stats(text)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Words", f"{stats['n_words']:,}")
        s2.metric("Exclamation marks", stats["exclamation_count"])
        s3.metric("ALL-CAPS words", stats["allcaps_words"])
        s4.metric("Avg word length", f"{stats['avg_word_len']:.1f}")
        st.caption(
            "Not used by the classifier - shown because heavy exclamation and "
            "ALL-CAPS use are classic sensationalism markers you can judge yourself."
        )

    with st.expander("Text used for classification"):
        st.text(text[:3000] + ("..." if len(text) > 3000 else ""))

    # Log + feedback loop
    top_terms = [[r.term, round(float(r.contribution), 5)] for r in contrib.itertuples()] \
        if contrib is not None and not contrib.empty else []
    row_id = log_prediction(text, result, model_key, source=source,
                            origin=origin, top_terms=top_terms)
    st.session_state["last_prediction_id"] = row_id

    st.divider()
    st.caption(f"Logged as prediction #{row_id}. Was this right?")
    f1, f2, _ = st.columns([1, 1, 4])
    if f1.button("Correct", key=f"ok_{row_id}"):
        record_feedback(row_id, correct=True)
        st.toast("Thanks - recorded as correct.")
    if f2.button("Wrong", key=f"no_{row_id}"):
        record_feedback(row_id, correct=False)
        st.toast("Thanks - recorded as incorrect. Exportable as training data.")


def tab_classify(model_key: str):
    sub_text, sub_url, sub_image = st.tabs(["Paste text", "URL", "Screenshot"])

    with sub_text:
        text = st.text_area("Paste a news headline or article", height=220, key="text_input")
        if st.button("Classify", type="primary", key="classify_text"):
            if not text.strip():
                st.warning("Please paste some text first.")
            else:
                show_result(text, model_key, source="text")

    with sub_url:
        if EXTRACT_ERROR:
            st.warning(f"URL input unavailable: {EXTRACT_ERROR}")
            st.code("pip install -r requirements.txt", language="bash")
        url = st.text_input("Article URL", placeholder="https://example.com/some-article",
                            disabled=bool(EXTRACT_ERROR))
        if st.button("Fetch & Classify", type="primary", key="classify_url",
                     disabled=bool(EXTRACT_ERROR)):
            if not url.strip():
                st.warning("Please enter a URL first.")
            else:
                with st.spinner("Fetching article..."):
                    try:
                        extracted = extract_text_from_url(url)
                    except Exception as exc:
                        st.error(f"Couldn't fetch or parse that URL: {exc}")
                        extracted = ""
                if extracted:
                    if len(extracted) < 100:
                        st.warning(
                            "Very little text extracted - this site probably blocks "
                            "scraping or renders with JavaScript. Paste the text manually instead."
                        )
                    show_result(extracted, model_key, source="url", origin=url)

    with sub_image:
        st.caption("Requires the Tesseract OCR engine installed on this machine - see README.")
        if EXTRACT_ERROR:
            st.warning(f"Screenshot input unavailable: {EXTRACT_ERROR}")
            st.code("pip install -r requirements.txt", language="bash")
        uploaded = st.file_uploader("Upload a screenshot", type=["png", "jpg", "jpeg"],
                                    disabled=bool(EXTRACT_ERROR))
        if uploaded and st.button("Extract & Classify", type="primary", key="classify_image"):
            with st.spinner("Running OCR..."):
                try:
                    extracted = extract_text_from_image(uploaded)
                except Exception as exc:
                    st.error(f"OCR failed: {exc}\n\nIs Tesseract installed and on your PATH?")
                    extracted = ""
            if extracted:
                if len(extracted) < 30:
                    st.warning("Very little text extracted - try a clearer, higher-resolution screenshot.")
                show_result(extracted, model_key, source="image", origin=uploaded.name)


# --- batch tab -------------------------------------------------------------

def tab_batch(model_key: str):
    st.subheader("Score a whole CSV")
    st.caption(
        "Upload a file with a text column. Every row is cleaned, vectorized and "
        "scored with the same pipeline as single predictions."
    )
    uploaded = st.file_uploader("CSV file", type=["csv"], key="batch_csv")
    if not uploaded:
        st.info("Waiting for a CSV. A `text` or `content` column is detected automatically.")
        return

    df = pd.read_csv(uploaded)
    st.write(f"{len(df):,} rows, {len(df.columns)} columns")
    st.dataframe(df.head(5), use_container_width=True)

    str_cols = [c for c in df.columns if df[c].dtype == object] or list(df.columns)
    default = next((c for c in ("content", "text", "article", "body") if c in df.columns),
                   str_cols[0])
    text_col = st.selectbox("Text column", str_cols, index=str_cols.index(default))
    label_col = st.selectbox("True-label column (optional, 0=fake 1=real)",
                             ["(none)"] + list(df.columns))

    if not st.button("Score all rows", type="primary"):
        return

    with st.spinner(f"Scoring {len(df):,} rows..."):
        vectorizer, model = load_vectorizer(), load_model(model_key)
        texts = df[text_col].fillna("").astype(str)
        if text_col != "title" and "title" in df.columns:
            texts = (df["title"].fillna("").astype(str) + " " + texts).str.strip()

        cleaned = texts.map(clean_text)
        usable = cleaned.str.len() > 0
        X = vectorizer.transform(cleaned)
        proba_real = model.predict_proba(X)[:, list(model.classes_).index(1)]

        out = df.copy()
        out["prediction"] = [LABEL_NAMES[int(p >= 0.5)] if u else "UNUSABLE"
                             for p, u in zip(proba_real, usable)]
        out["proba_real"] = proba_real.round(4)
        out["confidence"] = [max(p, 1 - p).round(4) for p in proba_real]
        out.loc[~usable, ["proba_real", "confidence"]] = None

    flagged = (out["prediction"] == "FAKE").sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Flagged FAKE", f"{flagged:,}", f"{flagged / max(len(out), 1):.1%}")
    c2.metric("Mean confidence", f"{out['confidence'].mean():.1%}")
    c3.metric("Low confidence (<70%)", int((out["confidence"] < 0.7).sum()))

    if label_col != "(none)":
        y = pd.to_numeric(df[label_col], errors="coerce")
        mask = y.notna() & usable
        pred_int = (out.loc[mask, "prediction"] == "REAL").astype(int)
        acc = (pred_int.to_numpy() == y[mask].astype(int).to_numpy()).mean()
        st.metric("Accuracy against the supplied labels", f"{acc:.2%}")

    st.dataframe(out.head(50), use_container_width=True)
    st.download_button("Download scored CSV", out.to_csv(index=False).encode("utf-8"),
                       file_name="scored.csv", mime="text/csv")


# --- dataset tab -----------------------------------------------------------

def tab_dataset():
    report = REPORTS_DIR / "eda_report.md"
    if not report.exists():
        st.info("No EDA report yet. Run `python src/eda.py` to generate one, then reload.")
        return

    leak_path = REPORTS_DIR / "eda_tables" / "leakage_audit.csv"
    if leak_path.exists():
        leak = pd.read_csv(leak_path, index_col=0)
        worst = leak.iloc[0]
        st.error(
            f"**Label leakage detected.** The marker `{worst.name}` appears in "
            f"{worst['in_real']:.1%} of real articles and {worst['in_fake']:.1%} of fake ones. "
            f"A single-rule classifier using it alone scores "
            f"**{worst['one_rule_accuracy']:.1%}** - as good as the trained models. "
            "Most of the headline accuracy is publisher fingerprinting, not fake-vs-real signal."
        )
        st.dataframe(leak.style.format({"in_fake": "{:.2%}", "in_real": "{:.2%}",
                                        "gap": "{:.2%}", "one_rule_accuracy": "{:.2%}"}),
                     use_container_width=True)

    st.subheader("Figures")
    figures = sorted(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []
    eda_figs = [f for f in figures if f.stem in {
        "class_balance", "subject_by_class", "length_distribution",
        "articles_over_time", "style_features", "top_terms", "leakage_audit"}]
    for fig in eda_figs:
        st.image(str(fig), caption=fig.stem.replace("_", " "), use_container_width=True)

    with st.expander("Full EDA report (markdown)"):
        st.markdown(report.read_text(encoding="utf-8"))


# --- model tab -------------------------------------------------------------

def tab_model(model_key: str):
    metrics = load_metrics()
    if not metrics:
        st.info("No metrics.json yet. Run `python src/train.py` to generate it.")
        return

    run, models, baselines = metrics["run"], metrics["models"], metrics.get("baselines", {})

    st.caption(f"Trained {run['trained_at']} - {run['n_train']:,} train / "
               f"{run['n_test']:,} test articles, {run['n_features']:,} features. "
               f"Boilerplate stripped: **{run['strip_boilerplate']}**")

    table = pd.DataFrame({
        model_name(k): {
            "accuracy": v["accuracy"], "precision": v["precision"],
            "recall": v["recall"], "f1": v["f1"],
            "roc_auc": v.get("roc_auc"), "train_seconds": v["train_seconds"],
        } for k, v in models.items() if k in MODEL_REGISTRY
    }).T
    st.dataframe(table.style.format("{:.4f}"), use_container_width=True)

    if baselines:
        st.subheader("Baselines on the same test set")
        b1, b2 = st.columns(2)
        b1.metric("Majority class", f"{baselines['majority_class']:.2%}")
        one_rule = baselines.get("one_rule_mentions_reuters")
        best_acc = max(v["accuracy"] for v in models.values())
        b2.metric('One rule: "mentions Reuters"', f"{one_rule:.2%}",
                  delta=f"{one_rule - best_acc:+.2%} vs best model")
        if one_rule and one_rule >= best_acc - 0.005:
            st.warning(
                "A one-line keyword rule matches the trained models. That is the "
                "clearest possible sign the dataset, not the model, is doing the work."
            )

    st.subheader("What the model leans on overall")
    try:
        top = global_top_features(load_model(model_key), load_vectorizer(), 15)
        c1, c2 = st.columns(2)
        c1.caption("Strongest REAL indicators")
        c1.dataframe(top[["real_term", "real_weight"]].style.format({"real_weight": "{:.3f}"}),
                     hide_index=True, use_container_width=True)
        c2.caption("Strongest FAKE indicators")
        c2.dataframe(top[["fake_term", "fake_weight"]].style.format({"fake_weight": "{:.3f}"}),
                     hide_index=True, use_container_width=True)
    except Exception as exc:
        st.info(f"Feature weights unavailable for this model: {exc}")

    eval_figs = [f for f in (sorted(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else [])
                 if f.stem in {"roc_pr_curves", "calibration_curves",
                               "confusion_matrices", "threshold_sweep", "learning_curve"}]
    if eval_figs:
        st.subheader("Evaluation figures")
        for fig in eval_figs:
            st.image(str(fig), caption=fig.stem.replace("_", " "), use_container_width=True)
    else:
        st.caption("Run `python src/evaluate.py` for ROC, calibration and threshold plots.")

    card = MODELS_DIR / MODEL_CARD
    if card.exists():
        with st.expander("Model card"):
            st.markdown(card.read_text(encoding="utf-8"))


# --- history tab -----------------------------------------------------------

def tab_history():
    stats = summary_stats()
    if not stats["total"]:
        st.info("Nothing classified yet. Predictions made in this app are logged here.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predictions", f"{stats['total']:,}")
    c2.metric("Mean confidence", f"{stats['avg_confidence']:.1%}"
              if stats["avg_confidence"] else "n/a")
    c3.metric("Low confidence", f"{stats['low_confidence_count']}",
              f"{stats['low_confidence_share']:.0%} of all")
    c4.metric("Human-reviewed", stats["reviewed"])

    if stats["observed_accuracy"] is not None:
        st.metric("Observed accuracy on reviewed predictions",
                  f"{stats['observed_accuracy']:.1%}")
        st.caption(
            "This is measured on real inputs users actually submitted, not on the "
            "held-out test split. When it sits well below the reported test accuracy, "
            "that gap is the distribution shift between the dataset and reality."
        )

    d1, d2 = st.columns(2)
    with d1:
        st.caption("By predicted label")
        st.bar_chart(pd.Series(stats["by_label"]))
    with d2:
        st.caption("By input source")
        st.bar_chart(pd.Series(stats["by_source"]))

    st.subheader("Recent predictions")
    limit = st.slider("How many to show", 10, 500, 50, step=10)
    df = fetch_predictions(limit=limit)
    view = df[["id", "created_at", "source", "model", "label", "confidence",
               "user_feedback", "text_preview"]].copy()
    view["confidence"] = view["confidence"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "")
    st.dataframe(view, use_container_width=True, hide_index=True)

    e1, e2 = st.columns(2)
    e1.download_button("Download the full log (CSV)",
                       fetch_predictions(limit=0).to_csv(index=False).encode("utf-8"),
                       file_name="prediction_log.csv", mime="text/csv")
    corrections = df[df["true_label"].notna()]
    e2.download_button(f"Download corrections ({len(corrections)} rows)",
                       corrections.to_csv(index=False).encode("utf-8"),
                       file_name="corrections.csv", mime="text/csv",
                       disabled=corrections.empty)
    st.caption(f"Store: `{DB_PATH}`")


# --- main ------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")
    st.title("📰 Fake News Detector")
    st.caption("TF-IDF + classical ML (Naive Bayes / SVM / Logistic Regression), "
               "trained on the ISOT dataset - with the evidence behind every call")

    if not (MODELS_DIR / VECTORIZER_FILE).exists():
        st.error("No trained models found.")
        st.code("python src/setup_data.py\npython src/train.py", language="bash")
        st.stop()

    keys = available_models()
    if not keys:
        st.error("The vectorizer exists but no model files do. Re-run `python src/train.py`.")
        st.stop()

    with st.sidebar:
        st.header("Settings")
        choice = st.selectbox("Model", [model_name(k) for k in keys])
        model_key = DISPLAY_NAMES[choice]

        meta = load_run_meta()
        if meta:
            st.caption(f"Trained {meta['trained_at']}")
            st.caption(f"{meta['n_train']:,} train / {meta['n_test']:,} test")
            if not meta.get("strip_boilerplate"):
                st.warning("This model was trained **with** publisher boilerplate. "
                           "Its accuracy is inflated by dataset leakage - see the Dataset tab.")

        st.divider()
        st.caption(
            "This model recognises writing style and vocabulary patterns from its "
            "training data. **It does not fact-check.** A well-written lie reads "
            "REAL to it; a badly-written truth reads FAKE."
        )

    tabs = st.tabs(["Classify", "Batch", "Dataset", "Model", "History"])
    with tabs[0]:
        tab_classify(model_key)
    with tabs[1]:
        tab_batch(model_key)
    with tabs[2]:
        tab_dataset()
    with tabs[3]:
        tab_model(model_key)
    with tabs[4]:
        tab_history()


if __name__ == "__main__":
    main()
