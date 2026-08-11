"""
Explainability for the TF-IDF + linear-model pipeline.

The whole argument for using classical ML here instead of an LLM is that you
can point at the exact features behind a decision. This module is what makes
that claim true rather than aspirational.

All three trained models are linear over TF-IDF features, so a prediction
decomposes exactly:

    score = bias + sum over terms of ( tfidf_value(term) * weight(term) )

so each term's contribution is a real number you can rank and display -
not an approximation like SHAP or LIME.

Usage:
    python src/explain.py --text "Breaking: scientists confirm ..." --model lr
    python src/explain.py --global --model svm --top-n 25
    python src/explain.py --compare --text "..."       # all three models
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import (FAKE, LABEL_NAMES, MODEL_KEYS, MODELS_DIR, REAL,
                    VECTORIZER_FILE, model_name, model_path)
from preprocessing import clean_text


# --- weight extraction -----------------------------------------------------

def get_feature_weights(model) -> np.ndarray:
    """
    One weight per vocabulary term, oriented so that **positive = pushes REAL**
    and negative = pushes FAKE.

    Handles the three model types this project trains:
      - LogisticRegression / LinearSVC : coef_ row (binary -> single row)
      - MultinomialNB                  : difference of per-class log-probs
      - CalibratedClassifierCV         : mean coef_ over the CV fold estimators
    """
    # Calibrated wrapper (our SVM): average the underlying linear models
    if hasattr(model, "calibrated_classifiers_"):
        coefs = []
        for cc in model.calibrated_classifiers_:
            inner = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
            if inner is None or not hasattr(inner, "coef_"):
                continue
            coefs.append(np.asarray(inner.coef_).ravel())
        if not coefs:
            raise TypeError("Calibrated model exposes no linear sub-estimator to explain.")
        return np.mean(coefs, axis=0)

    # Multinomial Naive Bayes: log P(term|real) - log P(term|fake)
    if hasattr(model, "feature_log_prob_"):
        flp = model.feature_log_prob_
        classes = list(getattr(model, "classes_", [FAKE, REAL]))
        return flp[classes.index(REAL)] - flp[classes.index(FAKE)]

    # Plain linear model
    if hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        weights = coef.ravel() if coef.shape[0] == 1 else coef[1] - coef[0]
        # coef_ rows follow classes_ order; flip if class 1 isn't REAL
        classes = list(getattr(model, "classes_", [FAKE, REAL]))
        if coef.shape[0] == 1 and classes and classes[-1] != REAL:
            weights = -weights
        return weights

    raise TypeError(
        f"{type(model).__name__} is not a linear model - no per-feature weights "
        "to extract. Explanations only work for the NB/SVM/LR pipeline."
    )


def global_top_features(model, vectorizer, top_n: int = 20) -> pd.DataFrame:
    """The terms the *model as a whole* leans on hardest, per class."""
    weights = get_feature_weights(model)
    terms = np.asarray(vectorizer.get_feature_names_out())

    if len(weights) != len(terms):
        raise ValueError(
            f"Vectorizer has {len(terms)} features but the model has {len(weights)} "
            "weights - the model and vectorizer are out of sync. Retrain."
        )

    order = np.argsort(weights)
    real_idx, fake_idx = order[::-1][:top_n], order[:top_n]
    return pd.DataFrame({
        "real_term": terms[real_idx],
        "real_weight": weights[real_idx],
        "fake_term": terms[fake_idx],
        "fake_weight": weights[fake_idx],
    })


# --- per-prediction explanation -------------------------------------------

def explain_prediction(text: str, model, vectorizer, top_n: int = 10) -> dict:
    """
    Decompose one prediction into per-term contributions.

    Returns a dict with the prediction, probabilities, and a dataframe of the
    terms that mattered most (`contribution` = tfidf value * model weight;
    positive pushes REAL, negative pushes FAKE).
    """
    cleaned = clean_text(text)
    if not cleaned:
        return {"error": "No usable content after cleaning."}

    vec = vectorizer.transform([cleaned])
    pred = int(model.predict(vec)[0])
    classes = list(getattr(model, "classes_", [FAKE, REAL]))

    proba = None
    if hasattr(model, "predict_proba"):
        raw = model.predict_proba(vec)[0]
        # index by class value, never by position - classes_ order is not a promise
        proba = {int(c): float(raw[i]) for i, c in enumerate(classes)}

    terms = np.asarray(vectorizer.get_feature_names_out())
    present = vec.nonzero()[1]

    result = {
        "prediction": pred,
        "label": LABEL_NAMES[pred],
        "proba": proba,
        "confidence": proba[pred] if proba else None,
        # features include bigrams, so this can exceed the distinct-word count
        "n_terms_matched": int(len(present)),
        "n_words_kept": len(set(cleaned.split())),
        "cleaned_preview": cleaned[:300],
    }

    try:
        weights = get_feature_weights(model)
    except TypeError as exc:
        result["contributions"] = pd.DataFrame()
        result["explanation_error"] = str(exc)
        return result

    values = np.asarray(vec[0, present].todense()).ravel()
    contributions = values * weights[present]

    contrib_df = pd.DataFrame({
        "term": terms[present],
        "tfidf": values,
        "weight": weights[present],
        "contribution": contributions,
        "pushes": np.where(contributions >= 0, "REAL", "FAKE"),
    }).sort_values("contribution", key=np.abs, ascending=False).reset_index(drop=True)

    result["contributions"] = contrib_df.head(top_n)
    result["total_real_push"] = float(contributions[contributions > 0].sum())
    result["total_fake_push"] = float(-contributions[contributions < 0].sum())
    return result


def format_explanation(result: dict, width: int = 62) -> str:
    """Render `explain_prediction` output as a terminal-friendly block."""
    if "error" in result:
        return result["error"]

    lines = [
        f"Prediction: {result['label']}",
    ]
    if result["confidence"] is not None:
        lines.append(f"Confidence: {result['confidence'] * 100:.2f}%")
    lines.append(
        f"Matched {result['n_terms_matched']} vocabulary features "
        f"(from {result['n_words_kept']} distinct words after cleaning)"
    )

    if result.get("explanation_error"):
        lines.append(f"\n(no feature breakdown: {result['explanation_error']})")
        return "\n".join(lines)

    lines.append(f"\nEvidence pushing REAL: {result['total_real_push']:.4f}   "
                 f"pushing FAKE: {result['total_fake_push']:.4f}")
    lines.append("\nTop contributing terms:")
    lines.append(f"  {'term':<22}{'tfidf':>8}{'weight':>9}{'effect':>10}  pushes")
    lines.append("  " + "-" * width)
    for row in result["contributions"].itertuples():
        lines.append(
            f"  {row.term[:21]:<22}{row.tfidf:>8.3f}{row.weight:>9.3f}"
            f"{row.contribution:>10.4f}  {row.pushes}"
        )
    return "\n".join(lines)


# --- CLI -------------------------------------------------------------------

def load_artifacts(models_dir, model_key):
    models_dir = Path(models_dir)
    vec_path = models_dir / VECTORIZER_FILE
    mdl_path = model_path(model_key, models_dir)
    for p in (vec_path, mdl_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found - run `python src/train.py` first.")
    return joblib.load(vec_path), joblib.load(mdl_path)


def parse_args():
    p = argparse.ArgumentParser(description="Explain the model's decisions.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Text to explain")
    src.add_argument("--file", help="Path to a .txt file to explain")
    src.add_argument("--global", dest="global_view", action="store_true",
                     help="Show the model's strongest features overall, no input needed")
    p.add_argument("--model", choices=MODEL_KEYS, default="lr")
    p.add_argument("--compare", action="store_true", help="Explain with every trained model")
    p.add_argument("--models-dir", default=str(MODELS_DIR))
    p.add_argument("--top-n", type=int, default=12)
    return p.parse_args()


def main():
    args = parse_args()
    keys = MODEL_KEYS if args.compare else [args.model]

    for key in keys:
        try:
            vectorizer, model = load_artifacts(args.models_dir, key)
        except FileNotFoundError as exc:
            print(f"\n=== {model_name(key)} ===\n  skipped: {exc}")
            continue

        print(f"\n=== {model_name(key)} ===")

        if args.global_view:
            table = global_top_features(model, vectorizer, args.top_n)
            print(f"\nStrongest REAL-indicating and FAKE-indicating features:\n")
            print(table.to_string(index=False, float_format=lambda v: f"{v:>8.3f}"))
            continue

        text = args.text if args.text else Path(args.file).read_text(encoding="utf-8")
        print(format_explanation(explain_prediction(text, model, vectorizer, args.top_n)))


if __name__ == "__main__":
    main()
