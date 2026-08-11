"""
Classify news text as fake or real, with a confidence score.

Single item:
    python src/predict.py --text "Some headline or article body..."
    python src/predict.py --file article.txt --model svm --explain
    python src/predict.py --url https://example.com/story --json

Batch (this is the one you want for a dataset):
    python src/predict.py --csv inbox.csv --text-column content --out scored.csv

Everything can be logged to the local prediction store with --log
(inspect it later with `python src/db.py --stats`).
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config import (LABEL_NAMES, MODEL_KEYS, MODELS_DIR, REAL,
                    VECTORIZER_FILE, model_name, model_path)
from explain import explain_prediction, format_explanation
from preprocessing import clean_text


def parse_args():
    p = argparse.ArgumentParser(
        description="Predict fake/real news with a confidence score.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Raw text to classify")
    src.add_argument("--file", help="Path to a .txt file to classify")
    src.add_argument("--url", help="Fetch an article URL and classify it")
    src.add_argument("--csv", help="Batch mode: CSV of articles to score")

    p.add_argument("--models-dir", default=str(MODELS_DIR), help="Folder with saved models")
    p.add_argument("--model", choices=MODEL_KEYS + ["all"], default="svm",
                   help="Which model to use ('all' runs every model)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="P(real) above this is REAL. Raise it to be stricter about calling something real")

    # batch options
    p.add_argument("--text-column", default=None,
                   help="Column holding the article text (default: auto-detect)")
    p.add_argument("--out", default=None, help="Where to write batch results (default: <input>_scored.csv)")
    p.add_argument("--label-column", default=None,
                   help="Column of true labels (0/1). If given, batch mode reports accuracy")

    p.add_argument("--explain", action="store_true", help="Show the terms behind the decision")
    p.add_argument("--top-n", type=int, default=10, help="How many terms --explain shows")
    p.add_argument("--json", action="store_true", help="Emit JSON instead of formatted text")
    p.add_argument("--log", action="store_true", help="Record the prediction in reports/predictions.db")
    p.add_argument("--quiet", action="store_true", help="Batch mode: suppress the progress line")
    return p.parse_args()


def load_artifacts(models_dir: Path, keys):
    """Load the vectorizer once and every requested model."""
    vec_path = models_dir / VECTORIZER_FILE
    if not vec_path.exists():
        raise SystemExit(
            f"No trained models in {models_dir}.\n"
            "Run `python src/train.py` first (and `python src/setup_data.py` "
            "before that if data/ is empty)."
        )
    vectorizer = joblib.load(vec_path)

    models = {}
    for key in keys:
        path = model_path(key, models_dir)
        if not path.exists():
            print(f"warning: {path.name} not found, skipping {model_name(key)}", file=sys.stderr)
            continue
        models[key] = joblib.load(path)
    if not models:
        raise SystemExit("None of the requested models exist. Run `python src/train.py`.")
    return vectorizer, models


def read_input(args) -> tuple[str, str, str]:
    """Returns (text, source, origin)."""
    if args.text:
        return args.text, "text", None
    if args.file:
        path = Path(args.file)
        if not path.exists():
            raise SystemExit(f"File not found: {path}")
        return path.read_text(encoding="utf-8", errors="replace"), "file", str(path)
    if args.url:
        from extract import extract_text_from_url
        try:
            return extract_text_from_url(args.url), "url", args.url
        except Exception as exc:
            raise SystemExit(f"Couldn't fetch {args.url}: {exc}")
    raise SystemExit("No input given.")


def classify_one(text, model, vectorizer, threshold=0.5, top_n=10) -> dict:
    """Explainable single prediction, with an adjustable decision threshold."""
    result = explain_prediction(text, model, vectorizer, top_n=top_n)
    if "error" in result:
        return result

    # Re-decide at the requested threshold rather than trusting argmax
    if result["proba"] is not None and threshold != 0.5:
        pred = REAL if result["proba"][REAL] >= threshold else 1 - REAL
        result["prediction"] = pred
        result["label"] = LABEL_NAMES[pred]
        result["confidence"] = result["proba"][pred]
        result["threshold"] = threshold
    return result


# --- batch mode ------------------------------------------------------------

def pick_text_column(df: pd.DataFrame, explicit: str = None) -> str:
    if explicit:
        if explicit not in df.columns:
            raise SystemExit(f"Column {explicit!r} not in CSV. Found: {list(df.columns)}")
        return explicit
    for candidate in ("content", "text", "article", "body", "title"):
        if candidate in df.columns:
            return candidate
    # fall back to the widest string column
    str_cols = [c for c in df.columns if df[c].dtype == object]
    if not str_cols:
        raise SystemExit(f"No text column found in CSV. Columns: {list(df.columns)}")
    return max(str_cols, key=lambda c: df[c].astype(str).str.len().mean())


def run_batch(args, vectorizer, models):
    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"CSV not found: {path}")

    df = pd.read_csv(path)
    text_col = pick_text_column(df, args.text_column)
    if not args.quiet:
        print(f"Scoring {len(df):,} rows from {path.name} (text column: {text_col!r})")

    # Title + text is what the models were trained on - use both if present
    texts = df[text_col].fillna("").astype(str)
    if text_col != "title" and "title" in df.columns:
        texts = (df["title"].fillna("").astype(str) + " " + texts).str.strip()

    cleaned = texts.map(clean_text)
    usable = cleaned.str.len() > 0
    X = vectorizer.transform(cleaned)

    out = df.copy()
    for key, model in models.items():
        proba = model.predict_proba(X)[:, list(model.classes_).index(REAL)] \
            if hasattr(model, "predict_proba") else None
        if proba is not None:
            pred = (proba >= args.threshold).astype(int)
        else:
            pred = model.predict(X)
            proba = np.full(len(df), np.nan)

        suffix = "" if len(models) == 1 else f"_{key}"
        out[f"prediction{suffix}"] = np.where(usable, [LABEL_NAMES[p] for p in pred], "UNUSABLE")
        out[f"proba_real{suffix}"] = np.where(usable, proba, np.nan)
        out[f"confidence{suffix}"] = np.where(usable, np.maximum(proba, 1 - proba), np.nan)

        if not args.quiet:
            share_fake = (pred[usable.to_numpy()] == 0).mean() if usable.any() else 0
            print(f"  {model_name(key):<26} {share_fake:.1%} flagged FAKE, "
                  f"mean confidence {np.nanmean(np.maximum(proba, 1 - proba)[usable.to_numpy()]):.1%}")

        if args.label_column:
            if args.label_column not in df.columns:
                raise SystemExit(f"Label column {args.label_column!r} not in CSV.")
            y = pd.to_numeric(df[args.label_column], errors="coerce")
            mask = y.notna() & usable
            acc = (pred[mask.to_numpy()] == y[mask].astype(int).to_numpy()).mean()
            print(f"  {model_name(key):<26} accuracy on {int(mask.sum()):,} labelled rows: {acc:.4f}")

    n_unusable = int((~usable).sum())
    if n_unusable and not args.quiet:
        print(f"  {n_unusable} row(s) had no usable text after cleaning -> marked UNUSABLE")

    out_path = Path(args.out) if args.out else path.with_name(path.stem + "_scored.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    if args.log:
        from db import log_prediction
        key = next(iter(models))
        suffix = "" if len(models) == 1 else f"_{key}"
        for text, row in zip(texts, out.itertuples()):
            label = getattr(row, f"prediction{suffix}")
            if label == "UNUSABLE":
                continue
            proba_real = getattr(row, f"proba_real{suffix}")
            pred_int = 1 if label == "REAL" else 0
            log_prediction(
                text,
                {"prediction": pred_int, "label": label,
                 "proba": {0: 1 - proba_real, 1: proba_real},
                 "confidence": max(proba_real, 1 - proba_real)},
                key, source="csv", origin=str(path),
            )
        print(f"Logged {int(usable.sum())} predictions to the prediction store.")


# --- single mode -----------------------------------------------------------

def run_single(args, vectorizer, models):
    text, source, origin = read_input(args)
    if not text.strip():
        raise SystemExit("Input is empty.")

    payload = {"source": source, "origin": origin, "chars": len(text), "models": {}}

    for key, model in models.items():
        result = classify_one(text, model, vectorizer, args.threshold, args.top_n)

        if "error" in result:
            if args.json:
                payload["models"][key] = {"error": result["error"]}
            else:
                print(f"\n=== {model_name(key)} ===\n{result['error']}")
            continue

        top_terms = ([[r.term, round(float(r.contribution), 5)]
                      for r in result["contributions"].itertuples()]
                     if not result["contributions"].empty else [])

        if args.json:
            payload["models"][key] = {
                "prediction": result["label"],
                "confidence": result["confidence"],
                "proba_fake": result["proba"][0] if result["proba"] else None,
                "proba_real": result["proba"][1] if result["proba"] else None,
                "terms_matched": result["n_terms_matched"],
                "top_terms": top_terms,
            }
        else:
            if len(models) > 1:
                print(f"\n=== {model_name(key)} ===")
            if args.explain:
                print(format_explanation(result))
            else:
                print(f"Prediction: {result['label']}")
                if result["confidence"] is not None:
                    print(f"Confidence: {result['confidence'] * 100:.2f}%")
            if result["n_terms_matched"] < 5:
                print("  (warning: very few terms matched the model vocabulary - "
                      "this prediction is close to a guess)")

        if args.log:
            from db import log_prediction
            row_id = log_prediction(text, result, key, source=source,
                                    origin=origin, top_terms=top_terms)
            if not args.json:
                print(f"  logged as prediction #{row_id}")

    if args.json:
        print(json.dumps(payload, indent=2, default=float))


def main():
    args = parse_args()
    keys = MODEL_KEYS if args.model == "all" else [args.model]
    vectorizer, models = load_artifacts(Path(args.models_dir), keys)

    if args.csv:
        run_batch(args, vectorizer, models)
    else:
        run_single(args, vectorizer, models)


if __name__ == "__main__":
    main()
