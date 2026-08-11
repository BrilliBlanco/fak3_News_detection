"""
Train the TF-IDF + Naive Bayes / SVM / Logistic Regression models.

Usage:
    python src/train.py
    python src/train.py --strip-boilerplate      # the honest-accuracy run
    python src/train.py --cv 5                   # add cross-validation
    python src/train.py --help

Produces (in --models-dir):
    tfidf_vectorizer.joblib
    naive_bayes_model.joblib / svm_model.joblib / logistic_regression_model.joblib
    metrics.txt          human-readable classification reports
    metrics.json         machine-readable metrics (used by evaluate.py and the app)
    run_metadata.json    every setting this run used, so the split is reproducible
    model_card.md        what was trained, on what, with which caveats

Why `--strip-boilerplate` exists
--------------------------------
~99% of ISOT's "real" articles carry a Reuters dateline and almost none of the
fake ones do, so a model can score ~99% by detecting one token. Training with
`--strip-boilerplate` removes those publisher fingerprints; the gap between the
two runs is the share of your accuracy that was leakage rather than signal.
Report both numbers.
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from config import (DATA_DIR, MAX_FEATURES, METRICS_JSON, METRICS_TXT,
                    MODEL_CARD, MODEL_REGISTRY, MODELS_DIR, RANDOM_STATE,
                    RUN_META, TARGET_NAMES, TEST_SIZE, VECTORIZER_FILE)
from preprocessing import clean_text, load_data


def parse_args():
    p = argparse.ArgumentParser(
        description="Train fake news detection models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default=str(DATA_DIR), help="Folder containing Fake.csv and True.csv")
    p.add_argument("--models-dir", default=str(MODELS_DIR), help="Where to save trained models")
    p.add_argument("--test-size", type=float, default=TEST_SIZE)
    p.add_argument("--random-state", type=int, default=RANDOM_STATE)
    p.add_argument("--max-features", type=int, default=MAX_FEATURES)
    p.add_argument("--ngram-max", type=int, default=2, help="Upper bound of the n-gram range")
    p.add_argument("--min-df", type=int, default=2, help="Ignore terms in fewer than N documents")
    p.add_argument("--strip-boilerplate", action="store_true",
                   help="Remove publisher fingerprints (Reuters datelines etc.) before training")
    p.add_argument("--keep-duplicates", action="store_true",
                   help="Do not drop duplicate articles (not recommended - inflates accuracy)")
    p.add_argument("--cv", type=int, default=0,
                   help="Run N-fold cross-validation on the training set (0 = skip)")
    return p.parse_args()


def build_models(random_state: int) -> dict:
    """Model key -> untrained estimator. Add a model here and it flows everywhere."""
    return {
        # Fast, strong text baseline. Gives calibrated-ish probabilities natively.
        "nb": MultinomialNB(),
        # LinearSVC has no predict_proba, so wrap it to get real probabilities
        # instead of raw decision-function distances.
        "svm": CalibratedClassifierCV(
            LinearSVC(random_state=random_state, max_iter=5000), cv=3
        ),
        "lr": LogisticRegression(max_iter=1000, random_state=random_state),
    }


def score_model(y_true, preds, proba=None) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
    }
    if proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, proba)
    return metrics


def baseline_scores(y_train, y_test, raw_test_text) -> dict:
    """
    Context for the headline number. A model is only impressive relative to
    the dumbest thing that also works.
    """
    dummy = DummyClassifier(strategy="most_frequent").fit(
        np.zeros((len(y_train), 1)), y_train
    )
    majority = accuracy_score(y_test, dummy.predict(np.zeros((len(y_test), 1))))

    # The one-rule classifier: "mentions Reuters -> real".
    one_rule = accuracy_score(
        y_test, raw_test_text.str.contains(r"\breuters\b", case=False, na=False).astype(int)
    )
    return {"majority_class": majority, "one_rule_mentions_reuters": one_rule}


def write_model_card(path: Path, meta: dict, results: dict, baselines: dict) -> None:
    rows = "\n".join(
        f"| {MODEL_REGISTRY[k][0]} | {v['accuracy']:.4f} | {v['precision']:.4f} | "
        f"{v['recall']:.4f} | {v['f1']:.4f} | "
        f"{v.get('roc_auc', float('nan')):.4f} | {v['train_seconds']:.1f}s |"
        for k, v in results.items()
    )
    stripped = meta["strip_boilerplate"]

    path.write_text(f"""# Model Card - Fake News Detector

Generated by `src/train.py` on {meta['trained_at']}.

## Intended use

Flags whether a news article's **writing style and vocabulary** resemble the
fake or the real half of its training corpus. It is a stylistic classifier,
**not a fact-checker** - it has no knowledge base and cannot verify a claim.
Suitable as a triage/prioritisation signal for human review. Not suitable as
a sole basis for removing, labelling, or penalising content.

## Training data

- Source: ISOT Fake and Real News Dataset (Fake.csv / True.csv)
- Articles used: {meta['n_articles']:,} ({meta['n_train']:,} train / {meta['n_test']:,} test,
  stratified, `random_state={meta['random_state']}`)
- Duplicates dropped: {"no" if meta['keep_duplicates'] else "yes"}
- Publisher boilerplate stripped: {"yes" if stripped else "**no**"}
- Field classified: article title + body, concatenated

## Preprocessing

Lowercase, strip URLs/HTML/punctuation/digits, drop English stopwords and
tokens shorter than 3 characters. TF-IDF with `max_features={meta['max_features']}`,
`ngram_range={tuple(meta['ngram_range'])}`, `min_df={meta['min_df']}`
({meta['n_features']:,} features actually retained).

## Results (held-out test set)

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Train time |
|---|---|---|---|---|---|---|
{rows}

Precision/recall/F1 are for the positive class (`real` = 1).

### Baselines on the same test set

| Baseline | Accuracy |
|---|---|
| Majority class | {baselines['majority_class']:.4f} |
| One rule: "mentions Reuters -> real" | {baselines['one_rule_mentions_reuters']:.4f} |

## Limitations

1. **Publisher leakage.** The real half of ISOT is almost entirely Reuters wire
   copy; the fake half is not. The one-rule baseline above quantifies how much of
   the task is solvable by spotting the publisher.
   {"This run stripped that boilerplate, so its numbers are the more honest ones."
    if stripped else
    "This run did **not** strip it - compare against a `--strip-boilerplate` run before quoting the accuracy."}
2. **Topic and period bound.** Training data is US/world politics from 2015-2018.
   Accuracy on other domains or later events is unmeasured and likely much lower.
3. **In-distribution evaluation only.** The test split shares the training
   distribution. Run `src/cross_dataset_eval.py` against a second corpus for a
   real generalisation estimate.
4. **Style, not truth.** A carefully-written falsehood reads "real" to this model;
   a sloppily-written true report reads "fake".

## Reproducing

```bash
python src/train.py {"--strip-boilerplate" if stripped else ""} \\
  --random-state {meta['random_state']} --max-features {meta['max_features']} --min-df {meta['min_df']}
```

Environment: Python {meta['python']}, scikit-learn {meta['sklearn']}, {meta['platform']}.
""", encoding="utf-8")


def main():
    args = parse_args()
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("[1/6] Loading data...")
    df = load_data(
        args.data_dir,
        strip_boilerplate=args.strip_boilerplate,
        drop_duplicates=not args.keep_duplicates,
        verbose=True,
    )

    print("[2/6] Cleaning text...")
    t0 = time.time()
    df["clean_content"] = df["content"].map(clean_text)
    df = df[df["clean_content"].str.len() > 0].reset_index(drop=True)
    print(f"    done in {time.time() - t0:.1f}s ({len(df):,} usable articles)")

    idx_train, idx_test = train_test_split(
        df.index, test_size=args.test_size,
        random_state=args.random_state, stratify=df["label"],
    )
    X_train, X_test = df.loc[idx_train, "clean_content"], df.loc[idx_test, "clean_content"]
    y_train, y_test = df.loc[idx_train, "label"], df.loc[idx_test, "label"]

    print("[3/6] Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"    {X_train_tfidf.shape[1]:,} features over {X_train_tfidf.shape[0]:,} training docs")

    print("[4/6] Training models...")
    models, results = build_models(args.random_state), {}
    for key, model in models.items():
        name = MODEL_REGISTRY[key][0]
        t0 = time.time()
        model.fit(X_train_tfidf, y_train)
        elapsed = time.time() - t0

        preds = model.predict(X_test_tfidf)
        proba = model.predict_proba(X_test_tfidf)[:, list(model.classes_).index(1)] \
            if hasattr(model, "predict_proba") else None

        results[key] = score_model(y_test, preds, proba)
        results[key]["train_seconds"] = elapsed
        results[key]["predictions"] = preds
        print(f"    {name:<26} acc={results[key]['accuracy']:.4f}  "
              f"f1={results[key]['f1']:.4f}  ({elapsed:.1f}s)")

    if args.cv:
        print(f"[4b/6] {args.cv}-fold cross-validation on the training set...")
        folds = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
        for key, model in build_models(args.random_state).items():
            scores = cross_val_score(model, X_train_tfidf, y_train, cv=folds, scoring="f1")
            results[key]["cv_f1_mean"] = float(scores.mean())
            results[key]["cv_f1_std"] = float(scores.std())
            print(f"    {MODEL_REGISTRY[key][0]:<26} f1={scores.mean():.4f} +/- {scores.std():.4f}")

    print("[5/6] Baselines...")
    baselines = baseline_scores(y_train, y_test, df.loc[idx_test, "content"])
    for name, acc in baselines.items():
        print(f"    {name:<26} acc={acc:.4f}")

    print("[6/6] Saving artifacts...")
    meta = {
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_articles": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": int(X_train_tfidf.shape[1]),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "max_features": args.max_features,
        "ngram_range": [1, args.ngram_max],
        "min_df": args.min_df,
        "strip_boilerplate": args.strip_boilerplate,
        "keep_duplicates": args.keep_duplicates,
        "cv_folds": args.cv,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
    }

    # Human-readable report
    report_lines = [f"Trained {meta['trained_at']} | {meta['command']}",
                    f"{meta['n_train']:,} train / {meta['n_test']:,} test | "
                    f"{meta['n_features']:,} features | "
                    f"boilerplate stripped: {args.strip_boilerplate}"]
    for key, res in results.items():
        report_lines.append(f"\n=== {MODEL_REGISTRY[key][0]} ===")
        report_lines.append(f"Accuracy: {res['accuracy']:.4f}")
        report_lines.append(classification_report(y_test, res["predictions"],
                                                  target_names=TARGET_NAMES))
        report_lines.append(f"Confusion matrix:\n{np.array(res['confusion_matrix'])}")
    report_lines.append("\n=== Baselines ===")
    for name, acc in baselines.items():
        report_lines.append(f"{name}: {acc:.4f}")
    (models_dir / METRICS_TXT).write_text("\n".join(report_lines), encoding="utf-8")

    # Machine-readable metrics (evaluate.py and the Streamlit app read this)
    json_results = {k: {kk: vv for kk, vv in v.items() if kk != "predictions"}
                    for k, v in results.items()}
    (models_dir / METRICS_JSON).write_text(
        json.dumps({"run": meta, "models": json_results, "baselines": baselines},
                   indent=2, default=float),
        encoding="utf-8",
    )
    (models_dir / RUN_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    joblib.dump(vectorizer, models_dir / VECTORIZER_FILE)
    for key, model in models.items():
        joblib.dump(model, models_dir / MODEL_REGISTRY[key][1])

    write_model_card(models_dir / MODEL_CARD, meta, json_results, baselines)

    print(f"\nSaved models, {METRICS_JSON}, {RUN_META} and {MODEL_CARD} "
          f"to {models_dir.resolve()}")
    print(f"Total time: {time.time() - started:.1f}s")

    best = max(results, key=lambda k: results[k]["f1"])
    print(f"Best by F1: {MODEL_REGISTRY[best][0]} ({results[best]['f1']:.4f})")
    if not args.strip_boilerplate:
        print("\nNote: publisher boilerplate was NOT stripped. Re-run with "
              "--strip-boilerplate and compare - the difference is leakage.")


if __name__ == "__main__":
    main()
