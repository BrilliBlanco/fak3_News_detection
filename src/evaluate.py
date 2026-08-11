"""
Deep evaluation of already-trained models.

`train.py` prints accuracy. That is the least interesting thing you can say
about a classifier. This module produces the things a report actually needs:

  - ROC and precision-recall curves for all three models on shared axes
  - Calibration (reliability) curves - is "90% confident" actually right 90%
    of the time? Matters here, because the UI shows a confidence number
  - Confusion matrices side by side
  - A threshold sweep, so you can pick an operating point instead of
    accepting 0.5 by default
  - Error analysis: the mistakes the model was *most confident* about, which
    is where you learn what it actually keyed on
  - Cross-validated F1 with standard deviation, so single-split luck is visible

It rebuilds the exact train/test split `train.py` used by reading
`models/run_metadata.json`, so the numbers here are directly comparable.

Usage:
    python src/evaluate.py
    python src/evaluate.py --learning-curve --cv 5
"""

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (auc, average_precision_score, brier_score_loss,
                             classification_report, confusion_matrix,
                             precision_recall_curve, roc_curve)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     learning_curve, train_test_split)

from config import (FIGURES_DIR, MODEL_KEYS, MODEL_REGISTRY, MODELS_DIR,
                    REAL, REPORTS_DIR, RUN_META, TARGET_NAMES,
                    VECTORIZER_FILE, ensure_dirs, model_name, model_path)
from preprocessing import clean_text, load_data

COLORS = {"nb": "#e07a5f", "svm": "#3d5a80", "lr": "#81b29a"}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained models in depth.")
    p.add_argument("--models-dir", default=str(MODELS_DIR))
    p.add_argument("--data-dir", default=None)
    p.add_argument("--cv", type=int, default=0, help="Add N-fold cross-validated F1 (0 = skip)")
    p.add_argument("--learning-curve", action="store_true",
                   help="Plot accuracy vs training-set size (slow)")
    p.add_argument("--n-errors", type=int, default=15,
                   help="How many high-confidence mistakes to dump per model")
    return p.parse_args()


def rebuild_split(models_dir: Path, data_dir):
    """
    Recreate train.py's exact split from run_metadata.json so every number
    here lines up with models/metrics.json.
    """
    meta_path = models_dir / RUN_META
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Run `python src/train.py` first - evaluate.py "
            "needs the run metadata to reproduce the same test split."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    df = load_data(
        data_dir,
        strip_boilerplate=meta.get("strip_boilerplate", False),
        drop_duplicates=not meta.get("keep_duplicates", False),
    )
    df["clean_content"] = df["content"].map(clean_text)
    df = df[df["clean_content"].str.len() > 0].reset_index(drop=True)

    idx_train, idx_test = train_test_split(
        df.index, test_size=meta["test_size"],
        random_state=meta["random_state"], stratify=df["label"],
    )
    return df, idx_train, idx_test, meta


def positive_proba(model, X):
    """P(real) for each row, indexed by class value rather than column position."""
    if not hasattr(model, "predict_proba"):
        return None
    raw = model.predict_proba(X)
    return raw[:, list(model.classes_).index(REAL)]


def save_fig(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}"


# --- plots -----------------------------------------------------------------

def plot_roc_pr(y_true, scores: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    for key, p in scores.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        axes[0].plot(fpr, tpr, color=COLORS[key], linewidth=1.8,
                     label=f"{model_name(key)} (AUC={auc(fpr, tpr):.4f})")

        prec, rec, _ = precision_recall_curve(y_true, p)
        axes[1].plot(rec, prec, color=COLORS[key], linewidth=1.8,
                     label=f"{model_name(key)} (AP={average_precision_score(y_true, p):.4f})")

    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8, label="chance")
    axes[0].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curve")
    axes[1].set(xlabel="recall", ylabel="precision", title="Precision-Recall curve")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        ax.spines[["top", "right"]].set_visible(False)
    return save_fig(fig, "roc_pr_curves.png")


def plot_calibration(y_true, scores: dict):
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    brier = {}
    for key, p in scores.items():
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
        brier[key] = brier_score_loss(y_true, p)
        ax.plot(mean_pred, frac_pos, "o-", color=COLORS[key], linewidth=1.6,
                markersize=4, label=f"{model_name(key)} (Brier={brier[key]:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="perfectly calibrated")
    ax.set(xlabel="mean predicted P(real)", ylabel="observed fraction real",
           title="Calibration - is the confidence number honest?")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    return save_fig(fig, "calibration_curves.png"), brier


def plot_confusion_matrices(y_true, preds: dict):
    fig, axes = plt.subplots(1, len(preds), figsize=(4 * len(preds), 3.8))
    axes = np.atleast_1d(axes)
    for ax, (key, pred) in zip(axes, preds.items()):
        cm = confusion_matrix(y_true, pred)
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=11)
        ax.set_xticks([0, 1], TARGET_NAMES)
        ax.set_yticks([0, 1], TARGET_NAMES)
        ax.set(xlabel="predicted", ylabel="actual", title=model_name(key))
    return save_fig(fig, "confusion_matrices.png")


def threshold_sweep(y_true, scores: dict, best_key: str):
    """Precision/recall/F1 as the decision threshold moves off 0.5."""
    p = scores[best_key]
    grid = np.linspace(0.05, 0.95, 91)
    rows = []
    for t in grid:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 1.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"threshold": t, "precision": prec, "recall": rec, "f1": f1,
                     "accuracy": (pred == y_true).mean()})
    table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 4))
    for col, color in (("precision", "#3d5a80"), ("recall", "#e07a5f"), ("f1", "#2a9d8f")):
        ax.plot(table["threshold"], table[col], label=col, color=color, linewidth=1.7)
    best_t = table.loc[table["f1"].idxmax(), "threshold"]
    ax.axvline(0.5, color="gray", linestyle=":", label="default 0.5")
    ax.axvline(best_t, color="black", linestyle="--", linewidth=0.9,
               label=f"best F1 @ {best_t:.2f}")
    ax.set(xlabel="threshold on P(real)", ylabel="score",
           title=f"Threshold sweep - {model_name(best_key)}")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return table, best_t, save_fig(fig, "threshold_sweep.png")


def plot_learning_curve(estimator, X, y, cv_folds=3):
    sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv_folds, scoring="f1",
        train_sizes=np.linspace(0.05, 1.0, 8), n_jobs=1,
    )
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for scores, label, color in ((train_scores, "train", "#3d5a80"),
                                 (test_scores, "validation", "#e07a5f")):
        mean, std = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(sizes, mean, "o-", color=color, label=label, markersize=4)
        ax.fill_between(sizes, mean - std, mean + std, alpha=0.15, color=color)
    ax.set(xlabel="training examples", ylabel="F1",
           title="Learning curve - would more data help?")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return save_fig(fig, "learning_curve.png")


def error_analysis(df, idx_test, y_true, preds, scores, key, n=15):
    """The mistakes the model was most sure about."""
    wrong = preds[key] != y_true.to_numpy()
    if not wrong.any():
        return pd.DataFrame()

    conf = np.where(preds[key] == REAL, scores[key], 1 - scores[key])
    out = pd.DataFrame({
        "actual": [TARGET_NAMES[v] for v in y_true.to_numpy()[wrong]],
        "predicted": [TARGET_NAMES[v] for v in preds[key][wrong]],
        "confidence": conf[wrong],
        "subject": df.loc[idx_test, "subject"].to_numpy()[wrong],
        "title": df.loc[idx_test, "title"].to_numpy()[wrong],
        "excerpt": [t[:200].replace("\n", " ")
                    for t in df.loc[idx_test, "content"].to_numpy()[wrong]],
    }).sort_values("confidence", ascending=False).head(n)
    return out.reset_index(drop=True)


def md_table(df, floatfmt="{:.4f}"):
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: floatfmt.format(v) if pd.notna(v) else "")
    header = "| " + " | ".join([out.index.name or ""] + [str(c) for c in out.columns]) + " |"
    sep = "|" + "|".join(["---"] * (len(out.columns) + 1)) + "|"
    rows = ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |"
            for i, r in zip(out.index, out.values)]
    return "\n".join([header, sep] + rows)


def main():
    args = parse_args()
    ensure_dirs()
    models_dir = Path(args.models_dir)

    print("[1/6] Rebuilding the training run's test split...")
    df, idx_train, idx_test, meta = rebuild_split(models_dir, args.data_dir)
    y_test = df.loc[idx_test, "label"]
    print(f"    {len(idx_test):,} test articles "
          f"(boilerplate stripped: {meta.get('strip_boilerplate')})")

    vectorizer = joblib.load(models_dir / VECTORIZER_FILE)
    X_test = vectorizer.transform(df.loc[idx_test, "clean_content"])

    print("[2/6] Scoring models...")
    preds, scores, available = {}, {}, []
    for key in MODEL_KEYS:
        p = model_path(key, models_dir)
        if not p.exists():
            print(f"    skipping {model_name(key)} ({p.name} not found)")
            continue
        model = joblib.load(p)
        preds[key] = model.predict(X_test)
        prob = positive_proba(model, X_test)
        if prob is not None:
            scores[key] = prob
        available.append(key)

    if not available:
        raise SystemExit("No trained models found - run `python src/train.py` first.")

    print("[3/6] Curves + confusion matrices...")
    roc_fig = plot_roc_pr(y_test, scores) if scores else ""
    cal_fig, brier = plot_calibration(y_test, scores) if scores else ("", {})
    cm_fig = plot_confusion_matrices(y_test, preds)

    summary = pd.DataFrame({
        model_name(k): {
            "accuracy": (preds[k] == y_test.to_numpy()).mean(),
            "roc_auc": auc(*roc_curve(y_test, scores[k])[:2]) if k in scores else np.nan,
            "avg_precision": average_precision_score(y_test, scores[k]) if k in scores else np.nan,
            "brier": brier.get(k, np.nan),
        } for k in available
    }).T
    summary.index.name = "model"

    print("[4/6] Threshold sweep...")
    best_key = max(scores, key=lambda k: (preds[k] == y_test.to_numpy()).mean()) if scores else available[0]
    sweep_table, best_t, sweep_fig = (threshold_sweep(y_test, scores, best_key)
                                      if scores else (pd.DataFrame(), 0.5, ""))

    cv_lines = ""
    if args.cv:
        print(f"[4b/6] {args.cv}-fold cross-validation...")
        from train import build_models
        X_train = vectorizer.transform(df.loc[idx_train, "clean_content"])
        y_train = df.loc[idx_train, "label"]
        folds = StratifiedKFold(n_splits=args.cv, shuffle=True,
                                random_state=meta["random_state"])
        cv_rows = {}
        for key, est in build_models(meta["random_state"]).items():
            s = cross_val_score(est, X_train, y_train, cv=folds, scoring="f1")
            cv_rows[model_name(key)] = {"cv_f1_mean": s.mean(), "cv_f1_std": s.std()}
            print(f"    {model_name(key):<26} f1={s.mean():.4f} +/- {s.std():.4f}")
        cv_df = pd.DataFrame(cv_rows).T
        cv_df.index.name = "model"
        cv_lines = f"\n## Cross-validated F1 ({args.cv}-fold, training set)\n\n{md_table(cv_df)}\n"

    lc_fig = ""
    if args.learning_curve:
        print("[4c/6] Learning curve (slow)...")
        from sklearn.linear_model import LogisticRegression
        X_train = vectorizer.transform(df.loc[idx_train, "clean_content"])
        lc_fig = plot_learning_curve(
            LogisticRegression(max_iter=1000, random_state=meta["random_state"]),
            X_train, df.loc[idx_train, "label"], cv_folds=3,
        )

    print("[5/6] Error analysis...")
    error_sections = []
    for key in available:
        if key not in scores:
            continue
        errs = error_analysis(df, idx_test, y_test, preds, scores, key, args.n_errors)
        if errs.empty:
            error_sections.append(f"### {model_name(key)}\n\nNo misclassifications.\n")
            continue
        path = REPORTS_DIR / f"errors_{key}.csv"
        errs.to_csv(path, index=False)
        shown = errs[["actual", "predicted", "confidence", "title"]].head(8).copy()
        shown["title"] = shown["title"].str.slice(0, 70)
        error_sections.append(
            f"### {model_name(key)}\n\n{len(preds[key] != y_test.to_numpy())} test rows, "
            f"{int((preds[key] != y_test.to_numpy()).sum())} wrong. "
            f"Full list: `reports/errors_{key}.csv`\n\n"
            f"{md_table(shown.set_index('actual'))}\n"
        )

    print("[6/6] Writing report...")

    def embed(path, cap):
        return f"\n![{cap}]({path})\n" if path else ""

    report = f"""# Model Evaluation Report

Generated by `python src/evaluate.py`. Test split reproduced from
`models/run_metadata.json` ({len(idx_test):,} articles, `random_state=
{meta['random_state']}`), so these numbers match `models/metrics.json`.

Publisher boilerplate stripped during training: **{meta.get('strip_boilerplate')}**

## Headline comparison

{md_table(summary)}

`brier` is the Brier score - mean squared error of the probability estimate.
Lower is better; it penalises a model that is confidently wrong far harder
than accuracy does.

## Discrimination: ROC and precision-recall
{embed(roc_fig, "ROC and PR curves")}

## Calibration
{embed(cal_fig, "calibration curves")}
The app shows users a confidence percentage. This plot is the check on whether
that percentage means anything: points on the diagonal mean "90% confident" is
right about 90% of the time. Points below the diagonal mean the model is
overconfident, which is the dangerous direction for a misinformation tool.

## Confusion matrices
{embed(cm_fig, "confusion matrices")}

## Operating point
{embed(sweep_fig, "threshold sweep")}
Best F1 for {model_name(best_key)} is at threshold **{best_t:.2f}**
(default is 0.50). Raising the threshold trades recall for precision - useful
if a false "REAL" verdict is more costly than a false "FAKE" one.
{cv_lines}
{"## Learning curve" + embed(lc_fig, "learning curve") if lc_fig else ""}

## Error analysis - the confident mistakes

{"".join(error_sections)}

## Reading these results honestly

Accuracy near 1.0 on this dataset is a warning sign, not a triumph. `src/eda.py`
shows a single-keyword rule scoring about the same, which means most of the
apparent skill is publisher fingerprinting. Two follow-ups make the evaluation
credible:

- `python src/train.py --strip-boilerplate` then re-run this script - the drop
  is the leakage share.
- `python src/cross_dataset_eval.py --data-file data/WELFake_Dataset.csv` -
  out-of-distribution accuracy is the only number that reflects real-world use.
"""
    out = REPORTS_DIR / "evaluation_report.md"
    out.write_text(report, encoding="utf-8")

    print(f"\nWrote {out}")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
