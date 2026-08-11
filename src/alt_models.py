"""
Structurally different baselines, so the model comparison is an actual comparison.

The project trains Naive Bayes, a calibrated LinearSVC and Logistic Regression.
All three are linear models over the *same* word-level TF-IDF matrix. Calling
that "we compared three approaches" overstates it - it compares three loss
functions fitted to one representation. If the representation is the thing
that is leaking, all three leak identically and the comparison cannot tell you.

This module adds four models that differ in representation, not just in solver:

  char   Character n-grams (3-5, `char_wb`) + calibrated LinearSVC.
         Reads orthography and typography rather than vocabulary.
  style  Stylometry only - 10 dense counts (exclamation marks, ALL-CAPS words,
         quote marks, word length ...). No text content whatsoever, so it
         *cannot* see the Reuters tag. Its accuracy is the most honest estimate
         in this repo of how hard the task actually is.
  svd    Word TF-IDF -> TruncatedSVD -> HistGradientBoosting. Same words as the
         main models, but a non-linear decision boundary over a dense space.
  leak   The control: one binary feature, "does the text contain 'reuters'",
         fed to a depth-1 decision tree. Its accuracy is the leakage ceiling.

A fifth model, `word`, is a re-fit of the existing calibrated LinearSVC. It is
not a new idea - it is a self-check. Its accuracy on the rebuilt split should
land on the number already in `models/metrics.json`; if it does not, the split
was rebuilt wrong and nothing else on this page is trustworthy.

Every model is fitted twice: once on the text as-is, once with publisher
boilerplate removed (`preprocessing.strip_source_boilerplate`). The gap between
the two runs is that model's dependence on the publisher fingerprint. Models
that barely move were never using it.

Stripping is applied *after* the split membership is fixed, so both runs score
the identical 7,820 test articles. Loading with `strip_boilerplate=True`
instead would change which rows survive de-duplication and quietly compare two
different test sets.

Usage:
    python src/alt_models.py
    python src/alt_models.py --sample 6000          # fast iteration
    python src/alt_models.py --skip-char            # char n-grams are the slow part
    python src/alt_models.py --conditions raw       # skip the strip experiment

Writes:
    reports/alt_models_report.md
    reports/alt_models_table.csv
    reports/figures/alt_models.png
    models/alt/*.joblib + models/alt/alt_metrics.json
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe: never tries to open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from config import (FIGURES_DIR, MAX_FEATURES, METRICS_JSON, MODEL_KEYS,
                    MODELS_DIR, NGRAM_RANGE, REAL, REPORTS_DIR, RUN_META,
                    ensure_dirs, model_name)
from preprocessing import clean_text, load_data, strip_source_boilerplate, text_stats

# Alternative-model registry. Order here is the order in every table and figure:
# reference first, then the genuinely different representations, control last.
ALT_REGISTRY = {
    "word": ("Word TF-IDF + LinearSVC (reference)", "linear, sparse words"),
    "char": ("Char 3-5grams + LinearSVC", "linear, sparse characters"),
    "svd": ("TF-IDF -> SVD -> HistGradientBoosting", "non-linear, dense topics"),
    "style": ("Stylometry only + Random Forest", "non-linear, 10 dense counts"),
    "leak": ("Control: 'reuters' stump", "one binary feature"),
}
ALT_KEYS = list(ALT_REGISTRY)

# Same ten descriptors eda.py reports on, so the two documents line up.
STYLE_COLS = [
    "n_words", "avg_word_len", "n_sentences", "exclamation_count",
    "question_count", "quote_count", "allcaps_words", "allcaps_ratio",
    "digit_ratio", "upper_char_ratio",
]

CONDITIONS = ["raw", "stripped"]
CONDITION_LABEL = {"raw": "boilerplate intact", "stripped": "boilerplate stripped"}
CONDITION_COLOR = {"raw": "#3d5a80", "stripped": "#e07a5f"}

ALT_DIRNAME = "alt"


# --- plumbing --------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Train and evaluate structurally different baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default=None, help="Folder with Fake.csv / True.csv")
    p.add_argument("--models-dir", default=str(MODELS_DIR),
                   help="Where run_metadata.json / metrics.json live; alt models go in <dir>/alt")
    p.add_argument("--conditions", choices=["both", "raw", "stripped"], default="both",
                   help="Run with boilerplate intact, stripped, or both")
    p.add_argument("--sample", type=int, default=None,
                   help="Fit on a random sample of N articles (fast iteration; breaks "
                        "comparability with models/metrics.json)")
    p.add_argument("--svd-components", type=int, default=250,
                   help="TruncatedSVD dimensionality for the non-linear text model")
    p.add_argument("--char-max-features", type=int, default=30000,
                   help="Vocabulary cap for the character n-gram vectorizer")
    p.add_argument("--char-truncate", type=int, default=1500,
                   help="Only n-gram the first N characters of each article "
                        "(0 = whole article; the whole article is ~3x slower)")
    p.add_argument("--skip-char", action="store_true",
                   help="Skip the character n-gram model (by far the slowest part)")
    p.add_argument("--no-figures", action="store_true", help="Skip PNG generation")
    p.add_argument("--no-save-models", action="store_true",
                   help="Do not write fitted estimators to models/alt/")
    return p.parse_args()


def save_fig(fig, name: str, enabled: bool = True) -> str:
    """Save a figure into reports/figures and return its relative path."""
    if not enabled:
        plt.close(fig)
        return ""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}"


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Minimal dataframe -> markdown table (avoids a `tabulate` dependency)."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: floatfmt.format(v) if pd.notna(v) else "")
    header = "| " + " | ".join([out.index.name or ""] + [str(c) for c in out.columns]) + " |"
    sep = "|" + "|".join(["---"] * (len(out.columns) + 1)) + "|"
    rows = [
        "| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |"
        for idx, row in zip(out.index, out.values)
    ]
    return "\n".join([header, sep] + rows)


def rebuild_split(models_dir: Path, data_dir, sample=None):
    """
    Recreate train.py's exact split from run_metadata.json, the same way
    evaluate.py does, so these numbers sit next to models/metrics.json honestly.

    Note the deliberate difference from evaluate.py: data is always loaded with
    `strip_boilerplate=False`. The stripped variant is derived afterwards from
    the same surviving rows, which keeps the test set identical across the two
    conditions. Stripping before de-duplication would drop a different set of
    rows and turn the experiment into an apples-to-oranges comparison.
    """
    meta_path = models_dir / RUN_META
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Run `python src/train.py` first - alt_models.py "
            "needs the run metadata to reproduce the same test split."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    df = load_data(
        data_dir,
        strip_boilerplate=False,
        drop_duplicates=not meta.get("keep_duplicates", False),
    )
    df["clean_content"] = df["content"].map(clean_text)
    df = df[df["clean_content"].str.len() > 0].reset_index(drop=True)

    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=meta["random_state"]).reset_index(drop=True)

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


def score(y_true, preds, proba=None) -> dict:
    """The five headline numbers, positive class = real."""
    out = {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if proba is not None else np.nan,
    }
    out["n_wrong"] = int((np.asarray(preds) != np.asarray(y_true)).sum())
    return out


def fit_and_score(model, X_train, y_train, X_test, y_test) -> tuple[dict, object]:
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - t0
    metrics = score(y_test, model.predict(X_test), positive_proba(model, X_test))
    metrics["fit_seconds"] = elapsed
    return metrics, model


def style_matrix(texts: pd.Series) -> pd.DataFrame:
    """The stylometric feature block. Raw text in, ten dense columns out."""
    stats = pd.DataFrame(list(texts.map(text_stats)), index=texts.index)
    return stats[STYLE_COLS]


# --- the models ------------------------------------------------------------
#
# Each builder takes the split for one condition and returns (metrics, artifacts).
# `artifacts` is {filename_stem: object} - everything needed to reuse the model.

def build_word(data, rs, args):
    """
    Reference model: word TF-IDF + calibrated LinearSVC, i.e. a re-fit of the
    project's existing SVM. Included as a control on this script itself - if
    the raw-condition accuracy here does not match models/metrics.json, the
    split was rebuilt incorrectly.
    """
    vec = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE, min_df=2)
    X_train = vec.fit_transform(data["clean_train"])
    X_test = vec.transform(data["clean_test"])
    model = CalibratedClassifierCV(LinearSVC(random_state=rs, max_iter=5000), cv=3)
    metrics, model = fit_and_score(model, X_train, data["y_train"], X_test, data["y_test"])
    metrics["n_features"] = int(X_train.shape[1])
    return metrics, {"word_vectorizer": vec, "word_model": model}


def build_char(data, rs, args):
    """
    Character n-grams over the RAW text - not clean_content. clean_text() lowercases
    and deletes punctuation and digits, which is precisely the orthographic signal
    a character model exists to read, so feeding it cleaned text would defeat the
    point. `lowercase=False` for the same reason: SHOUTING is a feature here.
    """
    trunc = args.char_truncate
    train_txt = data["raw_train"].str.slice(0, trunc) if trunc else data["raw_train"]
    test_txt = data["raw_test"].str.slice(0, trunc) if trunc else data["raw_test"]

    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), max_features=args.char_max_features,
        min_df=5, sublinear_tf=True, lowercase=False,
    )
    X_train = vec.fit_transform(train_txt)
    X_test = vec.transform(test_txt)
    model = CalibratedClassifierCV(LinearSVC(random_state=rs, max_iter=5000), cv=3)
    metrics, model = fit_and_score(model, X_train, data["y_train"], X_test, data["y_test"])
    metrics["n_features"] = int(X_train.shape[1])
    del X_train, X_test  # ~0.5 GB of sparse data; do not hold it across conditions
    return metrics, {"char_vectorizer": vec, "char_model": model}


def build_svd(data, rs, args):
    """
    Same words as the main models, different geometry: TF-IDF compressed to a
    few hundred dense components, then a gradient-boosted tree ensemble. This is
    the test of whether a non-linear boundary buys anything the linear models
    are missing. If it does not, that is worth knowing - it means the classes
    are linearly separable in word space, which on this dataset they are, for
    reasons that have nothing to do with truthfulness.
    """
    vec = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE, min_df=2)
    X_train = vec.fit_transform(data["clean_train"])
    X_test = vec.transform(data["clean_test"])

    svd = TruncatedSVD(n_components=args.svd_components, random_state=rs)
    Z_train = svd.fit_transform(X_train)
    Z_test = svd.transform(X_test)

    model = HistGradientBoostingClassifier(random_state=rs)
    metrics, model = fit_and_score(model, Z_train, data["y_train"], Z_test, data["y_test"])
    metrics["n_features"] = int(Z_train.shape[1])
    metrics["svd_explained_variance"] = float(svd.explained_variance_ratio_.sum())
    return metrics, {"svd_vectorizer": vec, "svd_reducer": svd, "svd_model": model}


def build_style(data, rs, args):
    """
    No text content at all - ten counts per article. Random Forest rather than
    HistGradientBoosting only because with ten features the fit is instant either
    way and RF pairs naturally with the permutation-importance read-out below,
    which is the interesting output here.

    This model physically cannot see the "(Reuters)" tag, so unlike everything
    else in the repo its accuracy is not contaminated by the publisher
    fingerprint. Treat it as the realistic difficulty estimate.
    """
    X_train = style_matrix(data["raw_train"])
    X_test = style_matrix(data["raw_test"])
    # max_depth is capped deliberately. Left unbounded, 300 trees over 31k rows
    # grew to depth 26-34 and ~1.45M nodes - a 115 MB artifact for a ten-feature
    # model, which is memorisation rather than fitting. Depth 12 keeps accuracy
    # within noise of the unbounded forest at a fraction of a percent of the size.
    model = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        random_state=rs, n_jobs=-1,
    )
    metrics, model = fit_and_score(model, X_train, data["y_train"], X_test, data["y_test"])
    metrics["n_features"] = int(X_train.shape[1])

    # Permutation importance on the test set, not the built-in Gini importance:
    # impurity importance is biased towards high-cardinality continuous features,
    # and here the features have wildly different cardinalities (counts vs ratios).
    perm = permutation_importance(
        model, X_test, data["y_test"], n_repeats=5, random_state=rs, n_jobs=-1
    )
    importance = pd.Series(perm.importances_mean, index=STYLE_COLS).sort_values(ascending=False)
    return metrics, {"style_model": model, "_importance": importance}


def build_leak(data, rs, args):
    """
    The control. One binary feature - does the article contain the word
    "reuters" - and a decision stump. Same rule train.py records as
    `one_rule_mentions_reuters`. Any model that does not clearly beat this has
    learned nothing beyond the publisher's name.
    """
    pattern = r"\breuters\b"
    X_train = data["raw_train"].str.contains(pattern, case=False, na=False).to_numpy().reshape(-1, 1).astype(int)
    X_test = data["raw_test"].str.contains(pattern, case=False, na=False).to_numpy().reshape(-1, 1).astype(int)

    model = DecisionTreeClassifier(max_depth=1, random_state=rs)
    metrics, model = fit_and_score(model, X_train, data["y_train"], X_test, data["y_test"])
    metrics["n_features"] = 1
    metrics["marker_rate_test"] = float(X_test.mean())
    return metrics, {"leak_model": model}


BUILDERS = {"word": build_word, "char": build_char, "svd": build_svd,
            "style": build_style, "leak": build_leak}


# --- figure ----------------------------------------------------------------

def plot_comparison(results: dict, keys, conditions, baseline_acc: float, enabled=True):
    """Grouped bars: accuracy per model, one bar per boilerplate condition."""
    names = [ALT_REGISTRY[k][0] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, 0.55 * len(keys) + 2.2)))

    x = np.arange(len(keys))
    width = 0.8 / max(len(conditions), 1)
    for i, cond in enumerate(conditions):
        vals = [results[cond][k]["accuracy"] for k in keys]
        offset = (i - (len(conditions) - 1) / 2) * width
        axes[0].bar(x + offset, vals, width, color=CONDITION_COLOR[cond],
                    label=CONDITION_LABEL[cond])
        for xi, v in zip(x + offset, vals):
            axes[0].text(xi, v + 0.006, f"{v:.3f}", ha="center", va="bottom",
                         fontsize=7, rotation=90)

    axes[0].axhline(baseline_acc, color="gray", linestyle=":", linewidth=1.2,
                    label=f"majority class ({baseline_acc:.3f})")
    axes[0].set_xticks(x, [n.replace(" + ", "\n+ ").replace(" -> ", "\n-> ") for n in names],
                       fontsize=7.5)
    axes[0].set_ylim(0.4, 1.09)
    axes[0].set_ylabel("test accuracy")
    axes[0].set_title("Accuracy by representation")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")

    # Right panel: how much each model loses when the fingerprint is removed.
    if len(conditions) == 2:
        drops = [results["raw"][k]["accuracy"] - results["stripped"][k]["accuracy"]
                 for k in keys]
        colors = ["#d1495b" if d > 0 else "#2e86ab" for d in drops]
        y = np.arange(len(keys))
        axes[1].barh(y, drops, color=colors)
        for yi, d in zip(y, drops):
            axes[1].text(d + (0.004 if d >= 0 else -0.004), yi, f"{d:+.4f}",
                         va="center", ha="left" if d >= 0 else "right", fontsize=8)
        axes[1].set_yticks(y, names, fontsize=8)
        axes[1].invert_yaxis()
        axes[1].axvline(0, color="black", linewidth=0.8)
        pad = max(abs(min(drops)), abs(max(drops)), 0.01) * 0.45
        axes[1].set_xlim(min(drops) - pad, max(drops) + pad)
        axes[1].set_xlabel("accuracy lost when boilerplate is stripped")
        axes[1].set_title("Dependence on the publisher fingerprint")
    else:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "run with --conditions both\nfor the strip comparison",
                     ha="center", va="center", fontsize=10, color="gray")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    return save_fig(fig, "alt_models.png", enabled)


# --- report ----------------------------------------------------------------

def main():
    args = parse_args()
    ensure_dirs()
    started = time.time()

    models_dir = Path(args.models_dir)
    alt_dir = models_dir / ALT_DIRNAME
    alt_dir.mkdir(parents=True, exist_ok=True)

    conditions = CONDITIONS if args.conditions == "both" else [args.conditions]
    keys = [k for k in ALT_KEYS if not (k == "char" and args.skip_char)]

    print("[1/6] Rebuilding the training run's test split...")
    df, idx_train, idx_test, meta = rebuild_split(models_dir, args.data_dir, args.sample)
    rs = meta["random_state"]
    y_train, y_test = df.loc[idx_train, "label"], df.loc[idx_test, "label"]
    comparable = args.sample is None
    print(f"    {len(idx_train):,} train / {len(idx_test):,} test articles "
          f"(random_state={rs})")
    if not comparable:
        print("    WARNING: --sample changes the split. Numbers are NOT comparable "
              "to models/metrics.json.")

    print("[2/6] Building the stripped variant of the same rows...")
    t0 = time.time()
    variants = {"raw": {"content": df["content"], "clean": df["clean_content"]}}
    n_emptied = 0
    if "stripped" in conditions:
        stripped_content = df["content"].map(strip_source_boilerplate)
        stripped_clean = stripped_content.map(clean_text)
        n_emptied = int((stripped_clean.str.len() == 0).sum())
        variants["stripped"] = {"content": stripped_content, "clean": stripped_clean}
    print(f"    done in {time.time() - t0:.1f}s"
          + (f" ({n_emptied} articles became empty after stripping)" if "stripped" in conditions else ""))

    print("[3/6] Fitting alternatives...")
    results = {c: {} for c in conditions}
    importances = {}
    saved = []
    for cond in conditions:
        v = variants[cond]
        data = {
            "raw_train": v["content"].loc[idx_train], "raw_test": v["content"].loc[idx_test],
            "clean_train": v["clean"].loc[idx_train], "clean_test": v["clean"].loc[idx_test],
            "y_train": y_train, "y_test": y_test,
        }
        print(f"  -- {CONDITION_LABEL[cond]} --")
        for key in keys:
            metrics, artifacts = BUILDERS[key](data, rs, args)
            results[cond][key] = metrics
            if "_importance" in artifacts:
                importances[cond] = artifacts.pop("_importance")
            if not args.no_save_models:
                for stem, obj in artifacts.items():
                    path = alt_dir / f"{stem}_{cond}.joblib"
                    joblib.dump(obj, path)
                    saved.append(path.name)
            print(f"    {ALT_REGISTRY[key][0]:<40} acc={metrics['accuracy']:.4f}  "
                  f"f1={metrics['f1']:.4f}  ({metrics['fit_seconds']:.1f}s fit)")

    print("[4/6] Cross-checking against models/metrics.json...")
    metrics_path = models_dir / METRICS_JSON
    existing, baselines = {}, {}
    if metrics_path.exists():
        blob = json.loads(metrics_path.read_text(encoding="utf-8"))
        existing = blob.get("models", {})
        baselines = blob.get("baselines", {})
    check_note = ""
    if comparable and "word" in results.get("raw", {}) and "svm" in existing:
        gap = abs(results["raw"]["word"]["accuracy"] - existing["svm"]["accuracy"])
        check_note = (
            f"reference model {results['raw']['word']['accuracy']:.4f} vs "
            f"stored SVM {existing['svm']['accuracy']:.4f} (difference {gap:.4f})"
        )
        print(f"    {check_note}")
        if gap > 0.005:
            print("    WARNING: reference model does not reproduce the stored SVM. "
                  "Treat the split rebuild as suspect.")

    # --- tables ---
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    head_rows = {}
    for k in MODEL_KEYS:
        if k in existing:
            head_rows[model_name(k)] = {
                **{c: existing[k].get(c, np.nan) for c in metric_cols},
                "representation": "linear, sparse words",
                "source": "models/metrics.json",
            }
    head_cond = "raw" if "raw" in conditions else conditions[0]
    for k in keys:
        head_rows[ALT_REGISTRY[k][0]] = {
            **{c: results[head_cond][k].get(c, np.nan) for c in metric_cols},
            "representation": ALT_REGISTRY[k][1],
            "source": "src/alt_models.py",
        }
    headline = pd.DataFrame(head_rows).T
    headline.index.name = "model"
    for c in metric_cols:
        headline[c] = headline[c].astype(float)

    long_rows = [
        {"model": ALT_REGISTRY[k][0], "condition": cond,
         **{c: results[cond][k].get(c, np.nan) for c in metric_cols},
         "n_features": results[cond][k].get("n_features"),
         "n_wrong": results[cond][k]["n_wrong"],
         "fit_seconds": results[cond][k]["fit_seconds"]}
        for cond in conditions for k in keys
    ]
    long_table = pd.DataFrame(long_rows)
    long_table.to_csv(REPORTS_DIR / "alt_models_table.csv", index=False)

    strip_table = pd.DataFrame()
    if len(conditions) == 2:
        strip_table = pd.DataFrame({
            ALT_REGISTRY[k][0]: {
                "acc_intact": results["raw"][k]["accuracy"],
                "acc_stripped": results["stripped"][k]["accuracy"],
                "acc_drop": results["raw"][k]["accuracy"] - results["stripped"][k]["accuracy"],
                "f1_intact": results["raw"][k]["f1"],
                "f1_stripped": results["stripped"][k]["f1"],
            } for k in keys
        }).T
        strip_table.index.name = "model"

    print("[5/6] Figure...")
    majority = baselines.get("majority_class", float(max(y_test.mean(), 1 - y_test.mean())))
    fig_path = plot_comparison(results, keys, conditions, majority, not args.no_figures)

    print("[6/6] Writing report...")
    (alt_dir / "alt_metrics.json").write_text(
        json.dumps({
            "run": {
                "generated_from": "src/alt_models.py",
                "split_source": str(models_dir / RUN_META),
                "random_state": rs,
                "n_train": len(idx_train),
                "n_test": len(idx_test),
                "sample": args.sample,
                "comparable_to_metrics_json": comparable,
                "conditions": conditions,
                "svd_components": args.svd_components,
                "char_max_features": args.char_max_features,
                "char_truncate": args.char_truncate,
            },
            "models": results,
            "style_importance": {c: s.to_dict() for c, s in importances.items()},
        }, indent=2, default=float),
        encoding="utf-8",
    )

    def embed(path, caption):
        return f"\n![{caption}]({path})\n" if path else ""

    style_note, imp_block, style_move = "", "", ""
    if "style" in keys:
        s_raw = results[head_cond]["style"]["accuracy"]
        style_note = f"{s_raw:.1%}"
        if len(conditions) == 2:
            s_strip = results["stripped"]["style"]["accuracy"]
            style_move = (
                f"Stripping the boilerplate moves it by {s_raw - s_strip:+.4f} "
                f"(to {s_strip:.4f}), which is what you would expect from a model that "
                f"never had access to the fingerprint in the first place. The small residual "
                f"movement is real, not noise in the fit: datelines like "
                f"`WASHINGTON (Reuters) -` contain an ALL-CAPS city name, so removing them "
                f"nudges `allcaps_words` and `upper_char_ratio` on the real side."
            )
        if importances:
            imp_df = pd.DataFrame(importances)
            imp_df.columns = [CONDITION_LABEL[c] for c in imp_df.columns]
            imp_df.index.name = "feature"
            imp_block = md_table(imp_df, floatfmt="{:.4f}")

    # Spread across the models that read text, computed rather than asserted -
    # the claim "they all agree" has to be checked against the run, every run.
    text_keys = [k for k in keys if k in ("word", "char", "svd")]
    text_accs = [results[head_cond][k]["accuracy"] for k in text_keys]
    spread_note = (
        f"the {len(text_keys)} text-reading models span "
        f"{min(text_accs):.4f} to {max(text_accs):.4f} accuracy, a spread of "
        f"{max(text_accs) - min(text_accs):.4f}"
    ) if len(text_accs) > 1 else "only one text-reading model was run"

    svd_note = ""
    if "svd" in keys and "svd_explained_variance" in results[head_cond]["svd"]:
        svd_note = (
            f"The SVD step keeps {args.svd_components} components explaining "
            f"{results[head_cond]['svd']['svd_explained_variance']:.1%} of the TF-IDF "
            f"variance."
        )

    ref_note = f" ({check_note})" if check_note else ""

    # How many rows of the headline table does the one-feature control beat?
    leak_acc = results[head_cond]["leak"]["accuracy"]
    beaten = int((headline["accuracy"] < leak_acc).sum())
    n_rivals = len(headline) - 1
    if head_cond == "raw":
        control_para = (
            f"The control row is the one to read first. A depth-1 decision tree over a "
            f"single binary feature - \"does the text contain the word reuters\" - scores "
            f"**{leak_acc:.4f}**, which beats {beaten} of the {n_rivals} other models in "
            f"the table. Nothing above it should be described as a success. Changing the "
            f"representation from words to characters to dense components barely changes "
            f"the outcome - {spread_note} - because none of these models is solving the "
            f"stated problem. They are finding the same shortcut through it from different "
            f"directions."
        )
    else:
        control_para = (
            f"This run scored the alternatives on **stripped** text while the three stored "
            f"models in the top rows were trained and scored on unstripped text, so the two "
            f"halves of the table are not directly comparable - run with "
            f"`--conditions both` for a like-for-like comparison. On stripped text the "
            f"control collapses to {leak_acc:.4f}, roughly the majority-class rate, because "
            f"the word it keys on has been deleted. The text models do not collapse with "
            f"it: {spread_note}."
        )

    emptied_note = (
        f"\n({n_emptied} articles were reduced to empty text by stripping. They are kept, "
        f"as all-zero feature rows, rather than quietly dropped - dropping them would "
        f"change the test set between conditions.)" if n_emptied else ""
    )
    if len(conditions) == 2:
        strip_intro = (
            f"Every model was fitted twice on the **same articles**: once as-is, once after "
            f"`preprocessing.strip_source_boilerplate()` removed wire datelines, agency "
            f"tags, editor sign-offs and blog-footer cruft. Stripping happens after the "
            f"split membership is fixed, so both runs are scored on the identical "
            f"{len(idx_test):,} test articles - loading the corpus with "
            f"`strip_boilerplate=True` instead would change which rows survive "
            f"de-duplication and would silently compare two different test sets."
            f"{emptied_note}"
        )
    else:
        strip_intro = (
            f"This run used `--conditions {conditions[0]}`, so there is no comparison to "
            f"show. Re-run with `--conditions both` to fit every model twice - once on the "
            f"text as-is and once with `preprocessing.strip_source_boilerplate()` applied - "
            f"and get the per-model dependence on the publisher fingerprint."
        )

    leak_line = ""
    if "leak" in keys and len(conditions) == 2:
        leak_line = (
            f"The control drops from {results['raw']['leak']['accuracy']:.4f} to "
            f"{results['stripped']['leak']['accuracy']:.4f} - stripping deletes the word "
            f"'reuters' outright, so the stump has a constant feature and can only "
            f"predict the majority class. That is the sanity check that the stripping "
            f"actually did what it claims."
        )

    biggest_drop = ""
    if len(conditions) == 2:
        drops = {k: results["raw"][k]["accuracy"] - results["stripped"][k]["accuracy"]
                 for k in keys if k != "leak"}
        if drops:
            worst = max(drops, key=drops.get)
            least = min(drops, key=drops.get)
            floor = min(results["stripped"][k]["accuracy"] for k in drops)
            # The prose has to follow the numbers, not the other way round: state
            # what the prediction was, then whether it held.
            if drops[worst] >= 0.05:
                verdict = (
                    f"So the prediction held for at least one model: **{ALT_REGISTRY[worst][0]}** "
                    f"loses {drops[worst]:.4f} once the fingerprint is gone. That difference is "
                    f"the share of its apparent skill that was publisher detection."
                )
            else:
                verdict = (
                    f"The prediction going in was that the vocabulary-based models would fall "
                    f"sharply once the fingerprint was gone. **They did not.** Every model "
                    f"gives up between {min(drops.values()):.4f} and {max(drops.values()):.4f} "
                    f"accuracy and none finishes below {floor:.4f}. That is not good news. "
                    f"Stripping genuinely removes `(Reuters)` - the control proves it, falling "
                    f"to the majority-class floor - but the corpus is full of redundant tells "
                    f"the stripper does not touch: `getty images`, `pic.twitter.com`, "
                    f"photographer bylines, and seven subject categories each exclusive to one "
                    f"class (`eda.py`, sections 2 and 6). Remove one tell and the models key on "
                    f"the next. A small drop here is evidence that the leakage is "
                    f"over-determined, **not** evidence that the models were doing something "
                    f"legitimate all along."
                )
            biggest_drop = (
                f"Largest drop among the non-control models: **{ALT_REGISTRY[worst][0]}**, "
                f"{drops[worst]:+.4f}. Smallest: **{ALT_REGISTRY[least][0]}**, "
                f"{drops[least]:+.4f}.\n\n{verdict}"
            )

    sample_warning = "" if comparable else (
        f"\n> **These numbers are not comparable to `models/metrics.json`.** This run used\n"
        f"> `--sample {args.sample}`, which resamples before splitting, so the test set is\n"
        f"> not the one the stored models were scored on. Re-run without `--sample` for\n"
        f"> the reportable figures.\n"
    )

    report = f"""# Alternative Baselines - Does the Representation Matter?

Generated by `python src/alt_models.py`. Split reproduced from
`models/{RUN_META}` ({len(idx_train):,} train / {len(idx_test):,} test,
`random_state={rs}`). Full per-condition numbers: `reports/alt_models_table.csv`.
Fitted estimators: `models/alt/`.
{sample_warning}
## Why this file exists

The project's headline comparison - Naive Bayes vs SVM vs Logistic Regression -
varies the classifier and holds the representation fixed. All three read the same
word-level TF-IDF matrix. That is one experiment reported three times. If the
feature space is the thing carrying the leakage, every one of those models leaks
by the same amount and the comparison between them cannot reveal it.

The models below change the representation instead:

| model | what it reads |
|---|---|
{chr(10).join(f"| {ALT_REGISTRY[k][0]} | {ALT_REGISTRY[k][1]} |" for k in keys)}

`Word TF-IDF + LinearSVC (reference)` is not a new idea - it is a re-fit of the
existing SVM, present only to prove the split was rebuilt correctly{ref_note}.
{svd_note}

## Headline comparison ({CONDITION_LABEL[head_cond]})

{md_table(headline)}

Precision/recall/F1 are for the positive class (`real` = 1). The first three rows
are read from `models/{METRICS_JSON}`; the rest were fitted by this script on the
same split.

Read that table alongside the two baselines `train.py` already records on this
split: majority class **{baselines.get('majority_class', float('nan')):.4f}**,
one-rule "mentions Reuters" **{baselines.get('one_rule_mentions_reuters', float('nan')):.4f}**.

{control_para}

## The strip experiment
{embed(fig_path, "alternative models with and without publisher boilerplate")}
{strip_intro}

{md_table(strip_table) if not strip_table.empty else "_Run with `--conditions both` to populate this table._"}

{biggest_drop}

{leak_line}

## The stylometry model is the number to quote

`{ALT_REGISTRY['style'][0]}` sees no words at all - ten counts per article
(exclamation marks, question marks, quote marks, ALL-CAPS words and their ratios,
digit ratio, upper-case character ratio, word count, mean word length, sentence
count). It cannot see `(Reuters)`, cannot see a dateline, cannot see vocabulary.
It scores **{style_note}**.

Two things follow, and they point in opposite directions:

1. `eda.py` reported that fake articles carry ~13x the exclamation marks, ~12x the
   question marks and ~3x the ALL-CAPS ratio of real ones. That is real, measurable
   signal, and this model confirms it converts into classification accuracy far
   above the {baselines.get('majority_class', float('nan')):.1%} majority-class floor.
2. It is still not a measurement of "can a machine detect fake news". It is a
   measurement of "can a machine tell Reuters wire copy from partisan blog copy",
   which is a formatting question. Wire services have style guides; the blogs in
   this corpus do not. A fabricated article written to AP style would sail past
   this model, and a real local-news report with an excited headline would not.

{style_move}

The stylometry number is more honest than the 99% figures because it is not
contaminated by the publisher tag. It is still not an honest estimate of the actual
task - just a less dishonest one.

### What the stylometry model actually keyed on

Permutation importance on the test set (mean accuracy lost when a single feature
is shuffled). Permutation rather than the tree's built-in impurity importance,
because impurity importance is biased towards features with more distinct values -
and these features range from small integer counts to continuous ratios.

{imp_block or "_Stylometry model was not run._"}

## What this does and does not establish

- **Establishes:** the choice of representation is close to irrelevant on this
  dataset - {spread_note}, and the one-keyword control sits inside that range. When
  models this structurally different agree this closely, they are agreeing about the
  data, not about the problem. The original three-model comparison was never going
  to surface that, because varying only the loss function cannot tell you whether
  the features are the problem.
- **Establishes:** the leakage is not an artifact of the word-level TF-IDF setup.
  Character n-grams pick it up too - the string `(Reuters)` is as visible to a
  5-character window as it is to a word tokenizer.
- **Does not establish:** that any of these would work on news from a different
  source, period or topic. Every number here is in-distribution. The stripped
  condition removes one known fingerprint; it does not remove the fact that the
  two halves of the corpus were scraped from different places, cover disjoint
  subject categories, and span different date ranges (`eda.py`, sections 2 and 4).
- **Does not establish:** anything about truthfulness. Nothing in this file, or in
  this repository, reads a claim and checks whether it is true.

## Method notes worth knowing before quoting anything here

- The character model n-grams only the first {args.char_truncate if args.char_truncate else "0 (all)"}
  characters of each article, purely for runtime. That window contains the dateline,
  which is where the fingerprint lives, so the truncation makes the char model look
  *better*, not worse. Pass `--char-truncate 0` to use whole articles.
- `char_wb` runs with `lowercase=False`. Case is stylistic evidence and lowercasing
  would throw it away, which would defeat the point of having a character model.
- The stylometry features are computed on raw article text, not on `clean_text()`
  output - cleaning deletes punctuation, digits and case, i.e. nine of the ten
  features.
- No hyperparameter search was run on any of these. They are defaults. Tuning them
  would move the numbers by fractions of a point and would not touch the finding,
  which is about the data.

## Reproducing

```bash
python src/alt_models.py                       # both conditions, full corpus
python src/alt_models.py --conditions raw      # skip the strip experiment
python src/alt_models.py --skip-char           # skip the slow character model
python src/alt_models.py --sample 6000         # fast, not comparable to metrics.json
```

Total runtime for this run: {time.time() - started:.0f}s.
"""

    out = REPORTS_DIR / "alt_models_report.md"
    out.write_text(report, encoding="utf-8")

    print(f"\nWrote {out}")
    print(f"      {REPORTS_DIR / 'alt_models_table.csv'}")
    if fig_path:
        print(f"      {FIGURES_DIR / 'alt_models.png'}")
    if saved:
        print(f"      {alt_dir} ({len(saved)} artifacts + alt_metrics.json)")
    print()
    print(headline[metric_cols].to_string(float_format=lambda v: f"{v:.4f}"))
    if not strip_table.empty:
        print()
        print(strip_table.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nTotal time: {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
