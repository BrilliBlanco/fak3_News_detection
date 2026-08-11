"""
Hyperparameter search and feature-count ablation.

`max_features=5000`, `ngram_range=(1, 2)` and `min_df=2` are inherited numbers
that nobody in this project ever justified. This module does two things about
that.

1. Searches them properly. A Pipeline of TfidfVectorizer + classifier is
   cross-validated on the **training split only** - the test split is never
   consulted while choosing anything - so the "best" parameters are not
   selected on the data they are afterwards scored on.

2. Ablates the vocabulary size from 50 terms up to 20,000. This is the part
   worth reading. If a 50-term vocabulary already separates the two classes,
   the task is not hard, and the 99% headline in `models/metrics.json` is not
   an achievement - it is another symptom of the leakage `src/eda.py`
   documented. The ablation therefore runs twice, once on the raw text and
   once with publisher boilerplate stripped, so the two curves sit on the
   same axes.

The search and the ablation answer different questions and neither replaces
the other: the search asks "which settings score best", the ablation asks
"how little does this dataset actually require". On a leaky dataset the
second question is the informative one.

Usage:
    python src/tune.py
    python src/tune.py --quick --model nb
    python src/tune.py --skip-search                 # ablation only
    python src/tune.py --full --sample 5000          # exhaustive grid, tiny data
    python src/tune.py --strip-boilerplate           # search on de-leaked text

Writes:
    reports/tuning_report.md
    reports/best_params.json
    reports/tuning_cv_results.csv
    reports/feature_ablation.csv
    reports/first_features.csv
    reports/figures/tuning_results.png
    reports/figures/feature_ablation.png
"""

import argparse
import json
import platform
import time
import warnings
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe: never tries to open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import (FAKE, FIGURES_DIR, MAX_FEATURES, MODEL_KEYS, NGRAM_RANGE,
                    RANDOM_STATE, REAL, REPORTS_DIR, TEST_SIZE, ensure_dirs,
                    model_name)
from preprocessing import clean_text, load_data, strip_source_boilerplate
from train import build_models

COLORS = {"nb": "#e07a5f", "svm": "#3d5a80", "lr": "#81b29a"}

# The settings train.py ships with. Everything is measured against these.
DEFAULT_MIN_DF = 2
DEFAULT_SUBLINEAR_TF = False
DEFAULT_TFIDF = {
    "max_features": MAX_FEATURES,
    "ngram_range": NGRAM_RANGE,
    "min_df": DEFAULT_MIN_DF,
    "sublinear_tf": DEFAULT_SUBLINEAR_TF,
}

DEFAULT_ABLATION_FEATURES = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]

# Vectorizer half of the grid, shared by every classifier.
# 'default' is deliberately small: the feature-count question is answered far
# more thoroughly by the ablation below, so the search only needs to bracket it.
VECTOR_GRIDS = {
    "quick": {
        "tfidf__max_features": [5000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [2],
        "tfidf__sublinear_tf": [False, True],
    },
    "default": {
        "tfidf__max_features": [5000, 20000],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "tfidf__min_df": [2, 5],
        "tfidf__sublinear_tf": [False, True],
    },
    "full": {
        "tfidf__max_features": [1000, 2000, 5000, 10000, 20000, 50000],
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "tfidf__min_df": [1, 2, 5],
        "tfidf__sublinear_tf": [False, True],
    },
}

# Classifier half. Each model contributes exactly one hyperparameter; the
# sklearn default for all three of these is the middle-ish value (alpha=1.0,
# C=1.0), which is what the shipped models use.
CLF_GRIDS = {
    "nb": {"quick": {"clf__alpha": [0.1, 1.0]},
           "default": {"clf__alpha": [0.1, 1.0]},
           "full": {"clf__alpha": [0.01, 0.1, 0.5, 1.0]}},
    "svm": {"quick": {"clf__C": [0.1, 1.0]},
            "default": {"clf__C": [0.1, 1.0]},
            "full": {"clf__C": [0.01, 0.1, 1.0, 10.0]}},
    "lr": {"quick": {"clf__C": [1.0, 10.0]},
           "default": {"clf__C": [1.0, 10.0]},
           "full": {"clf__C": [0.1, 1.0, 10.0, 100.0]}},
}


# --- plumbing --------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Hyperparameter search and feature-count ablation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default=None, help="Folder with Fake.csv / True.csv")
    p.add_argument("--model", choices=MODEL_KEYS + ["all"], default="all",
                   help="Which classifier(s) to search")
    size = p.add_mutually_exclusive_group()
    size.add_argument("--quick", action="store_true", help="Reduced grid (smoke test)")
    size.add_argument("--full", action="store_true",
                      help="Exhaustive grid - slow, expect tens of minutes")
    p.add_argument("--cv", type=int, default=3, help="Cross-validation folds")
    p.add_argument("--search-sample", type=int, default=12000,
                   help="Stratified subsample of the TRAINING split used for the "
                        "search (0 = use all of it; the full split takes ~4x longer)")
    p.add_argument("--max-candidates", type=int, default=None,
                   help="Cap on grid points; above it RandomizedSearchCV is used "
                        "instead of GridSearchCV. Default 40, or no cap with --full. "
                        "0 disables the cap.")
    p.add_argument("--sample", type=int, default=None,
                   help="Subsample the whole dataset before splitting (fast iteration)")
    p.add_argument("--strip-boilerplate", action="store_true",
                   help="Run the SEARCH on de-leaked text (the ablation does both anyway)")
    p.add_argument("--ablation-features", type=int, nargs="+",
                   default=DEFAULT_ABLATION_FEATURES,
                   help="max_features values to ablate over")
    p.add_argument("--ablation-variants", choices=["raw", "stripped", "both"],
                   default="both", help="Which text variants to ablate")
    p.add_argument("--top-terms", type=int, default=30,
                   help="How many of the first-selected vocabulary terms to list")
    p.add_argument("--n-boot", type=int, default=2000,
                   help="Bootstrap resamples for the noise floor on test F1")
    p.add_argument("--skip-search", action="store_true", help="Ablation only")
    p.add_argument("--skip-ablation", action="store_true", help="Search only")
    p.add_argument("--random-state", type=int, default=RANDOM_STATE)
    p.add_argument("--test-size", type=float, default=TEST_SIZE)
    p.add_argument("--jobs", type=int, default=-1, help="Parallel workers (-1 = all cores)")
    p.add_argument("--no-figures", action="store_true", help="Skip PNG generation")
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
    rows = ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |"
            for i, r in zip(out.index, out.values)]
    return "\n".join([header, sep] + rows)


def fmt_params(params: dict) -> str:
    """Render a parameter dict compactly, stripping the pipeline step prefixes."""
    return ", ".join(f"{k.split('__')[-1]}={v}" for k, v in sorted(params.items()))


# --- data ------------------------------------------------------------------

def prepare_data(args):
    """
    Load once, produce both text variants on *identical rows*, and split.

    Deliberately not `load_data(strip_boilerplate=True)`: that strips before
    de-duplicating, so it returns a slightly different row set and the two
    ablation curves would no longer be measured on the same articles. Here the
    stripped variant is derived from the surviving rows, so raw vs stripped is
    a paired comparison.
    """
    df = load_data(args.data_dir, strip_boilerplate=False, drop_duplicates=True)
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=args.random_state).reset_index(drop=True)

    df["clean_raw"] = df["content"].map(clean_text)
    df = df[df["clean_raw"].str.len() > 0].reset_index(drop=True)
    df["clean_stripped"] = df["content"].map(strip_source_boilerplate).map(clean_text)

    idx_train, idx_test = train_test_split(
        df.index, test_size=args.test_size,
        random_state=args.random_state, stratify=df["label"],
    )
    return df, idx_train, idx_test


def subsample_index(idx, labels, n, random_state):
    """Stratified subsample of an index, or the index itself if n is 0/too big."""
    if not n or n >= len(idx):
        return idx
    kept, _ = train_test_split(idx, train_size=n, random_state=random_state,
                               stratify=labels.loc[idx])
    return kept


# --- search ----------------------------------------------------------------

def search_space(key: str, level: str):
    """
    (estimator, parameter grid) for one model key.

    `svm` searches a bare LinearSVC rather than the CalibratedClassifierCV
    wrapper that train.py ships. Calibration fits three inner LinearSVCs per
    candidate, which would triple an already slow search while tuning exactly
    the same `C`. The C selected here is meant to be passed to the wrapped
    model; the wrapper changes the probability estimates, not which C fits the
    training data best.
    """
    estimators = {
        "nb": MultinomialNB(),
        "svm": LinearSVC(random_state=RANDOM_STATE, max_iter=5000),
        "lr": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    }
    grid = dict(VECTOR_GRIDS[level])
    grid.update(CLF_GRIDS[key][level])
    return estimators[key], grid


def grid_size(grid: dict) -> int:
    return int(np.prod([len(v) for v in grid.values()]))


def default_params(key: str) -> dict:
    """The configuration train.py currently uses, in pipeline-parameter form."""
    params = {f"tfidf__{k}": v for k, v in DEFAULT_TFIDF.items()}
    params["clf__alpha" if key == "nb" else "clf__C"] = 1.0
    return params


def run_search(key, level, X, y, folds, args):
    """One model's search. Returns (fitted search object, was_randomized)."""
    estimator, grid = search_space(key, level)
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", estimator)])

    n_points = grid_size(grid)
    cap = args.max_candidates
    randomized = bool(cap) and n_points > cap

    common = dict(scoring="f1", cv=folds, n_jobs=args.jobs, refit=False,
                  return_train_score=False)
    if randomized:
        search = RandomizedSearchCV(pipe, grid, n_iter=cap,
                                    random_state=args.random_state, **common)
        kind = f"RandomizedSearchCV({cap} of {n_points} points)"
    else:
        search = GridSearchCV(pipe, grid, **common)
        kind = f"GridSearchCV({n_points} points)"

    n_fits = (cap if randomized else n_points) * folds.get_n_splits()
    print(f"    {model_name(key):<26} {kind}, {n_fits} fits...", flush=True)
    t0 = time.time()
    search.fit(X, y)
    print(f"    {'':<26} best f1={search.best_score_:.4f}  ({time.time() - t0:.0f}s)")
    return search, kind


def default_cv_score(key, X, y, folds, jobs):
    """CV score of the shipped configuration, on the same folds as the search."""
    estimator, _ = search_space(key, "default")
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", estimator)])
    pipe.set_params(**default_params(key))
    scores = cross_val_score(pipe, X, y, cv=folds, scoring="f1", n_jobs=jobs)
    return float(scores.mean()), float(scores.std())


def parameter_effects(cv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Marginal effect of each hyperparameter: average CV F1 over every candidate
    that used a given value. A tiny spread means the knob does not matter.
    """
    rows = []
    for col in [c for c in cv_df.columns if c.startswith("param_")]:
        # Concatenating several models' results unions their parameter columns,
        # so a model's frame carries all-empty columns belonging to the others.
        if cv_df[col].isna().all():
            continue
        g = cv_df.groupby(cv_df[col].astype(str))["mean_test_score"].mean()
        if len(g) < 2:  # a parameter with one value in the grid explains nothing
            continue
        rows.append({
            "parameter": col.replace("param_", "").split("__")[-1],
            "best_value": g.idxmax(), "best_f1": g.max(),
            "worst_value": g.idxmin(), "worst_f1": g.min(),
            "spread": g.max() - g.min(),
        })
    return pd.DataFrame(rows).sort_values("spread", ascending=False)


# --- honest comparison on the test split -----------------------------------

def fit_and_predict(params, key, train_text, y_train, test_text, random_state):
    """Fit one pipeline configuration on the full training split, predict test."""
    estimator, _ = search_space(key, "default")
    if hasattr(estimator, "random_state"):
        estimator.set_params(random_state=random_state)
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", estimator)])
    pipe.set_params(**params)
    pipe.fit(train_text, y_train)
    return pipe.predict(test_text)


def _f1(y, pred):
    """F1 for the positive class (real = 1). Inlined so the bootstrap is cheap."""
    tp = np.count_nonzero((pred == REAL) & (y == REAL))
    fp = np.count_nonzero((pred == REAL) & (y != REAL))
    fn = np.count_nonzero((pred != REAL) & (y == REAL))
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def paired_bootstrap(y, pred_default, pred_tuned, n_boot, seed):
    """
    Resample the test set and recompute both F1 scores on each resample.

    Paired, so the two configurations always see the same articles: the
    spread of the *difference* is what tells you whether a tuning gain is
    real or is just which articles happened to land in the test split.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = np.empty(n_boot)
    base = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, n, n)
        yb = y[pick]
        d = _f1(yb, pred_default[pick])
        t = _f1(yb, pred_tuned[pick])
        base[i] = d
        deltas[i] = t - d
    return {
        "test_f1_default": _f1(y, pred_default),
        "test_f1_tuned": _f1(y, pred_tuned),
        "delta": _f1(y, pred_tuned) - _f1(y, pred_default),
        "delta_lo": float(np.percentile(deltas, 2.5)),
        "delta_hi": float(np.percentile(deltas, 97.5)),
        "noise_sd": float(base.std()),
    }


# --- feature ablation ------------------------------------------------------

def ablate_once(k, train_text, y_train, test_text, y_test, random_state):
    """Train every model on a k-term vocabulary and score it on the test split."""
    vec = TfidfVectorizer(max_features=k, ngram_range=NGRAM_RANGE, min_df=DEFAULT_MIN_DF)
    X_train = vec.fit_transform(train_text)
    X_test = vec.transform(test_text)
    rows = []
    for key, est in build_models(random_state).items():
        est.fit(X_train, y_train)
        pred = est.predict(X_test)
        rows.append({
            "max_features": k,
            "n_features": int(X_train.shape[1]),
            "model": key,
            "accuracy": accuracy_score(y_test, pred),
            "f1": f1_score(y_test, pred, zero_division=0),
        })
    return rows


def run_ablation(variant, df, idx_train, idx_test, feature_counts, args):
    col = "clean_raw" if variant == "raw" else "clean_stripped"
    train_text = df.loc[idx_train, col]
    test_text = df.loc[idx_test, col]
    y_train, y_test = df.loc[idx_train, "label"], df.loc[idx_test, "label"]

    batches = joblib.Parallel(n_jobs=args.jobs, verbose=0)(
        joblib.delayed(ablate_once)(k, train_text, y_train, test_text, y_test,
                                    args.random_state)
        for k in feature_counts
    )
    out = pd.DataFrame([r for batch in batches for r in batch])
    out["variant"] = variant
    return out


def vocabulary_ranking(train_text, y_train, top_n, probe=20000):
    """
    Which terms a TF-IDF vocabulary keeps first, and how class-specific they are.

    `max_features=k` is an *unsupervised* cut: sklearn keeps the k terms with
    the highest total corpus frequency, with no reference to the label. So the
    ranking below is exactly the order in which terms enter the vocabulary as k
    grows, and `in_fake` / `in_real` show what each one is worth as a signal.
    """
    vec = CountVectorizer(ngram_range=NGRAM_RANGE, min_df=DEFAULT_MIN_DF,
                          max_features=probe)
    X = vec.fit_transform(train_text)
    terms = np.array(vec.get_feature_names_out())
    tfs = np.asarray(X.sum(axis=0)).ravel()
    order = (-tfs).argsort()

    present = (X > 0)
    is_fake = (y_train == FAKE).to_numpy()
    n_fake, n_real = int(is_fake.sum()), int((~is_fake).sum())
    df_fake = np.asarray(present[is_fake].sum(axis=0)).ravel()
    df_real = np.asarray(present[~is_fake].sum(axis=0)).ravel()

    ranked_terms = terms[order]
    table = pd.DataFrame({
        "rank": np.arange(1, top_n + 1),
        "term": ranked_terms[:top_n],
        "corpus_tf": tfs[order][:top_n],
        "in_fake": df_fake[order][:top_n] / max(n_fake, 1),
        "in_real": df_real[order][:top_n] / max(n_real, 1),
    }).set_index("rank")
    table["gap"] = (table["in_real"] - table["in_fake"]).abs()

    # Where do the publisher fingerprints sit in that ordering?
    hits = [(i + 1, t) for i, t in enumerate(ranked_terms) if "reuters" in t]
    return table, hits, len(terms)


# --- figures ---------------------------------------------------------------

def plot_tuning(summary: pd.DataFrame, cv_df: pd.DataFrame, enabled=True):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    keys = list(summary.index)
    x = np.arange(len(keys))

    axes[0].bar(x - 0.2, summary["default_cv_f1"], 0.4, color="#adb5bd", label="default")
    axes[0].bar(x + 0.2, summary["tuned_cv_f1"], 0.4,
                color=[COLORS[k] for k in keys], label="tuned")
    for i, k in enumerate(keys):
        d = summary.loc[k, "cv_delta"]
        axes[0].text(i, max(summary.loc[k, "default_cv_f1"], summary.loc[k, "tuned_cv_f1"]),
                     f"{d:+.4f}", ha="center", va="bottom", fontsize=8)
    lo = min(summary[["default_cv_f1", "tuned_cv_f1"]].min()) - 0.01
    axes[0].set_ylim(max(lo, 0), 1.005)
    axes[0].set_xticks(x, [model_name(k) for k in keys], fontsize=8)
    axes[0].set_ylabel("cross-validated F1 (training split)")
    axes[0].set_title("Default vs tuned - the gain is the number on top")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")

    rng = np.random.default_rng(0)
    for i, k in enumerate(keys):
        s = cv_df.loc[cv_df["model"] == k, "mean_test_score"].to_numpy()
        axes[1].scatter(i + rng.uniform(-0.16, 0.16, len(s)), s, s=14, alpha=0.65,
                        color=COLORS[k], edgecolors="none")
        axes[1].hlines(summary.loc[k, "default_cv_f1"], i - 0.3, i + 0.3,
                       color="black", linewidth=1.2)
        axes[1].text(i, s.min(), f"range\n{s.max() - s.min():.4f}", ha="center",
                     va="top", fontsize=7)
    axes[1].set_xticks(x, [model_name(k) for k in keys], fontsize=8)
    axes[1].set_ylabel("mean CV F1")
    axes[1].set_title("Every candidate in the grid (black bar = default)")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    return save_fig(fig, "tuning_results.png", enabled)


def plot_ablation(table: pd.DataFrame, one_rule: dict, enabled=True):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    styles = {"raw": "-", "stripped": "--"}

    for ax, metric in zip(axes, ("f1", "accuracy")):
        for variant, sub_v in table.groupby("variant"):
            for key, sub in sub_v.groupby("model"):
                sub = sub.sort_values("max_features")
                ax.plot(sub["max_features"], sub[metric], styles[variant],
                        color=COLORS[key], linewidth=1.7, marker="o", markersize=3.5)
        ax.axhline(one_rule[metric], color="gray", linestyle=":", linewidth=1.2)
        ax.set_xscale("log")
        ax.set_xlabel("max_features (TF-IDF vocabulary size)")
        ax.set_ylabel(f"test {metric}")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_title("Test F1 vs vocabulary size")
    axes[1].set_title("Test accuracy vs vocabulary size")

    model_handles = [plt.Line2D([], [], color=COLORS[k], label=model_name(k))
                     for k in table["model"].unique()]
    style_handles = [plt.Line2D([], [], color="black", linestyle=styles[v],
                                label=f"{v} text") for v in table["variant"].unique()]
    style_handles.append(plt.Line2D([], [], color="gray", linestyle=":",
                                    label="one rule: 'mentions Reuters' (raw)"))
    leg = axes[0].legend(handles=model_handles, frameon=False, fontsize=8, loc="lower right")
    axes[0].add_artist(leg)
    axes[1].legend(handles=style_handles, frameon=False, fontsize=7, loc="lower right")
    return save_fig(fig, "feature_ablation.png", enabled)


# --- report ----------------------------------------------------------------

def main():
    args = parse_args()
    ensure_dirs()
    started = time.time()
    figs = not args.no_figures

    # LinearSVC occasionally hits max_iter on a 50k-feature candidate. The fit
    # is still usable and the warning floods the log from every worker.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    level = "quick" if args.quick else ("full" if args.full else "default")
    if args.max_candidates is None:
        # --full means "exhaustive" unless the user explicitly caps it.
        args.max_candidates = 0 if args.full else 40
    keys = MODEL_KEYS if args.model == "all" else [args.model]

    print("[1/7] Loading data and building both text variants...")
    df, idx_train, idx_test = prepare_data(args)
    y_train_all, y_test = df.loc[idx_train, "label"], df.loc[idx_test, "label"]
    n_emptied = int((df["clean_stripped"].str.len() == 0).sum())
    print(f"    {len(df):,} articles -> {len(idx_train):,} train / {len(idx_test):,} test")
    print(f"    stripping boilerplate empties {n_emptied} article(s) completely")

    print("[2/7] Ranking the vocabulary (which terms get selected first)...")
    search_col = "clean_stripped" if args.strip_boilerplate else "clean_raw"
    first_terms, reuters_hits, probe_vocab = vocabulary_ranking(
        df.loc[idx_train, search_col], y_train_all, args.top_terms)
    first_terms.to_csv(REPORTS_DIR / "first_features.csv")
    if reuters_hits:
        print(f"    first Reuters-bearing term is rank {reuters_hits[0][0]} "
              f"({reuters_hits[0][1]!r})")
    else:
        print("    no Reuters-bearing term in the probe vocabulary")

    summary = pd.DataFrame()
    cv_df = pd.DataFrame()
    effects = pd.DataFrame()
    best_params = {}
    boot = {}
    search_kind = ""
    n_search = 0

    if not args.skip_search:
        folds = StratifiedKFold(n_splits=args.cv, shuffle=True,
                                random_state=args.random_state)
        idx_search = subsample_index(idx_train, df["label"], args.search_sample,
                                     args.random_state)
        n_search = len(idx_search)
        X_search = df.loc[idx_search, search_col]
        y_search = df.loc[idx_search, "label"]
        print(f"[3/7] Cross-validating the shipped default config "
              f"({n_search:,} docs, {args.cv} folds)...")
        defaults = {}
        for key in keys:
            m, s = default_cv_score(key, X_search, y_search, folds, args.jobs)
            defaults[key] = (m, s)
            print(f"    {model_name(key):<26} f1={m:.4f} +/- {s:.4f}")

        print(f"[4/7] Searching ({level} grid, training split only)...")
        rows, frames = [], []
        for key in keys:
            search, search_kind = run_search(key, level, X_search, y_search, folds, args)
            res = pd.DataFrame(search.cv_results_)
            res["model"] = key
            frames.append(res)
            best = search.best_params_
            best_params[key] = best
            rows.append({
                "model": model_name(key),
                "key": key,
                "default_cv_f1": defaults[key][0],
                "tuned_cv_f1": float(search.best_score_),
                "cv_delta": float(search.best_score_) - defaults[key][0],
                "cv_std_at_best": float(res.loc[search.best_index_, "std_test_score"]),
                "best_params": fmt_params(best),
            })
        summary = pd.DataFrame(rows).set_index("key")
        cv_df = pd.concat(frames, ignore_index=True)
        cv_df.to_csv(REPORTS_DIR / "tuning_cv_results.csv", index=False)
        effects = pd.concat(
            [parameter_effects(cv_df[cv_df["model"] == k]).assign(model=model_name(k))
             for k in keys], ignore_index=True
        ).set_index("model")

        print("[5/7] Refitting default and tuned configs on the FULL training "
              "split, scoring the test split once...")
        train_text = df.loc[idx_train, search_col]
        test_text = df.loc[idx_test, search_col]
        jobs = joblib.Parallel(n_jobs=args.jobs)(
            joblib.delayed(fit_and_predict)(params, key, train_text, y_train_all,
                                            test_text, args.random_state)
            for key in keys
            for params in (default_params(key), best_params[key])
        )
        y_test_arr = y_test.to_numpy()
        for i, key in enumerate(keys):
            boot[key] = paired_bootstrap(y_test_arr, jobs[2 * i], jobs[2 * i + 1],
                                         args.n_boot, args.random_state)
            b = boot[key]
            print(f"    {model_name(key):<26} test f1 {b['test_f1_default']:.4f} -> "
                  f"{b['test_f1_tuned']:.4f}  (delta {b['delta']:+.4f}, "
                  f"95% CI [{b['delta_lo']:+.4f}, {b['delta_hi']:+.4f}])")

        summary["test_f1_default"] = [boot[k]["test_f1_default"] for k in summary.index]
        summary["test_f1_tuned"] = [boot[k]["test_f1_tuned"] for k in summary.index]
    else:
        print("[3/7] --skip-search: no search run")
        print("[4/7] --skip-search: no search run")
        print("[5/7] --skip-search: no test comparison")

    ablation = pd.DataFrame()
    one_rule = {"accuracy": float("nan"), "f1": float("nan")}
    if not args.skip_ablation:
        variants = ["raw", "stripped"] if args.ablation_variants == "both" \
            else [args.ablation_variants]
        counts = sorted(args.ablation_features)
        print(f"[6/7] Feature ablation: {len(counts)} vocabulary sizes x "
              f"{len(variants)} text variant(s) x 3 models...")
        parts = []
        for variant in variants:
            t0 = time.time()
            part = run_ablation(variant, df, idx_train, idx_test, counts, args)
            parts.append(part)
            best_row = part.loc[part["f1"].idxmax()]
            print(f"    {variant:<9} best f1={best_row['f1']:.4f} at "
                  f"max_features={int(best_row['max_features'])} "
                  f"({model_name(best_row['model'])}), {time.time() - t0:.0f}s")
        ablation = pd.concat(parts, ignore_index=True)
        ablation.to_csv(REPORTS_DIR / "feature_ablation.csv", index=False)

        # The zero-machine-learning baseline, on the same test articles.
        rule_pred = df.loc[idx_test, "content"].str.contains(
            r"\breuters\b", case=False, na=False).astype(int).to_numpy()
        one_rule = {
            "accuracy": accuracy_score(y_test, rule_pred),
            "f1": f1_score(y_test, rule_pred, zero_division=0),
        }
        print(f"    one-rule 'mentions Reuters' baseline: "
              f"acc={one_rule['accuracy']:.4f} f1={one_rule['f1']:.4f}")
    else:
        print("[6/7] --skip-ablation: no ablation run")

    print("[7/7] Figures and report...")
    tuning_fig = plot_tuning(summary, cv_df, figs) if not summary.empty else ""
    ablation_fig = plot_ablation(ablation, one_rule, figs) if not ablation.empty else ""

    write_outputs(args, level, keys, summary, cv_df, effects, best_params, boot,
                  ablation, one_rule, first_terms, reuters_hits, probe_vocab,
                  n_search, len(idx_train), len(idx_test), n_emptied,
                  search_kind, tuning_fig, ablation_fig)

    print(f"\nWrote {REPORTS_DIR / 'tuning_report.md'}")
    print(f"      {REPORTS_DIR / 'best_params.json'}")
    print(f"Total time: {time.time() - started:.0f}s")


def verdict_lines(summary, boot):
    """The 'was tuning worth it' paragraph, driven by the bootstrap CI."""
    if summary.empty:
        return "_Search skipped, so there is nothing to judge here._"
    lines = []
    for key in summary.index:
        b = boot[key]
        crosses_zero = b["delta_lo"] <= 0 <= b["delta_hi"]
        verdict = ("**not worth it** - the 95% CI on the change straddles zero"
                   if crosses_zero else
                   "a real but small change - the CI excludes zero")
        lines.append(
            f"- **{model_name(key)}**: CV F1 {summary.loc[key, 'default_cv_f1']:.4f} -> "
            f"{summary.loc[key, 'tuned_cv_f1']:.4f} ({summary.loc[key, 'cv_delta']:+.4f}); "
            f"test F1 {b['test_f1_default']:.4f} -> {b['test_f1_tuned']:.4f} "
            f"({b['delta']:+.4f}, 95% CI [{b['delta_lo']:+.4f}, {b['delta_hi']:+.4f}], "
            f"bootstrap SD of a single test F1 = {b['noise_sd']:.4f}). {verdict}."
        )
    return "\n".join(lines)


def write_outputs(args, level, keys, summary, cv_df, effects, best_params, boot,
                  ablation, one_rule, first_terms, reuters_hits, probe_vocab,
                  n_search, n_train, n_test, n_emptied, search_kind,
                  tuning_fig, ablation_fig):
    def embed(path, cap):
        return f"\n![{cap}]({path})\n" if path else ""

    # --- best_params.json: feeds straight back into train.py ---------------
    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "search": {
            "grid_level": level, "search_kind": search_kind, "cv_folds": args.cv,
            "scoring": "f1", "search_docs": n_search, "train_docs": n_train,
            "strip_boilerplate": args.strip_boilerplate,
            "random_state": args.random_state,
        },
        "environment": {"python": platform.python_version(),
                        "sklearn": sklearn.__version__},
        "defaults": {k: (list(v) if isinstance(v, tuple) else v)
                     for k, v in DEFAULT_TFIDF.items()},
        "models": {},
    }
    for key in (summary.index if not summary.empty else []):
        params = {k.split("__")[-1]: (list(v) if isinstance(v, tuple) else v)
                  for k, v in best_params[key].items()}
        payload["models"][key] = {
            "params": params,
            "cv_f1_tuned": float(summary.loc[key, "tuned_cv_f1"]),
            "cv_f1_default": float(summary.loc[key, "default_cv_f1"]),
            "cv_delta": float(summary.loc[key, "cv_delta"]),
            "test_f1_default": boot[key]["test_f1_default"],
            "test_f1_tuned": boot[key]["test_f1_tuned"],
            "test_delta_ci95": [boot[key]["delta_lo"], boot[key]["delta_hi"]],
            # train.py exposes only these three as flags today; sublinear_tf and
            # the classifier parameter would need a small change there.
            "train_py_command": (
                f"python src/train.py --max-features {params['max_features']} "
                f"--ngram-max {params['ngram_range'][1]} --min-df {params['min_df']}"
            ),
            "not_exposed_by_train_py": {
                k: v for k, v in params.items()
                if k in ("sublinear_tf", "alpha", "C")
            },
        }
    (REPORTS_DIR / "best_params.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    # --- markdown report ---------------------------------------------------
    if summary.empty:
        search_section = "_Skipped (`--skip-search`)._\n"
        effects_section = ""
    else:
        shown = summary[["model", "default_cv_f1", "tuned_cv_f1", "cv_delta",
                         "cv_std_at_best", "best_params"]].set_index("model")
        shown.index.name = "model"
        search_section = md_table(shown)
        eff = effects[["parameter", "best_value", "best_f1", "worst_value",
                       "worst_f1", "spread"]]
        effects_section = f"""
### Which knob actually moved the score

Marginal means: for each parameter value, the average CV F1 over every
candidate that used it. `spread` is the gap between the best and worst value -
a parameter with a spread near zero is one this dataset does not care about.

{md_table(eff)}
"""

    if ablation.empty:
        ablation_section = "_Skipped (`--skip-ablation`)._\n"
    else:
        pivot = ablation.pivot_table(index="max_features",
                                     columns=["variant", "model"], values="f1")
        pivot.columns = [f"{v}/{m}" for v, m in pivot.columns]
        pivot.index.name = "max_features"
        ablation_section = md_table(pivot)

    # Headline numbers for the honest reading, computed rather than asserted.
    ablation_reading = ""
    ablation_point = "the ablation was skipped"
    if not ablation.empty:
        def best_acc(frame, k):
            hit = frame.loc[frame["max_features"] == k, "accuracy"]
            return float(hit.max()) if len(hit) else float("nan")

        raw = ablation[ablation["variant"] == "raw"]
        stripped = ablation[ablation["variant"] == "stripped"]
        smallest = int(raw["max_features"].min())
        largest = int(raw["max_features"].max())
        best_small, best_large = best_acc(raw, smallest), best_acc(raw, largest)
        # How flat is the curve once it has saturated? Take everything from the
        # second-smallest vocabulary upwards and measure the total range.
        tail = raw[raw["max_features"] > smallest].groupby("max_features")["accuracy"].max()
        tail_range = float(tail.max() - tail.min()) if len(tail) else float("nan")
        at200_line = (f"Two hundred features reach {best_acc(raw, 200):.4f}."
                      if (raw["max_features"] == 200).any() else "")
        strip_small = best_acc(stripped, smallest) if len(stripped) else float("nan")
        strip_large = best_acc(stripped, largest) if len(stripped) else float("nan")
        stripped_para = (
            f"With boilerplate **stripped**, the same {smallest}-term vocabulary gives "
            f"{strip_small:.4f} and {largest} terms give {strip_large:.4f}. The gap "
            "between the two curves is the part of the score the publisher "
            "fingerprint was carrying."
            if len(stripped) else
            "The stripped variant was not run, so there is nothing to compare against."
        )
        ablation_point = (f"{smallest} features already reach {best_small:.1%} accuracy "
                          f"and the curve moves by only {tail_range:.4f} above that")

        ablation_reading = f"""
Accuracy at the same points (the table above is F1; both are in
`reports/feature_ablation.csv` and both panels are in the figure). Numbers are
the best of the three models at that vocabulary size.

With the **raw** text, {smallest} features reach **{best_small:.4f} accuracy**
and {largest} features reach {best_large:.4f}. {at200_line} Across every
vocabulary size from {smallest * 2 if smallest * 2 <= largest else smallest}
upwards the accuracy moves by {tail_range:.4f} in total.

A model that needs {smallest} words to hit {best_small:.1%} is not answering
"is this article true". It is answering "which of two publishers wrote this",
and a handful of high-frequency tokens is enough for that. This is the same
conclusion `reports/eda_report.md` reaches from the other direction, and it is
why `max_features` is not a meaningful thing to tune on this corpus: every
value in the grid sits far past the point where the leak is exhausted, so the
search can only pick between configurations that have all already saturated.

{stripped_para}
"""

    if reuters_hits:
        rank, term = reuters_hits[0]
        nearby = ", ".join(f"`{t}` (rank {r})" for r, t in reuters_hits[:6])
        reuters_note = f"""
`reuters` first enters the vocabulary at **rank {rank}** of
{probe_vocab:,} probed terms, as `{term}`. The first six Reuters-bearing
terms are {nearby}.

This ranking is unsupervised - `max_features=k` keeps the k most frequent
terms and never looks at the label - so the ordering is not a model choosing
a good feature. It is simply that the word appears in essentially every real
article and essentially no fake one, which makes it frequent *and* perfectly
discriminative at the same time. Any vocabulary large enough to include rank
{rank} gets the leak for free.
"""
    else:
        reuters_note = ("\nNo Reuters-bearing term appeared in the probed vocabulary, "
                        "which is what you would expect if the text was de-leaked "
                        "before this ran.\n")

    if summary.empty:
        sample_line = "The search was skipped, so sections 1 and 2 are empty."
    elif n_search < n_train:
        sample_line = (
            f"The search ran on a stratified subsample of {n_search:,} training "
            f"documents ({n_search / max(n_train, 1):.0%} of the split). Searching the "
            "whole split costs roughly four times as much wall-clock for differences "
            "that, as the results below show, sit inside the noise. "
            "`--search-sample 0` uses all of it."
        )
    else:
        sample_line = f"The search used the whole training split ({n_train:,} documents)."

    one_rule_line = (f"one rule (\"mentions Reuters\") scores "
                     f"accuracy {one_rule['accuracy']:.4f} / F1 {one_rule['f1']:.4f} "
                     f"on the same test articles"
                     if one_rule["accuracy"] == one_rule["accuracy"]
                     else "the one-rule baseline was not computed (ablation skipped)")

    # Section 5 makes claims about how little any of this mattered. Measure the
    # claims rather than asserting them, so a different dataset would change them.
    if cv_df.empty:
        grid_line = ("The search was skipped, so nothing here says whether the "
                     "defaults were well chosen.")
    else:
        spans = cv_df.groupby("model")["mean_test_score"].agg(lambda s: s.max() - s.min())
        worst = cv_df["mean_test_score"].min()
        grid_line = (f"Best and worst candidate in the grid differ by at most "
                     f"{spans.max():.4f} F1 (worst candidate anywhere: {worst:.4f}). "
                     f"Reporting \"we tuned the model and reached "
                     f"{cv_df['mean_test_score'].max():.4f}\" would be true and "
                     f"uninformative - the score was already there before tuning, "
                     f"and a keyword rule reaches it too.")

    report = f"""# Hyperparameter Tuning and Feature Ablation

Generated by `python src/tune.py`. Raw tables: `reports/tuning_cv_results.csv`,
`reports/feature_ablation.csv`, `reports/first_features.csv`. Best parameters
in machine-readable form: `reports/best_params.json`.

Dataset: {n_train:,} training articles / {n_test:,} test articles
(`test_size={args.test_size}`, `random_state={args.random_state}`, duplicates dropped).

## Protocol

- Every parameter choice below was made by cross-validating on the **training
  split only** ({args.cv}-fold stratified, `scoring='f1'`). The test split was
  scored exactly once, at the end, to report the outcome - never to pick it.
- {sample_line}
- Grid level: `{level}`. Search: {search_kind or "not run"}.
- `svm` is searched as a bare `LinearSVC`; the shipped model wraps it in
  `CalibratedClassifierCV`, which changes the probability estimates but is
  tuned over the same `C`. Its numbers here are therefore close to, but not
  identical to, `models/metrics.json`.
- Text variant used for the search: **{"boilerplate stripped" if args.strip_boilerplate else "raw"}**.
  Stripping empties {n_emptied} article(s) entirely.

## 1. Search results

{search_section}
{effects_section}
{embed(tuning_fig, "tuning results")}

## 2. Was tuning worth it?

The honest test is not "did the number go up" but "did it go up by more than
the test split's own sampling noise". The interval below is a paired bootstrap
({args.n_boot:,} resamples) of the *difference* in test F1 between the default
and tuned configurations, refit on the full training split.

{verdict_lines(summary, boot)}

A confidence interval that contains zero means the tuned configuration and the
default one are indistinguishable on this data, and the tuning was not worth
doing. Where an interval excludes zero, check the size of the effect before
celebrating: a change of a few thousandths of F1 on a task a keyword rule
already solves is not a modelling improvement in any sense that matters.

## 3. Feature-count ablation

Test F1 at each vocabulary size, holding `ngram_range={NGRAM_RANGE}` and
`min_df={DEFAULT_MIN_DF}` fixed so only `max_features` varies. Columns are
`variant/model`.

{ablation_section}
{embed(ablation_fig, "feature ablation")}

For reference, {one_rule_line}.
{ablation_reading}

## 4. What the first features actually are

The {len(first_terms)} terms a TF-IDF vocabulary selects first, with the share
of each class that contains them. `gap` near 1.0 means the term alone
separates the classes.

{md_table(first_terms, floatfmt="{:.3f}")}
{reuters_note}

## 5. Reading this honestly

1. **Tuning is not where the problem is.** {grid_line}
2. **`max_features={MAX_FEATURES}` was never load-bearing.** In the ablation,
   {ablation_point}. The defensible claim about the current value is
   "comfortably past saturation", not "optimal", and that is the claim this
   project should make.
3. **The interesting axis is the text, not the hyperparameters.** The gap
   between the raw and stripped ablation curves is larger than anything the
   grid produced. That is where the modelling effort belongs.
4. **What tuning is still good for.** It rules out the possibility that the
   defaults were badly wrong, and it turns three magic numbers into three
   measured ones. That is a real if unglamorous result, and it is the only
   claim these numbers support.

## Feeding this back into training

`reports/best_params.json` stores the winning parameters. `train.py` currently
exposes `--max-features`, `--ngram-max` and `--min-df`; `sublinear_tf` and the
classifier's own parameter (`alpha` / `C`) are not CLI flags yet, so applying
the full configuration would need a small change there. Given section 2, doing
so would not measurably change the results.
"""

    (REPORTS_DIR / "tuning_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
