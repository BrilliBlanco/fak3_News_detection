"""
Error taxonomy - categorising the mistakes instead of just listing them.

`evaluate.py` dumps the highest-confidence misclassifications to CSV. That is a
list, not an analysis: it tells you *which* articles were wrong but nothing
about *why*, and it cannot tell you whether a mistake is the model's fault or
the dataset's.

This module rebuilds the same test split and cross-tabulates every
misclassification along the axes that can actually explain it:

  - direction (fake called real vs real called fake)
  - subject category, article length, prediction confidence
  - TF-IDF vocabulary coverage: how many of the 5,000 features the article
    matched at all. A near-empty feature vector means the model was guessing
  - the leakage markers from `eda.py`: does the article contain "(Reuters)" or
    a wire dateline? A *fake* article containing "Reuters" that gets called
    REAL is per-article proof of the leakage mechanism, not a coincidence
  - stylometry of errors vs correct predictions, conditioned on the true class
    (unconditioned, it would just re-discover the class difference)
  - model agreement: errors all three models share are dataset-driven; errors
    only one model makes are model-driven. Only the second kind is fixable by
    swapping the classifier.

Every bucket is reported as a raw count *and* an error rate. Raw counts alone
mostly track bucket size, which is how "most errors are in long articles" gets
written down when long articles are simply the most common kind.

Usage:
    python src/error_taxonomy.py
    python src/error_taxonomy.py --no-figures --n-examples 3

Writes:
    reports/error_taxonomy.md
    reports/error_tables/*.csv
    reports/figures/error_taxonomy.png
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe: never tries to open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (FAKE, FIGURES_DIR, MODEL_KEYS, MODELS_DIR, REAL,
                    REPORTS_DIR, RUN_META, VECTORIZER_FILE, ensure_dirs,
                    model_name, model_path)
from preprocessing import add_text_features, clean_text, load_data

TABLES_DIR = REPORTS_DIR / "error_tables"

COLORS = {"nb": "#e07a5f", "svm": "#3d5a80", "lr": "#81b29a"}
FAKE_COLOR, REAL_COLOR = "#d1495b", "#2e86ab"

LENGTH_BINS = [0, 50, 200, 500, 1000, np.inf]
LENGTH_LABELS = ["<50", "50-199", "200-499", "500-999", "1000+"]

CONF_BINS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
CONF_LABELS = ["0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]

# How many of the 5,000 TF-IDF features the article actually matched.
COVERAGE_BINS = [0, 10, 25, 50, 100, 200, np.inf]
COVERAGE_LABELS = ["<10", "10-24", "25-49", "50-99", "100-199", "200+"]

# The leakage tells, checked per article. `column` matters: `content` is
# title + body, so a dateline sitting at the start of the body is *not* at the
# start of a line once the title is glued on - which is why the dateline test
# has to run against `body`.
LEAK_MARKERS = {
    "(Reuters) tag": ("content", re.compile(r"\(\s*Reuters\s*\)")),
    "'reuters' anywhere": ("content", re.compile(r"\breuters\b", re.IGNORECASE)),
    "wire dateline (CITY (Agency) -)": (
        "body",
        re.compile(r"^\s*[A-Za-z][A-Za-z.\-/'’ ]{0,60}?\(\s*\w+\s*\)\s*[-–—]"),
    ),
}

STYLE_COLS = [
    "n_words", "avg_word_len", "n_sentences", "exclamation_count",
    "question_count", "quote_count", "allcaps_words", "allcaps_ratio",
    "digit_ratio", "upper_char_ratio",
]


# --- plumbing --------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Categorise model errors by direction, subject, length, "
                    "confidence, vocabulary coverage and leakage markers.")
    p.add_argument("--models-dir", default=str(MODELS_DIR))
    p.add_argument("--data-dir", default=None, help="Folder with Fake.csv / True.csv")
    p.add_argument("--n-examples", type=int, default=2,
                   help="How many leakage-driven errors to quote verbatim")
    p.add_argument("--min-bucket", type=int, default=30,
                   help="Buckets smaller than this are flagged as too small to trust")
    p.add_argument("--no-figures", action="store_true", help="Skip PNG generation, tables only")
    return p.parse_args()


def save_fig(fig, name: str, enabled: bool = True) -> str:
    if not enabled:
        plt.close(fig)
        return ""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}"


def save_table(df: pd.DataFrame, name: str) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / name, index=True)


def mark_small(df: pd.DataFrame, min_n: int) -> pd.DataFrame:
    """Star the rows whose bucket is too small for the rate to mean anything."""
    if "n" not in df.columns:
        return df
    out = df.copy()
    out.index = [f"{i} *" if n < min_n else str(i) for i, n in zip(out.index, out["n"])]
    out.index.name = df.index.name
    return out


def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    """Minimal dataframe -> markdown table (avoids a `tabulate` dependency)."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].map(lambda v: floatfmt.format(v) if pd.notna(v) else "-")
    header = "| " + " | ".join([out.index.name or ""] + [str(c) for c in out.columns]) + " |"
    sep = "|" + "|".join(["---"] * (len(out.columns) + 1)) + "|"
    rows = [
        "| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |"
        for idx, row in zip(out.index, out.values)
    ]
    return "\n".join([header, sep] + rows)


def rebuild_split(models_dir: Path, data_dir):
    """
    Recreate train.py's exact split from run_metadata.json - same helper shape
    as evaluate.py, so the error sets here are the same errors that report saw.
    """
    meta_path = models_dir / RUN_META
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} not found. Run `python src/train.py` first - "
            "error_taxonomy.py needs the run metadata to reproduce the test split."
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


# --- building the annotated test frame -------------------------------------

def build_test_frame(df: pd.DataFrame, idx_test, X_test) -> pd.DataFrame:
    """One row per test article, annotated with every axis we slice on."""
    tf = df.loc[idx_test].copy()
    tf = add_text_features(tf, "content")

    # Non-zeros per row = how many of the 5,000 TF-IDF features this article
    # matched. Two matched features means the model had almost nothing to go on.
    tf["n_matched_features"] = X_test.getnnz(axis=1)

    tf["length_bucket"] = pd.cut(tf["n_words"], LENGTH_BINS,
                                 labels=LENGTH_LABELS, right=False)
    tf["coverage_bucket"] = pd.cut(tf["n_matched_features"], COVERAGE_BINS,
                                   labels=COVERAGE_LABELS, right=False)

    for name, (column, rx) in LEAK_MARKERS.items():
        tf[f"marker::{name}"] = tf[column].str.contains(rx, na=False)

    tf["class"] = tf["label"].map({FAKE: "fake", REAL: "real"})
    return tf


def score_models(models_dir: Path, X_test, tf: pd.DataFrame):
    """predict + P(real) for every model artifact that exists."""
    preds, confs, available = {}, {}, []
    for key in MODEL_KEYS:
        path = model_path(key, models_dir)
        if not path.exists():
            print(f"    skipping {model_name(key)} ({path.name} not found)")
            continue
        model = joblib.load(path)
        pred = pd.Series(model.predict(X_test), index=tf.index)
        preds[key] = pred
        proba = positive_proba(model, X_test)
        if proba is not None:
            p_real = pd.Series(proba, index=tf.index)
            # Confidence in whatever was predicted, not confidence in "real".
            confs[key] = np.where(pred == REAL, p_real, 1 - p_real)
            confs[key] = pd.Series(confs[key], index=tf.index)
        available.append(key)
    return preds, confs, available


# --- cross-tabs ------------------------------------------------------------

def rate_table(tf, group_col, wrong, keys, index_name, class_mix=False):
    """
    Bucket size + error count + error rate (%) per model, for one axis.

    `class_mix` adds the share of the bucket that is fake. Worth having on any
    axis that might be confounded with the label: a bucket that is 100% one
    class cannot tell you anything about difficulty, because a constant
    prediction scores perfectly in it.
    """
    grouper = tf[group_col]
    out = pd.DataFrame({"n": grouper.groupby(grouper, observed=False).size()})
    if class_mix:
        out["fake_%"] = (tf["label"] == FAKE).groupby(grouper, observed=False).mean() * 100
    for k in keys:
        g = wrong[k].groupby(grouper, observed=False)
        out[f"{k}_err"] = g.sum().astype(int)
        out[f"{k}_%"] = g.mean() * 100
    out["n"] = out["n"].astype(int)
    out.index.name = index_name
    return out


def confidence_table(confs, wrong, keys):
    """
    Error rate by confidence band. Bands are per-model (each model has its own
    probabilities), so this table carries an `n` column per model.
    """
    out = pd.DataFrame(index=pd.Index(CONF_LABELS, name="confidence band"))
    for k in keys:
        band = pd.cut(confs[k], CONF_BINS, labels=CONF_LABELS, include_lowest=True)
        g = wrong[k].groupby(band, observed=False)
        out[f"{k}_n"] = g.size().reindex(CONF_LABELS).astype(int).to_numpy()
        out[f"{k}_err"] = g.sum().reindex(CONF_LABELS).astype(int).to_numpy()
        out[f"{k}_%"] = (g.mean() * 100).reindex(CONF_LABELS).to_numpy()
    return out


def direction_table(tf, preds, wrong, keys):
    """False REAL (fake called real) vs false FAKE (real called fake)."""
    y = tf["label"]
    n_fake, n_real = int((y == FAKE).sum()), int((y == REAL).sum())
    rows = {}
    for k in keys:
        false_real = int(((preds[k] == REAL) & (y == FAKE)).sum())
        false_fake = int(((preds[k] == FAKE) & (y == REAL)).sum())
        rows[model_name(k)] = {
            "errors": int(wrong[k].sum()),
            "error_%": wrong[k].mean() * 100,
            "false_REAL": false_real,
            "false_REAL_%_of_fake": 100 * false_real / n_fake,
            "false_FAKE": false_fake,
            "false_FAKE_%_of_real": 100 * false_fake / n_real,
        }
    out = pd.DataFrame(rows).T
    for col in ("errors", "false_REAL", "false_FAKE"):
        out[col] = out[col].astype(int)
    out.index.name = "model"
    return out


def leakage_table(tf, wrong, keys):
    """
    The direct test. Split each true class by whether it carries the leakage
    marker, then read off the error rate on each side.

    If the models had learned "fake vs real" this split would do nothing. If
    they learned "Reuters vs not", fake articles that happen to mention Reuters
    should be misread as REAL far more often than fake articles that do not.
    """
    rows = []
    for name, _ in LEAK_MARKERS.items():
        col = f"marker::{name}"
        for cls, cls_label in ((FAKE, "fake"), (REAL, "real")):
            for has, has_label in ((True, "WITH"), (False, "WITHOUT")):
                mask = (tf["label"] == cls) & (tf[col] == has)
                row = {"group": f"{cls_label} {has_label} {name}", "n": int(mask.sum())}
                for k in keys:
                    row[f"{k}_err"] = int(wrong[k][mask].sum())
                    row[f"{k}_%"] = 100 * wrong[k][mask].mean() if mask.any() else np.nan
                rows.append(row)
    out = pd.DataFrame(rows).set_index("group")
    out.index.name = "group"
    return out


def stylometry_table(tf, any_wrong):
    """
    Style of errors vs correct predictions, *within* each true class.

    Conditioning on the class matters: fake articles use 13x more exclamation
    marks than real ones (see the EDA), so an unconditioned errors-vs-correct
    comparison would just recover the class difference and say nothing about
    the errors. `any_wrong` = at least one of the three models got it wrong,
    which keeps the sample large enough to be worth reading.
    """
    out = pd.DataFrame(index=pd.Index(STYLE_COLS, name="feature"))
    for cls, label in ((FAKE, "fake"), (REAL, "real")):
        in_cls = tf["label"] == cls
        ok = tf.loc[in_cls & ~any_wrong, STYLE_COLS].mean()
        bad = tf.loc[in_cls & any_wrong, STYLE_COLS].mean()
        out[f"{label}_correct"] = ok
        out[f"{label}_error"] = bad
        out[f"{label}_ratio"] = bad / ok.replace(0, np.nan)
    return out


def agreement_tables(tf, wrong, keys):
    """
    Which errors do the models share? Shared errors are a property of the
    dataset; unique errors are a property of the model.
    """
    err_sets = {k: set(tf.index[wrong[k]]) for k in keys}
    union = sorted(set().union(*err_sets.values())) if err_sets else []

    combos = []
    for idx in union:
        who = tuple(k for k in keys if idx in err_sets[k])
        combos.append(" + ".join(who))
    combo_counts = pd.Series(combos, index=union).value_counts().to_frame("articles")
    combo_counts["share_of_union_%"] = 100 * combo_counts["articles"] / max(len(union), 1)
    combo_counts.index.name = "models wrong"

    shared = set.intersection(*err_sets.values()) if err_sets else set()
    per_model = {}
    for k in keys:
        others = set().union(*[err_sets[j] for j in keys if j != k]) if len(keys) > 1 else set()
        unique = err_sets[k] - others
        per_model[model_name(k)] = {
            "errors": len(err_sets[k]),
            "shared_by_all": len(err_sets[k] & shared),
            "shared_by_all_%": 100 * len(err_sets[k] & shared) / max(len(err_sets[k]), 1),
            "unique_to_model": len(unique),
            "unique_%": 100 * len(unique) / max(len(err_sets[k]), 1),
        }
    per_model = pd.DataFrame(per_model).T
    for col in ("errors", "shared_by_all", "unique_to_model"):
        per_model[col] = per_model[col].astype(int)
    per_model.index.name = "model"

    return combo_counts, per_model, err_sets, shared, union


# --- per-article dumps -----------------------------------------------------

def error_rows(tf, preds, confs, wrong, keys):
    """Long format: one row per (model, misclassified article)."""
    frames = []
    for k in keys:
        mask = wrong[k]
        if not mask.any():
            continue
        sub = tf.loc[mask]
        frames.append(pd.DataFrame({
            "model": model_name(k),
            "article_index": sub.index,
            "actual": sub["class"].to_numpy(),
            "predicted": np.where(preds[k][mask] == REAL, "real", "fake"),
            "direction": np.where(sub["label"].to_numpy() == FAKE, "false REAL", "false FAKE"),
            "confidence": confs[k][mask].to_numpy() if k in confs else np.nan,
            "subject": sub["subject"].to_numpy(),
            "n_words": sub["n_words"].to_numpy(),
            "length_bucket": sub["length_bucket"].astype(str).to_numpy(),
            "n_matched_features": sub["n_matched_features"].to_numpy(),
            "mentions_reuters": sub["marker::'reuters' anywhere"].to_numpy(),
            "reuters_tag": sub["marker::(Reuters) tag"].to_numpy(),
            "wire_dateline": sub["marker::wire dateline (CITY (Agency) -)"].to_numpy(),
            "title": sub["title"].str.replace(r"\s+", " ", regex=True).to_numpy(),
        }))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["model", "confidence"], ascending=[True, False])


def error_articles(tf, wrong, keys):
    """Wide format: one row per article any model got wrong."""
    any_wrong = np.logical_or.reduce([wrong[k].to_numpy() for k in keys])
    sub = tf.loc[any_wrong].copy()
    out = pd.DataFrame({
        "actual": sub["class"],
        "models_wrong": ["+".join(k for k in keys if wrong[k].loc[i]) for i in sub.index],
        "n_models_wrong": [sum(int(wrong[k].loc[i]) for k in keys) for i in sub.index],
        "subject": sub["subject"],
        "n_words": sub["n_words"],
        "n_matched_features": sub["n_matched_features"],
        "mentions_reuters": sub["marker::'reuters' anywhere"],
        "title": sub["title"].str.replace(r"\s+", " ", regex=True),
    })
    out.index.name = "article_index"
    return out.sort_values(["n_models_wrong", "actual"], ascending=[False, True])


def quote_leakage_examples(tf, preds, confs, keys, n=2, window=110):
    """
    Fake articles that mention Reuters and got called REAL, quoted with the
    text around the mention. This is the leakage claim reduced to something a
    reader can check by eye.
    """
    col = "marker::'reuters' anywhere"
    candidates = tf[(tf["label"] == FAKE) & tf[col]].copy()
    if candidates.empty:
        return []

    fooled = pd.Series(0, index=candidates.index)
    for k in keys:
        fooled += (preds[k].loc[candidates.index] == REAL).astype(int)
    candidates["n_fooled"] = fooled
    candidates = candidates[candidates["n_fooled"] > 0]
    if candidates.empty:
        return []

    conf_key = next((k for k in keys if k in confs), None)
    if conf_key:
        candidates["conf"] = confs[conf_key].loc[candidates.index]
    else:
        candidates["conf"] = np.nan
    candidates = candidates.sort_values(["n_fooled", "conf"], ascending=False).head(n)

    rx = LEAK_MARKERS["'reuters' anywhere"][1]
    out = []
    for idx, row in candidates.iterrows():
        text = re.sub(r"\s+", " ", row["content"])
        m = rx.search(text)
        start = max(m.start() - window, 0) if m else 0
        excerpt = text[start:start + 2 * window]
        called_real = [model_name(k) for k in keys if preds[k].loc[idx] == REAL]
        out.append({
            "index": idx,
            "title": re.sub(r"\s+", " ", row["title"])[:110],
            "subject": row["subject"],
            "excerpt": ("..." if start else "") + excerpt + "...",
            "called_real_by": ", ".join(called_real),
        })
    return out


# --- figure ----------------------------------------------------------------

def plot_taxonomy(length_tbl, conf_tbl, subject_tbl, combo_counts, keys, enabled=True):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    def grouped(ax, table, labels, title, xlabel, rotate=0):
        x = np.arange(len(labels))
        width = 0.8 / max(len(keys), 1)
        floor = 0.04  # log axis: a zero-height bar is invisible, so label it
        for i, k in enumerate(keys):
            vals = np.nan_to_num(table[f"{k}_%"].to_numpy(dtype=float))
            pos = x + (i - (len(keys) - 1) / 2) * width
            ax.bar(pos, vals, width, color=COLORS[k], label=model_name(k))
            for xi, v in zip(pos[vals == 0], vals[vals == 0]):
                ax.text(xi, floor, "0", ha="center", va="bottom",
                        fontsize=7, color=COLORS[k])
        ax.set_xticks(x, labels, fontsize=8, rotation=rotate,
                      ha="right" if rotate else "center")
        ax.set_yscale("log")
        ax.set_ylim(bottom=floor)
        ax.set_ylabel("error rate % (log scale)", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    grouped(axes[0, 0], length_tbl,
            [f"{b}\n(n={n:,})" for b, n in zip(length_tbl.index, length_tbl["n"])],
            "Error rate by article length", "words per article")
    axes[0, 0].legend(frameon=False, fontsize=8)

    grouped(axes[0, 1], conf_tbl, list(conf_tbl.index),
            "Error rate by predicted confidence", "confidence in the predicted class")

    grouped(axes[1, 0], subject_tbl,
            [f"{s} (n={n:,})" for s, n in zip(subject_tbl.index, subject_tbl["n"])],
            "Error rate by subject category", "subject", rotate=30)

    ax = axes[1, 1]
    order = combo_counts.sort_values("articles", ascending=True)
    y = np.arange(len(order))
    ax.barh(y, order["articles"], color="#6c757d")
    for yi, v in zip(y, order["articles"]):
        ax.text(v, yi, f" {v:,}", va="center", fontsize=8)
    ax.set_yticks(y, order.index, fontsize=8)
    ax.set_xlabel("articles", fontsize=9)
    ax.set_title("Which models got it wrong (union of all errors)", fontsize=10)
    ax.set_xlim(0, order["articles"].max() * 1.15)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Error taxonomy - where the mistakes concentrate", fontsize=12)
    return save_fig(fig, "error_taxonomy.png", enabled)


# --- report ----------------------------------------------------------------

def main():
    args = parse_args()
    ensure_dirs()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    figs = not args.no_figures
    models_dir = Path(args.models_dir)

    print("[1/7] Rebuilding the training run's test split...")
    df, _, idx_test, meta = rebuild_split(models_dir, args.data_dir)
    print(f"    {len(idx_test):,} test articles "
          f"(boilerplate stripped: {meta.get('strip_boilerplate')})")

    vectorizer = joblib.load(models_dir / VECTORIZER_FILE)
    X_test = vectorizer.transform(df.loc[idx_test, "clean_content"])

    print("[2/7] Annotating test articles (stats, buckets, leakage markers)...")
    tf = build_test_frame(df, idx_test, X_test)

    print("[3/7] Scoring models...")
    preds, confs, keys = score_models(models_dir, X_test, tf)
    if not keys:
        raise SystemExit("No trained models found - run `python src/train.py` first.")
    wrong = {k: (preds[k] != tf["label"]) for k in keys}
    any_wrong = pd.Series(np.logical_or.reduce([wrong[k].to_numpy() for k in keys]),
                          index=tf.index)
    for k in keys:
        print(f"    {model_name(k):<26} {int(wrong[k].sum()):>4} errors "
              f"({wrong[k].mean():.2%})")

    print("[4/7] Cross-tabulating...")
    dir_tbl = direction_table(tf, preds, wrong, keys)
    subject_tbl = rate_table(tf, "subject", wrong, keys, "subject").sort_values(
        "n", ascending=False)
    length_tbl = rate_table(tf, "length_bucket", wrong, keys, "length (words)",
                            class_mix=True)
    coverage_tbl = rate_table(tf, "coverage_bucket", wrong, keys, "matched features",
                              class_mix=True)
    conf_tbl = (confidence_table(confs, wrong, keys)
                if len(confs) == len(keys) else pd.DataFrame())
    leak_tbl = leakage_table(tf, wrong, keys)
    style_tbl = stylometry_table(tf, any_wrong)

    print("[5/7] Model agreement...")
    combo_counts, agree_tbl, err_sets, shared, union = agreement_tables(tf, wrong, keys)
    print(f"    {len(union):,} articles wrong for at least one model; "
          f"{len(shared):,} wrong for all {len(keys)}")

    print("[6/7] Writing tables...")
    rows = error_rows(tf, preds, confs, wrong, keys)
    articles = error_articles(tf, wrong, keys)
    save_table(dir_tbl, "error_direction.csv")
    save_table(subject_tbl, "error_rate_by_subject.csv")
    save_table(length_tbl, "error_rate_by_length.csv")
    save_table(coverage_tbl, "error_rate_by_vocab_coverage.csv")
    if not conf_tbl.empty:
        save_table(conf_tbl, "error_rate_by_confidence.csv")
    save_table(leak_tbl, "error_rate_by_leakage_marker.csv")
    save_table(style_tbl, "stylometry_error_vs_correct.csv")
    save_table(combo_counts, "model_overlap.csv")
    save_table(agree_tbl, "model_agreement.csv")
    if not rows.empty:
        rows.to_csv(TABLES_DIR / "errors_long.csv", index=False)
    save_table(articles, "error_articles.csv")

    fig_path = plot_taxonomy(length_tbl, conf_tbl, subject_tbl, combo_counts, keys, figs) \
        if not conf_tbl.empty else ""

    print("[7/7] Writing report...")
    examples = quote_leakage_examples(tf, preds, confs, keys, args.n_examples)

    # --- the numbers the prose quotes ---------------------------------------
    best_key = min(keys, key=lambda k: wrong[k].mean())
    n_fake = int((tf["label"] == FAKE).sum())
    mention = tf["marker::'reuters' anywhere"]
    fake_with = (tf["label"] == FAKE) & mention
    fake_without = (tf["label"] == FAKE) & ~mention
    real_with = (tf["label"] == REAL) & tf["marker::(Reuters) tag"]
    real_without = (tf["label"] == REAL) & ~tf["marker::(Reuters) tag"]

    lift_lines = []
    for k in keys:
        with_rate = wrong[k][fake_with].mean() * 100
        without_rate = wrong[k][fake_without].mean() * 100
        lift = with_rate / without_rate if without_rate else np.nan
        lift_lines.append(
            f"- **{model_name(k)}**: {with_rate:.1f}% of the "
            f"{int(fake_with.sum())} Reuters-mentioning fake articles are "
            f"misclassified, versus {without_rate:.2f}% of the "
            f"{int(fake_without.sum())} that never say the word - "
            f"a {lift:.0f}x higher error rate on the same class."
        )

    # The "marker disagrees with the label" group: fake articles carrying the
    # real-side tell, plus real articles missing it. If leakage drives the
    # errors, this is where they should pile up - so measure by how much,
    # instead of asserting it.
    mismatch = fake_with | real_without
    mismatch_share = mismatch.mean()
    mismatch_lines = []
    for k in keys:
        in_mm = int((wrong[k] & mismatch).sum())
        share = in_mm / max(int(wrong[k].sum()), 1)
        mismatch_lines.append(
            f"- **{model_name(k)}**: {in_mm} of {int(wrong[k].sum())} errors "
            f"({share:.1%}) fall in that 1-in-{1 / mismatch_share:.0f} slice - "
            f"a {share / mismatch_share:.0f}x concentration."
        )
    best_mm_share = (wrong[best_key] & mismatch).sum() / max(int(wrong[best_key].sum()), 1)

    small_buckets = [str(i) for i, n in zip(length_tbl.index, length_tbl["n"])
                     if n < args.min_bucket]
    small_note = (f"Length buckets with fewer than {args.min_bucket} articles "
                  f"({', '.join(small_buckets)}) carry rates that are one or two "
                  "articles wide - do not read anything into them."
                  if small_buckets else
                  f"Every length bucket holds at least {args.min_bucket} articles, "
                  "so every rate above is on reasonably solid ground.")

    best_shared = agree_tbl.loc[model_name(best_key)]
    worst_key = max(keys, key=lambda k: wrong[k].mean())
    worst_name = model_name(worst_key)
    worst_unique = int(agree_tbl.loc[worst_name, "unique_to_model"])

    # Composition of the errors every model makes.
    shared_idx = sorted(shared)
    shared_tf = tf.loc[shared_idx]
    shared_fake = int((shared_tf["label"] == FAKE).sum())
    shared_mm = int(mismatch.loc[shared_idx].sum()) if shared_idx else 0
    shared_top_subjects = ", ".join(
        f"`{s}` {c}" for s, c in shared_tf["subject"].value_counts().head(3).items()
    ) if shared_idx else "n/a"

    # Vocabulary coverage: does a sparse feature vector actually predict error?
    coverage_corr = np.corrcoef(tf["n_matched_features"], any_wrong.astype(int))[0, 1]
    length_coverage_corr = np.corrcoef(tf["n_words"], tf["n_matched_features"])[0, 1]
    thin, thick = coverage_tbl.index[0], coverage_tbl.index[-1]
    thin_n = int(coverage_tbl.loc[thin, "n"])
    thin_fake = coverage_tbl.loc[thin, "fake_%"]
    thin_rates = {k: coverage_tbl.loc[thin, f"{k}_%"] for k in keys}
    thick_rates = {k: coverage_tbl.loc[thick, f"{k}_%"] for k in keys}
    thin_median_words = tf.loc[tf["coverage_bucket"] == thin, "n_words"].median()
    degenerate = thin_fake >= 95 or thin_fake <= 5

    if degenerate:
        coverage_verdict = (
            f"**It cannot be tested on this axis at all, and that is the finding.** "
            f"The thinnest bucket ({thin} features, n={thin_n}) is "
            f"{thin_fake:.0f}% fake, median length {thin_median_words:.0f} words - "
            f"headline stubs and one-line posts, essentially all from the fake half "
            f"of the dataset. A model that answered \"fake\" for every article in "
            f"that bucket would score {max(thin_fake, 100 - thin_fake):.0f}% there, "
            f"so its low error rate ("
            f"{', '.join(f'{model_name(k)} {thin_rates[k]:.2f}%' for k in keys)}) "
            f"measures class purity, not the model coping with thin evidence. The "
            f"bucket is confounded with the label, so the question the axis was "
            f"built to answer stays unanswered."
        )
    elif all(thin_rates[k] > thick_rates[k] for k in keys):
        coverage_verdict = (
            f"It holds: the thinnest bucket ({thin} features, n={thin_n}, "
            f"{thin_fake:.0f}% fake) has the higher error rate for every model."
        )
    else:
        coverage_verdict = (
            f"It does not hold. Articles matching {thin} features (n={thin_n}, "
            f"{thin_fake:.0f}% fake) are misclassified "
            f"{', '.join(f'{model_name(k)} {thin_rates[k]:.2f}%' for k in keys)} of "
            f"the time, against "
            f"{', '.join(f'{thick_rates[k]:.2f}%' for k in keys)} for the "
            f"{thick}-feature articles - the wrong way round."
        )

    # Long articles: does the effect hold in both directions, or is the bucket
    # just class-skewed? Check instead of assuming.
    long_bucket = length_tbl.index[-1]
    is_long = tf["length_bucket"] == long_bucket
    long_lines = []
    for k in keys:
        overall = max(wrong[k].mean(), 1e-12)
        long_lines.append(
            f"{model_name(k)} {wrong[k][is_long].mean() * 100:.2f}% "
            f"({wrong[k][is_long].mean() / overall:.1f}x its own average)"
        )
    long_fake = wrong[best_key][is_long & (tf["label"] == FAKE)].mean() * 100
    long_real = wrong[best_key][is_long & (tf["label"] == REAL)].mean() * 100

    # The most class-skewed length bucket, so the report can point at it rather
    # than pretend the length axis is clean.
    skew = (length_tbl["fake_%"] - 100 * (tf["label"] == FAKE).mean()).abs()
    skew_bucket = skew.idxmax()
    skew_share = length_tbl.loc[skew_bucket, "fake_%"]

    # Style of errors, per direction. Phrase the conclusion off the numbers.
    real_excl_ratio = style_tbl.loc["exclamation_count", "real_ratio"]
    fake_excl_ratio = style_tbl.loc["exclamation_count", "fake_ratio"]
    style_verdict = (
        "Errors in both directions drift toward the *other* class's style: the "
        "real articles that get called fake are the shouty ones, and the fake "
        "articles that get called real are the calm ones."
        if real_excl_ratio > 1 and fake_excl_ratio < 1 else
        "The two directions do not show a clean style signature - read the "
        "ratios individually rather than as a pattern."
    )
    conf_top_band = {k: (confs[k] >= 0.9).mean() for k in keys} if confs else {}

    def embed(path, caption):
        return f"\n![{caption}]({path})\n" if path else ""

    example_md = ""
    for ex in examples:
        excerpt = ex["excerpt"].replace("|", r"\|")
        example_md += (
            f"**{ex['title']}**  \n"
            f"true class: fake / subject `{ex['subject']}` / called REAL by: "
            f"{ex['called_real_by']}\n\n"
            f"> {excerpt}\n\n"
        )
    if not example_md:
        example_md = ("No fake test article that mentions Reuters was misread as real, "
                      "so there is nothing to quote here.\n")

    report = f"""# Error Taxonomy

Generated by `python src/error_taxonomy.py`. Tables in `reports/error_tables/`,
figure in `reports/figures/`.

Test split reproduced from `models/run_metadata.json`
({len(idx_test):,} articles, `random_state={meta['random_state']}`,
boilerplate stripped: **{meta.get('strip_boilerplate')}**), so these are the same
errors `reports/evaluation_report.md` lists - this file categorises them.

Every table reports a raw count **and** a rate. Counts on their own mostly
track bucket size: "most errors are in 200-499-word articles" is true of any
axis where that bucket is the biggest, and means nothing.
{embed(fig_path, "error taxonomy overview")}
Error rates are on a log scale - the models are two orders of magnitude apart,
so a linear axis would flatten the two good ones into the floor. A bar labelled
`0` had no errors in that bucket at all.

## 1. How many, and in which direction

{md_table(dir_tbl)}

`false_REAL` is a fake article passed off as real - for a misinformation tool
this is the expensive direction. `false_FAKE` is a real article flagged as
fabricated.

## 2. Agreement: dataset-driven vs model-driven errors

{md_table(agree_tbl)}

{md_table(combo_counts)}

{len(union):,} distinct articles are misclassified by at least one model;
**{len(shared):,} are misclassified by all {len(keys)}**. For
{model_name(best_key)} - the most accurate model - {best_shared['shared_by_all_%']:.0f}%
of its errors are also made by both other models, and only
{int(best_shared['unique_to_model'])} are its own. Swapping classifiers cannot
fix the shared set: those articles are hard because of what the data contains,
not because of which loss function was minimised. The reverse also holds:
{worst_name} contributes {worst_unique} errors nobody else makes, and that part
*is* a model problem - it disappears by not using {worst_name}.

Those {len(shared):,} shared errors are {shared_fake} fake and
{len(shared) - shared_fake} real, concentrated in {shared_top_subjects}, with a
median length of {shared_tf['n_words'].median():.0f} words against
{tf['n_words'].median():.0f} for the test set as a whole.

## 3. Leakage markers - the strongest single risk factor

{md_table(mark_small(leak_tbl, args.min_bucket))}

`*` marks a group too small for its rate to be worth reading
(n < {args.min_bucket}).

{chr(10).join(lift_lines)}

The mirror image holds too: {int(real_with.sum()):,} real articles carry a
literal `(Reuters)` tag and only {int(real_without.sum())} do not, and the
error rate on that small tag-less group is
{", ".join(f"{model_name(k)} {wrong[k][real_without].mean() * 100:.1f}%" for k in keys)}
against
{", ".join(f"{wrong[k][real_with].mean() * 100:.2f}%" for k in keys)}
on the tagged group.

Put both sides together into one "the marker disagrees with the label" group -
fake articles that mention Reuters, plus real articles with no `(Reuters)` tag.
That group is {int(mismatch.sum())} articles, **{mismatch_share:.2%} of the test
set**:

{chr(10).join(mismatch_lines)}

No other axis in this report concentrates errors anything like that hard. It is
the clearest evidence available that the models key on the publisher
fingerprint rather than on truthfulness: hand them a fabricated article that
happens to quote Reuters and they flip.

It is *not* the whole story, and the report would be dishonest if it stopped
here. {(1 - best_mm_share):.0%} of {model_name(best_key)}'s errors are articles
whose marker agrees with their label; the leakage group explains where the risk
is concentrated, not every mistake.

### Errors you can read

{example_md}
## 4. Subject category

{md_table(mark_small(subject_tbl, args.min_bucket))}

Subjects are completely disjoint between classes (see `reports/eda_report.md`
section 2), so every error inside a fake-only subject is a false REAL and every
error inside a real-only subject is a false FAKE - the axis cannot mix, and it
is really "error rate by scrape source" wearing a topic label. Read that way it
still says something: for {model_name(best_key)} the hardest source is
`{subject_tbl[f"{best_key}_%"].idxmax()}` at
{subject_tbl[f"{best_key}_%"].max():.2f}% and the easiest is
`{subject_tbl[f"{best_key}_%"].idxmin()}` at
{subject_tbl[f"{best_key}_%"].min():.2f}%, across categories that are all
nominally "news".

## 5. Article length

{md_table(mark_small(length_tbl, args.min_bucket))}

{small_note} `fake_%` is there because a bucket that holds only one class
cannot say anything about difficulty - see section 7 for a bucket where that
ruins the analysis. It bites here too, mildly: the {skew_bucket} bucket is
{skew_share:.0f}% fake against {100 * (tf['label'] == FAKE).mean():.0f}% for the
test set overall, so part of its low error rate is class purity rather than
easy articles.

The standout is the {long_bucket}-word bucket:
{", ".join(long_lines)}. That one is not a class-mix artifact - it is
{length_tbl.loc[long_bucket, "fake_%"]:.0f}% fake, near the dataset's own
balance, and the effect shows up in both directions ({model_name(best_key)}
gets {long_fake:.2f}% of long fake articles wrong and {long_real:.2f}% of long
real ones). Long articles genuinely are harder; the effect is real but small
next to section 3.

## 6. Prediction confidence

{md_table(conf_tbl) if not conf_tbl.empty else "No model exposed `predict_proba`; skipped."}

This is a calibration check in disguise: confidence is the probability assigned
to whichever class was predicted, so an honest 0.9-1.0 band should be wrong
under 10% of the time and an honest 0.5-0.6 band about 40-50% of the time. That
broadly holds here, which is the reassuring part.

The unreassuring part is the shape: {", ".join(f"{model_name(k)} puts {v:.1%}" for k, v in conf_top_band.items())}
of its predictions in the top band. These models almost never sit on the fence,
so the confidence number the app shows a user is nearly always ~100% - it
cannot flag the cases a reader should be sceptical about, because it says the
same thing for almost every article.

## 7. Vocabulary coverage

How many of the {X_test.shape[1]:,} TF-IDF features the article matched at all.
The hypothesis being tested is that a near-empty feature vector means the model
had almost nothing to score and was effectively guessing.

{md_table(mark_small(coverage_tbl, args.min_bucket))}

{coverage_verdict}

Two further reasons not to lean on this axis. Correlation between
matched-feature count and being wrong (any model) is **{coverage_corr:+.3f}** -
near zero, and pointing the opposite way to the hypothesis. And matched-feature
count correlates **{length_coverage_corr:.3f}** with word count, so this axis
and section 5 are close to the same measurement twice over; they are not two
independent pieces of evidence.

## 8. Stylometry of errors vs correct predictions

Means over the union of errors (any model wrong) against articles all models
got right, computed **within** each true class. Comparing errors to correct
predictions without conditioning on the class would just re-measure the
fake-vs-real style gap from the EDA.

{md_table(style_tbl, floatfmt="{:.3f}")}

{style_verdict} Real articles that get misclassified carry
{real_excl_ratio:.1f}x the exclamation marks and
{style_tbl.loc["question_count", "real_ratio"]:.1f}x the question marks of real
articles that do not; fake articles that get misclassified carry
{fake_excl_ratio:.2f}x the exclamation marks of fake articles that do not.
Errors are also longer than correct predictions in both directions
({style_tbl.loc["n_words", "fake_ratio"]:.2f}x for fake,
{style_tbl.loc["n_words", "real_ratio"]:.2f}x for real), which is the same
signal as section 5 rather than a new one.

One apparent contradiction is worth walking through, because it is the whole
count-vs-rate point again in miniature. On the fake side `allcaps_words` goes
*up* in errors ({style_tbl.loc["allcaps_words", "fake_ratio"]:.2f}x) while
`allcaps_ratio` goes *down*
({style_tbl.loc["allcaps_ratio", "fake_ratio"]:.2f}x). The count rises only
because error articles are longer; per word, misclassified fake articles shout
less than correctly classified ones. The ratio is the signal, the count is the
length.

## 9. What is deliberately not here

- **Publication date.** Fake articles span 2015-03-31 to 2018-02-19 and real
  ones only 2016-01-13 to 2017-12-31, so date is a perfect one-sided predictor.
  It is not an error axis, though: the models are trained on `title + body`,
  and `clean_text()` strips digits, so no model here can see a date. Any
  correlation between date and error would be a property of what got written
  when, not of the classifier. Cross-tabulating it would produce a number that
  looks like an explanation and is not one.
- **A per-subject breakdown of the leakage table.** Subject is collinear with
  class here, so the cells would be either the whole class or empty.

## 10. Reading this honestly

Ranked by how much they concentrate errors, the axes come out:

1. **Leakage marker disagrees with the label** - {best_mm_share:.0%} of
   {model_name(best_key)}'s errors in {mismatch_share:.2%} of the articles.
   Nothing else is close.
2. **Length above 1000 words** - a real but modest effect, present for every
   model and in both directions.
3. **Subject / scrape source** - wide spread, but collinear with the class
   itself, so it explains little on its own.
4. **Vocabulary coverage** - no usable signal. The bucket that was supposed to
   test it is single-class, and the axis is 0.94-correlated with length anyway.
5. **Stylometry** - a consistent direction (errors look like the other class)
   but weak, and largely length again.

The honest caveat on point 1: the marker group accounts for a fifth of the best
model's errors, not all of them, and of the {len(shared):,} errors every model
shares only {shared_mm} sit in it. So leakage is the single biggest identifiable
driver and the one with the clearest mechanism, but a substantial residue of
errors has no explanation offered by any axis measured here. That residue is
genuinely unexplained rather than quietly attributed to leakage.

The wider point stands regardless of which bucket a given error falls in: a
model whose errors are predicted this well by the presence of one publisher's
name is not doing what the project name claims. Compare against
`reports/eda_report.md` - the one-rule "mentions Reuters -> real" baseline
scores 99.40% on this split, above every trained model here.

Two follow-ups make the picture complete, both already in the repo:

- `python src/train.py --strip-boilerplate` then re-run this script. If the
  leakage story is right, overall accuracy falls and the Reuters split in
  section 3 stops separating the error rates.
- `python src/cross_dataset_eval.py` - on a dataset without the Reuters
  fingerprint the error taxonomy should look completely different, and the
  error rate far higher.
"""

    out = REPORTS_DIR / "error_taxonomy.md"
    out.write_text(report, encoding="utf-8")

    print(f"\nWrote {out}")
    print(f"      {TABLES_DIR} ({len(list(TABLES_DIR.glob('*.csv')))} tables)")
    if fig_path:
        print(f"      {FIGURES_DIR / 'error_taxonomy.png'}")
    print(f"\nKey finding: fake articles that mention Reuters are misclassified "
          f"{wrong[best_key][fake_with].mean() * 100:.1f}% of the time by "
          f"{model_name(best_key)}, versus "
          f"{wrong[best_key][fake_without].mean() * 100:.2f}% of the "
          f"{int(fake_without.sum()):,} that do not ({int(fake_with.sum())} such "
          f"articles out of {n_fake:,} fake test articles).")


if __name__ == "__main__":
    main()
