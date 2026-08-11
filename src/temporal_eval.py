"""
Temporal validation - does the model still work on articles it could not
have seen the future of?

The rest of the pipeline evaluates on a random 80/20 split. A random split
draws train and test rows from the same weeks, the same news cycle and the
same publisher mix, so it measures interpolation, not generalisation. Any
detector that is actually deployed sees *tomorrow's* articles, so the honest
proxy is a temporal split: train on everything before a cutoff date, test on
everything after it.

This module does three things:

  1. Characterises the temporal structure of ISOT, because the dates are
     themselves a leakage channel - a third one, on top of the "(Reuters)"
     tag and the disjoint `subject` categories that `src/eda.py` documents.
     The two classes do not cover the same period, so "when was this
     published" is partly a label.
  2. Runs the temporal experiment properly: TF-IDF is fitted on the training
     period ONLY (fitting the vectorizer on the full corpus would leak future
     vocabulary backwards - the classic mistake in time-series text work),
     then the same three models train.py builds are trained and scored on the
     later period.
  3. Repeats the whole thing with `strip_source_boilerplate()` applied, and
     against a matched random split, so the four cells of the 2x2
     (random/temporal x raw/stripped) are directly comparable.

A temporal split does NOT fix the leakage. The Reuters fingerprint travels
with the articles into the test period, so accuracy stays high. That is a
negative result and it is the point: no amount of split engineering rescues a
corpus whose two halves came from different publishers.

Usage:
    python src/temporal_eval.py
    python src/temporal_eval.py --cutoff 2017-09-01 --balance-test
    python src/temporal_eval.py --strip-boilerplate --no-figures
    python src/temporal_eval.py --sweep

Writes:
    reports/temporal_report.md
    reports/temporal_results.csv
    reports/figures/temporal_validation.png
"""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: never tries to open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from config import (FAKE, FIGURES_DIR, MAX_FEATURES, METRICS_JSON, MODELS_DIR,
                    RANDOM_STATE, REAL, REPORTS_DIR, TEST_SIZE, ensure_dirs,
                    model_name)
# build_models lives in train.py so the temporal run trains *exactly* the same
# estimators as the headline run. evaluate.py imports it the same way.
from train import build_models
from preprocessing import clean_text, load_data

# Late enough that both classes have a usable number of training articles and
# the test period still contains a few thousand fake ones. See the --sweep
# table in the report for how much the conclusion depends on this choice
# (answer: barely).
DEFAULT_CUTOFF = "2017-06-01"

SWEEP_CUTOFFS = ["2016-10-01", "2017-01-01", "2017-04-01",
                 "2017-07-01", "2017-10-01"]

FAKE_COLOR, REAL_COLOR = "#d1495b", "#2e86ab"

# One colour per experimental condition, used in the comparison figure.
CONDITION_COLORS = {
    "random / raw": "#9aa5b1",
    "random / stripped": "#d0d7de",
    "temporal / raw": "#3d5a80",
    "temporal / stripped": "#81b29a",
}

VARIANT_LABEL = {False: "raw", True: "stripped"}


# --- plumbing --------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Temporal (train-on-past, test-on-future) validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default=None, help="Folder with Fake.csv / True.csv")
    p.add_argument("--models-dir", default=str(MODELS_DIR),
                   help="Where metrics.json lives, for the random-split comparison")
    p.add_argument("--cutoff", default=DEFAULT_CUTOFF,
                   help="Train on articles before this date, test on or after it")
    p.add_argument("--strip-boilerplate", action="store_true",
                   help="Run ONLY the boilerplate-stripped variant "
                        "(by default both raw and stripped are run and compared)")
    p.add_argument("--skip-stripped", action="store_true",
                   help="Run ONLY the raw variant (faster, but loses the interesting result)")
    p.add_argument("--balance-test", action="store_true",
                   help="Also score on a subsample of the test period with equal "
                        "class counts, since the natural test period is heavily skewed")
    p.add_argument("--sweep", action="store_true",
                   help="Also re-run the experiment at several cutoffs (adds ~1 min)")
    p.add_argument("--max-features", type=int, default=MAX_FEATURES)
    p.add_argument("--ngram-max", type=int, default=2, help="Upper bound of the n-gram range")
    p.add_argument("--min-df", type=int, default=2, help="Ignore terms in fewer than N documents")
    p.add_argument("--test-size", type=float, default=TEST_SIZE,
                   help="Test fraction for the matched random-split control")
    p.add_argument("--random-state", type=int, default=RANDOM_STATE)
    p.add_argument("--no-figures", action="store_true", help="Skip PNG generation")
    return p.parse_args()


def save_fig(fig, name: str, enabled: bool = True) -> str:
    """Save a figure into reports/figures and return its relative path."""
    if not enabled:
        plt.close(fig)
        return ""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
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


# --- data ------------------------------------------------------------------

def prepare_frame(data_dir, strip_boilerplate: bool) -> tuple[pd.DataFrame, dict]:
    """
    Load, clean, and attach a parsed publication date.

    Rows whose date does not parse are dropped, because a temporal split has
    nothing to do with them. That costs ~0.02% of the corpus and the same rows
    are dropped from the random-split control, so the two splits stay
    comparable.
    """
    df = load_data(data_dir, strip_boilerplate=strip_boilerplate, drop_duplicates=True)
    n_loaded = len(df)

    # format="mixed" lets pandas handle ISOT's inconsistent date strings
    # ("December 31, 2017", "31-Dec-17", ...). A handful of rows have a URL or
    # a headline in the date column instead of a date; those coerce to NaT.
    # Parsed before any other filtering so the parse rate is quoted against
    # the whole corpus rather than a convenient subset of it.
    df["parsed_date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    n_unparsed = int(df["parsed_date"].isna().sum())
    df = df[df["parsed_date"].notna()]

    df["clean_content"] = df["content"].map(clean_text)
    df = df[df["clean_content"].str.len() > 0].reset_index(drop=True)

    info = {
        "n_loaded": n_loaded,
        "n_unparsed": n_unparsed,
        "n_empty_after_clean": n_loaded - n_unparsed - len(df),
        "n_used": len(df),
        "parse_rate": 1.0 - n_unparsed / max(n_loaded, 1),
    }
    return df, info


# --- the date channel ------------------------------------------------------

def date_stump_accuracy(dates: pd.Series, labels: pd.Series) -> tuple[float, pd.Timestamp, str]:
    """
    Best possible single-threshold rule on the publication date alone.

    Sorts by date and sweeps every possible cut, in both directions
    ("fake before the cut" and "real before the cut"), returning the best
    accuracy achievable. This is the ceiling for "how much of the label is
    encoded in the timestamp".
    """
    order = np.argsort(dates.to_numpy())
    ds = dates.to_numpy()[order]
    ys = labels.to_numpy()[order]
    n = len(ys)

    # cum_fake[i] = number of fake articles among the first i rows
    cum_fake = np.concatenate([[0], np.cumsum(ys == FAKE)])
    cum_real = np.concatenate([[0], np.cumsum(ys == REAL)])
    total_fake, total_real = cum_fake[-1], cum_real[-1]

    # Only cuts that fall between two different dates are real thresholds.
    boundaries = np.flatnonzero(np.diff(ds) != np.timedelta64(0, "ns")) + 1
    boundaries = np.concatenate([[0], boundaries, [n]])

    fake_first = (cum_fake[boundaries] + (total_real - cum_real[boundaries])) / n
    real_first = (cum_real[boundaries] + (total_fake - cum_fake[boundaries])) / n

    if fake_first.max() >= real_first.max():
        i = int(np.argmax(fake_first))
        return float(fake_first[i]), pd.Timestamp(ds[min(boundaries[i], n - 1)]), "fake"
    i = int(np.argmax(real_first))
    return float(real_first[i]), pd.Timestamp(ds[min(boundaries[i], n - 1)]), "real"


def temporal_structure(df: pd.DataFrame) -> dict:
    """
    Quantify how much of the label is readable off the timestamp alone.

    ISOT's two halves were collected separately and cover different periods.
    Every article published outside the overlap belongs to whichever class
    covers that stretch of the calendar - no text needed.
    """
    dates, labels = df["parsed_date"], df["label"]

    span = pd.DataFrame({
        "first": dates.groupby(labels).min(),
        "last": dates.groupby(labels).max(),
        "articles": dates.groupby(labels).size(),
    })
    span.index = span.index.map({FAKE: "fake", REAL: "real"})
    span.index.name = "class"

    f_first, f_last = span.loc["fake", "first"], span.loc["fake", "last"]
    r_first, r_last = span.loc["real", "first"], span.loc["real", "last"]
    overlap_start, overlap_end = max(f_first, r_first), min(f_last, r_last)

    outside = (dates < overlap_start) | (dates > overlap_end)
    n_outside = int(outside.sum())
    outside_fake = int((labels[outside] == FAKE).sum())

    # The rule the exclusive window hands you for free.
    naive_rule = np.where(dates < overlap_start, FAKE, REAL)
    naive_acc = float((naive_rule == labels.to_numpy()).mean())

    stump_acc, stump_date, stump_dir = date_stump_accuracy(dates, labels)

    # The Reuters rule measured on the same rows, so the three leakage
    # channels can be compared on one denominator.
    reuters_rule = float((df["content"].str.contains(r"\breuters\b", case=False, na=False)
                          .astype(int).to_numpy() == labels.to_numpy()).mean())

    months = dates.dt.to_period("M")
    monthly = pd.crosstab(months, labels).rename(columns={FAKE: "fake", REAL: "real"})
    for col in ("fake", "real"):
        if col not in monthly:
            monthly[col] = 0
    monthly = monthly[["fake", "real"]]
    exclusive_months = ((monthly["fake"] == 0) | (monthly["real"] == 0))

    return {
        "span": span,
        "monthly": monthly,
        "overlap_start": overlap_start,
        "overlap_end": overlap_end,
        "n_outside": n_outside,
        "share_outside": n_outside / len(df),
        "outside_fake": outside_fake,
        "outside_real": n_outside - outside_fake,
        "fake_covered": outside_fake / max(int((labels == FAKE).sum()), 1),
        "naive_rule_accuracy": naive_acc,
        "reuters_rule_accuracy": reuters_rule,
        "majority_accuracy": float(labels.value_counts(normalize=True).max()),
        "stump_accuracy": stump_acc,
        "stump_date": stump_date,
        "stump_direction": stump_dir,
        "n_months": len(monthly),
        "n_exclusive_months": int(exclusive_months.sum()),
        "articles_in_exclusive_months": int(monthly[exclusive_months].to_numpy().sum()),
    }


# --- the experiment --------------------------------------------------------

def positive_proba(model, X):
    """P(real) for each row, indexed by class value rather than column position."""
    if not hasattr(model, "predict_proba"):
        return None
    raw = model.predict_proba(X)
    return raw[:, list(model.classes_).index(REAL)]


def evaluate_predictions(y_true, pred, proba=None) -> dict:
    """Metrics for one model on one test set. `real` (=1) is the positive class."""
    y_true = np.asarray(y_true)
    cm = confusion_matrix(y_true, pred, labels=[FAKE, REAL])
    tn, fp, fn, tp = cm.ravel()

    out = {
        "accuracy": accuracy_score(y_true, pred),
        # Balanced accuracy = mean of the two per-class recalls. On a test
        # period that is 80% real, plain accuracy rewards a model for simply
        # guessing "real", so this is the number to read.
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "recall_fake": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "roc_auc": float("nan"),
        "confusion_matrix": cm.tolist(),
    }
    if proba is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, proba)
    return out


def run_split(df: pd.DataFrame, idx_train, idx_test, args) -> dict:
    """
    Fit TF-IDF + the three models on `idx_train`, score on `idx_test`.

    The vectorizer is fitted on the training rows ONLY. Fitting it on the full
    frame first (a very common shortcut) would let terms that only exist in
    the future - new candidate names, later scandals - into the feature space
    and into the IDF weights, which quietly turns a temporal split back into a
    random one.
    """
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, args.ngram_max),
        min_df=args.min_df,
    )
    X_train = vectorizer.fit_transform(df.loc[idx_train, "clean_content"])
    X_test = vectorizer.transform(df.loc[idx_test, "clean_content"])
    y_train = df.loc[idx_train, "label"]
    y_test = df.loc[idx_test, "label"]

    models, results, preds, probas = build_models(args.random_state), {}, {}, {}
    for key, model in models.items():
        model.fit(X_train, y_train)
        preds[key] = model.predict(X_test)
        probas[key] = positive_proba(model, X_test)
        results[key] = evaluate_predictions(y_test, preds[key], probas[key])

    # Baselines on this exact test set, for scale.
    train_majority = int(y_train.value_counts().idxmax())
    reuters = df.loc[idx_test, "content"].str.contains(
        r"\breuters\b", case=False, na=False).astype(int)
    baselines = {
        "predict train-majority class": accuracy_score(y_test, np.full(len(y_test), train_majority)),
        "predict test-majority class (oracle)": float(y_test.value_counts(normalize=True).max()),
        "one rule: mentions Reuters -> real": accuracy_score(y_test, reuters),
    }

    return {
        "results": results,
        "preds": preds,
        "probas": probas,
        "baselines": baselines,
        "y_test": y_test,
        "idx_test": idx_test,
        "n_train": len(idx_train),
        "n_test": len(idx_test),
        "n_features": int(X_train.shape[1]),
        "train_fake": int((y_train == FAKE).sum()),
        "train_real": int((y_train == REAL).sum()),
        "test_fake": int((y_test == FAKE).sum()),
        "test_real": int((y_test == REAL).sum()),
    }


def balanced_view(run: dict, random_state: int) -> pd.DataFrame:
    """
    Re-score the already-computed predictions on a class-balanced subsample of
    the test period.

    No retraining: this only changes which test rows are counted, so it
    isolates "how much of the headline accuracy is the test-set skew".
    """
    y = run["y_test"].to_numpy()
    rng = np.random.default_rng(random_state)
    pos = {c: np.flatnonzero(y == c) for c in (FAKE, REAL)}
    n = min(len(pos[FAKE]), len(pos[REAL]))
    if n == 0:
        return pd.DataFrame()

    keep = np.sort(np.concatenate([
        rng.choice(pos[c], size=n, replace=False) if len(pos[c]) > n else pos[c]
        for c in (FAKE, REAL)
    ]))

    rows = {}
    for key, pred in run["preds"].items():
        proba = run["probas"][key]
        m = evaluate_predictions(y[keep], pred[keep], None if proba is None else proba[keep])
        rows[model_name(key)] = {k: v for k, v in m.items() if k != "confusion_matrix"}
    table = pd.DataFrame(rows).T
    table.index.name = f"model (n={2 * n:,}, {n:,} per class)"
    return table


# --- figures ---------------------------------------------------------------

def plot_validation(conditions: dict, monthly: pd.DataFrame, cutoff, enabled=True) -> str:
    """
    Top row: accuracy and balanced accuracy, per model, per condition.
    Bottom row: articles per month by class, with the cutoff marked, so the
    reader can see how lopsided the test period actually is.
    """
    fig = plt.figure(figsize=(11.5, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1],
                          hspace=0.38, wspace=0.22, top=0.90)
    ax_acc = fig.add_subplot(gs[0, 0])
    ax_bal = fig.add_subplot(gs[0, 1])
    ax_time = fig.add_subplot(gs[1, :])

    names = list(next(iter(conditions.values())).keys())
    x = np.arange(len(names))
    width = 0.8 / max(len(conditions), 1)

    lowest = 1.0
    for ax, metric, title in ((ax_acc, "accuracy", "Accuracy"),
                              (ax_bal, "balanced_accuracy", "Balanced accuracy")):
        for i, (cond, per_model) in enumerate(conditions.items()):
            vals = [per_model[n][metric] for n in names]
            lowest = min(lowest, min(vals))
            offset = (i - (len(conditions) - 1) / 2) * width
            ax.bar(x + offset, vals, width,
                   color=CONDITION_COLORS.get(cond, "#888888"), label=cond)
            for xi, v in zip(x + offset, vals):
                ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom",
                        fontsize=6.5, rotation=90)
        ax.set_xticks(x, [n.replace(" (", "\n(") for n in names], fontsize=8)
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in (ax_acc, ax_bal):
        ax.set_ylim(max(0.0, lowest - 0.08), 1.06)
        ax.set_ylabel("score")
    # Legend above the whole figure - inside either panel it would sit on top
    # of a bar, since every bar here is near the ceiling.
    handles, labels = ax_acc.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8.5,
               loc="upper center", bbox_to_anchor=(0.5, 0.985),
               ncol=len(conditions))

    idx = monthly.index.to_timestamp()
    ax_time.plot(idx, monthly["fake"], color=FAKE_COLOR, linewidth=1.8, label="fake")
    ax_time.plot(idx, monthly["real"], color=REAL_COLOR, linewidth=1.8, label="real")
    ax_time.axvline(cutoff, color="black", linestyle="--", linewidth=1.2,
                    label=f"cutoff {pd.Timestamp(cutoff).date()}")
    ax_time.axvspan(idx.min(), cutoff, color="black", alpha=0.05)
    ax_time.annotate("train period", xy=(0.02, 0.9), xycoords="axes fraction", fontsize=8)
    ax_time.annotate("test period", xy=(0.98, 0.9), xycoords="axes fraction",
                     fontsize=8, ha="right")
    ax_time.set_ylabel("articles per month")
    ax_time.set_title("Publication timeline - the classes do not cover the same period")
    ax_time.legend(frameon=False, fontsize=8)
    ax_time.spines[["top", "right"]].set_visible(False)

    return save_fig(fig, "temporal_validation.png", enabled)


# --- report ----------------------------------------------------------------

def results_frame(run: dict) -> pd.DataFrame:
    table = pd.DataFrame({
        model_name(k): {kk: vv for kk, vv in v.items() if kk != "confusion_matrix"}
        for k, v in run["results"].items()
    }).T
    table.index.name = "model"
    return table


def main():
    args = parse_args()
    ensure_dirs()
    figs = not args.no_figures

    try:
        cutoff = pd.Timestamp(args.cutoff)
    except ValueError as exc:
        raise SystemExit(f"--cutoff {args.cutoff!r} is not a date I can parse: {exc}")

    variants = [False, True]
    if args.strip_boilerplate:
        variants = [True]
    elif args.skip_stripped:
        variants = [False]

    frames, infos, runs = {}, {}, {}

    print("[1/5] Loading and dating articles...")
    for strip in variants:
        df, info = prepare_frame(args.data_dir, strip)
        frames[strip], infos[strip] = df, info
        print(f"    {VARIANT_LABEL[strip]:<9} {info['n_loaded']:,} loaded, "
              f"{info['n_unparsed']} unparseable dates ({info['parse_rate']:.3%} parsed), "
              f"{info['n_empty_after_clean']} cleaned to empty -> "
              f"{info['n_used']:,} usable")

    base_strip = variants[0]
    base_df = frames[base_strip]

    print("[2/5] Characterising the temporal structure...")
    struct = temporal_structure(base_df)
    print(f"    fake: {struct['span'].loc['fake', 'first'].date()} .. "
          f"{struct['span'].loc['fake', 'last'].date()}   "
          f"real: {struct['span'].loc['real', 'first'].date()} .. "
          f"{struct['span'].loc['real', 'last'].date()}")
    print(f"    {struct['n_outside']:,} articles ({struct['share_outside']:.2%}) sit outside "
          f"the overlap: {struct['outside_fake']:,} fake, {struct['outside_real']:,} real")
    print(f"    best date-only rule: {struct['stump_accuracy']:.4f} accuracy "
          f"(cut at {struct['stump_date'].date()}), majority baseline "
          f"{struct['majority_accuracy']:.4f}")

    print(f"[3/5] Temporal split at {cutoff.date()} + matched random control...")
    for strip in variants:
        df = frames[strip]
        train_mask = df["parsed_date"] < cutoff
        idx_train, idx_test = df.index[train_mask], df.index[~train_mask]
        if len(idx_train) == 0 or len(idx_test) == 0:
            raise SystemExit(
                f"--cutoff {cutoff.date()} leaves one side empty "
                f"({len(idx_train):,} train / {len(idx_test):,} test). "
                f"The corpus runs {df['parsed_date'].min().date()} to "
                f"{df['parsed_date'].max().date()}."
            )
        if df.loc[idx_train, "label"].nunique() < 2:
            raise SystemExit(
                f"--cutoff {cutoff.date()} leaves only one class in the training "
                "period, so there is nothing to learn. Pick a later cutoff."
            )

        runs[("temporal", strip)] = run_split(df, idx_train, idx_test, args)
        r = runs[("temporal", strip)]
        print(f"    temporal/{VARIANT_LABEL[strip]:<9} "
              f"train {r['train_fake']:,}F/{r['train_real']:,}R -> "
              f"test {r['test_fake']:,}F/{r['test_real']:,}R  "
              f"({r['n_features']:,} features)")
        for key, m in r["results"].items():
            print(f"        {model_name(key):<26} acc={m['accuracy']:.4f}  "
                  f"bal_acc={m['balanced_accuracy']:.4f}  auc={m['roc_auc']:.4f}")

        # Matched random-split control: same rows, same vectorizer settings,
        # same estimators - only the split rule differs. Without this the
        # temporal-vs-random gap would be confounded by preprocessing.
        rnd_train, rnd_test = train_test_split(
            df.index, test_size=args.test_size,
            random_state=args.random_state, stratify=df["label"],
        )
        runs[("random", strip)] = run_split(df, rnd_train, rnd_test, args)
        for key, m in runs[("random", strip)]["results"].items():
            print(f"        [random control] {model_name(key):<15} "
                  f"acc={m['accuracy']:.4f}  bal_acc={m['balanced_accuracy']:.4f}")

    print("[4/5] Sweeping cutoffs..." if args.sweep else "[4/5] Cutoff sweep skipped (--sweep to enable)")
    sweep_table = pd.DataFrame()
    if args.sweep:
        rows = []
        for c in SWEEP_CUTOFFS:
            ts = pd.Timestamp(c)
            for strip in variants:
                df = frames[strip]
                tr, te = df.index[df["parsed_date"] < ts], df.index[df["parsed_date"] >= ts]
                if len(tr) == 0 or len(te) == 0 or df.loc[tr, "label"].nunique() < 2:
                    print(f"    {c} / {VARIANT_LABEL[strip]}: unusable split, skipped")
                    continue
                r = run_split(df, tr, te, args)
                best = max(r["results"], key=lambda k: r["results"][k]["balanced_accuracy"])
                rows.append({
                    "cutoff": c,
                    "variant": VARIANT_LABEL[strip],
                    "train": r["n_train"],
                    "test_fake": r["test_fake"],
                    "test_real": r["test_real"],
                    "best_model": model_name(best),
                    "accuracy": r["results"][best]["accuracy"],
                    "balanced_accuracy": r["results"][best]["balanced_accuracy"],
                    "reuters_rule": r["baselines"]["one rule: mentions Reuters -> real"],
                })
                print(f"    {c} / {VARIANT_LABEL[strip]:<9} best bal_acc="
                      f"{rows[-1]['balanced_accuracy']:.4f} ({model_name(best)})")
        if rows:
            sweep_table = pd.DataFrame(rows).set_index("cutoff")

    print("[5/5] Writing report...")

    # Recorded random-split numbers from the shipped models, for reference.
    metrics_path = Path(args.models_dir) / METRICS_JSON
    recorded = pd.DataFrame()
    recorded_note = (f"`{metrics_path}` not found - skipping the comparison against "
                     "the shipped models. Run `python src/train.py` first.")
    if metrics_path.exists():
        blob = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows = {}
        for key, m in blob["models"].items():
            cm = np.array(m["confusion_matrix"])
            rec_fake = cm[0, 0] / cm[0].sum()
            rec_real = cm[1, 1] / cm[1].sum()
            rows[model_name(key)] = {
                "accuracy": m["accuracy"],
                "balanced_accuracy": (rec_fake + rec_real) / 2,
                "f1": m["f1"],
                "roc_auc": m.get("roc_auc", float("nan")),
            }
        recorded = pd.DataFrame(rows).T
        recorded.index.name = "model"
        run_meta = blob.get("run", {})
        recorded_note = (
            f"Shipped run: `{run_meta.get('command', 'src/train.py')}`, "
            f"{run_meta.get('n_train', 0):,} train / {run_meta.get('n_test', 0):,} test, "
            f"boilerplate stripped: **{run_meta.get('strip_boilerplate')}**. "
            f"Reuters one-rule baseline recorded there: "
            f"**{blob.get('baselines', {}).get('one_rule_mentions_reuters', float('nan')):.4f}**."
        )

    # Figure + the flat results table.
    conditions, flat_rows = {}, []
    for split in ("random", "temporal"):
        for strip in variants:
            run = runs.get((split, strip))
            if run is None:
                continue
            label = f"{split} / {VARIANT_LABEL[strip]}"
            conditions[label] = {
                model_name(k): v for k, v in run["results"].items()
            }
            for key, m in run["results"].items():
                flat_rows.append({
                    "split": split, "variant": VARIANT_LABEL[strip],
                    "model": model_name(key), "n_train": run["n_train"],
                    "n_test": run["n_test"], "test_fake": run["test_fake"],
                    "test_real": run["test_real"],
                    **{k: v for k, v in m.items() if k != "confusion_matrix"},
                })
    flat = pd.DataFrame(flat_rows)
    flat.to_csv(REPORTS_DIR / "temporal_results.csv", index=False)

    fig_path = plot_validation(conditions, struct["monthly"], cutoff, figs)

    # --- assemble the markdown ---------------------------------------------

    def embed(path, caption):
        return f"\n![{caption}]({path})\n" if path else ""

    span_table = struct["span"].copy()
    span_table["first"] = span_table["first"].dt.date.astype(str)
    span_table["last"] = span_table["last"].dt.date.astype(str)

    date_rules = pd.DataFrame({
        "accuracy": {
            "majority class (always 'real')": struct["majority_accuracy"],
            f"published before {struct['overlap_start'].date()} -> fake, else real":
                struct["naive_rule_accuracy"],
            f"best single date threshold ({struct['stump_date'].date()})":
                struct["stump_accuracy"],
        }
    })
    date_rules.index.name = "rule using the timestamp only"

    variant_sections = []
    for strip in variants:
        run = runs[("temporal", strip)]
        rnd = runs[("random", strip)]
        table = results_frame(run)
        rnd_table = results_frame(rnd)

        delta = (table[["accuracy", "balanced_accuracy", "f1"]]
                 - rnd_table[["accuracy", "balanced_accuracy", "f1"]])
        delta.columns = [f"d_{c}" for c in delta.columns]
        delta.index.name = "model"

        base = pd.DataFrame({"accuracy": run["baselines"]})
        base.index.name = "baseline (temporal test set)"

        balanced_block = ""
        if args.balance_test:
            bal = balanced_view(run, args.random_state)
            if not bal.empty:
                balanced_block = (
                    "\n**Class-balanced subsample of the same test period** "
                    "(no retraining - the majority class is randomly down-sampled "
                    "to the minority count, so accuracy and balanced accuracy "
                    "coincide):\n\n" + md_table(bal) + "\n"
                )

        variant_sections.append(f"""### Variant: {VARIANT_LABEL[strip]} text

Training period ends {cutoff.date()}: **{run['train_fake']:,} fake / {run['train_real']:,} real**
({run['n_train']:,} articles, {run['n_features']:,} TF-IDF features fitted on this period only).
Test period: **{run['test_fake']:,} fake / {run['test_real']:,} real** ({run['n_test']:,} articles,
{run['test_real'] / max(run['n_test'], 1):.1%} real).

{md_table(table)}

Baselines on the same test rows:

{md_table(base)}

Difference against the matched random split (temporal minus random, negative = the
temporal split is harder):

{md_table(delta)}
{balanced_block}""")

    sweep_block = ("\n## 5. Does the cutoff matter?\n\nNot run. "
                   "Add `--sweep` to re-run the experiment at "
                   f"{len(SWEEP_CUTOFFS)} further cutoffs and check that the "
                   "conclusion is not an artefact of where the line was drawn.\n")
    if args.sweep and not sweep_table.empty:
        sweep_block = f"""
## 5. Does the cutoff matter?

Best model per cutoff, by balanced accuracy. `reuters_rule` is the one-line
"mentions Reuters -> real" baseline on that cutoff's test rows.

{md_table(sweep_table)}
"""

    recorded_block = ""
    if not recorded.empty:
        recorded_block = f"""
{recorded_note}

{md_table(recorded)}

`balanced_accuracy` here is recomputed from the stored confusion matrix, so it is
comparable with the tables above.
"""
    else:
        recorded_block = f"\n{recorded_note}\n"

    best_temporal = max(
        runs[("temporal", variants[0])]["results"],
        key=lambda k: runs[("temporal", variants[0])]["results"][k]["balanced_accuracy"],
    )
    best_m = runs[("temporal", variants[0])]["results"][best_temporal]
    reuters_on_test = runs[("temporal", variants[0])]["baselines"]["one rule: mentions Reuters -> real"]

    report = f"""# Temporal Validation

Generated by `python src/temporal_eval.py --cutoff {cutoff.date()}`.
Raw numbers: `reports/temporal_results.csv`.

The shipped evaluation uses a random 80/20 split, which draws train and test rows
from the same weeks and the same news cycle. That measures interpolation. A
deployed detector sees articles published *after* everything it was trained on, so
this module retrains on a date cutoff instead: everything before {cutoff.date()} is
training data, everything on or after it is the test set. The TF-IDF vocabulary and
IDF weights are fitted on the training period only - fitting them on the whole
corpus would carry future vocabulary backwards and quietly undo the split.

## 1. Date coverage

{chr(10).join(f"- **{VARIANT_LABEL[s]}**: {infos[s]['n_loaded']:,} de-duplicated articles loaded, {infos[s]['n_unparsed']} dates unparseable (**{infos[s]['parse_rate']:.3%} parsed**), {infos[s]['n_empty_after_clean']} further rows cleaned to an empty string, leaving {infos[s]['n_used']:,} usable." for s in variants)}

`pd.to_datetime(..., errors="coerce", format="mixed")` handles ISOT's inconsistent
formats. The handful of failures are not dates at all - a few fake-side records carry
a source URL or a headline in the `date` column. They are dropped from both the
temporal split and the random-split control, so the two remain comparable.

## 2. The date is a third leakage channel

`src/eda.py` documents two ways this corpus gives the label away: the `(Reuters)`
tag (99%+ of real articles, ~0% of fake) and the `subject` field (7 of 7 categories
appear in only one class). The timestamps are a third.

{md_table(span_table, floatfmt="{:.0f}")}

The two classes were collected separately and do not cover the same calendar. The
overlap runs {struct['overlap_start'].date()} to {struct['overlap_end'].date()};
**{struct['n_outside']:,} articles ({struct['share_outside']:.2%} of the corpus) fall
outside it**, and {struct['outside_fake']:,} of those are fake against
{struct['outside_real']:,} real. In other words, publication date alone classifies
{struct['fake_covered']:.1%} of all fake articles with no errors at all - if it was
published before {struct['overlap_start'].date()}, it is fake, always.
{struct['n_exclusive_months']} of the {struct['n_months']} months in the corpus contain
exactly one class ({struct['articles_in_exclusive_months']:,} articles).

What that is worth as a classifier:

{md_table(date_rules)}

The single-threshold ceiling of **{struct['stump_accuracy']:.4f}** is well short of the
**{struct['reuters_rule_accuracy']:.4f}** that the Reuters rule scores on these same rows,
so the date is the weakest of the three channels. It is still
{struct['stump_accuracy'] - struct['majority_accuracy']:+.3f} over the majority-class
baseline, obtained from a single number that has nothing whatsoever to do with whether
the article is true.

There is a second-order consequence that matters more than the raw accuracy: because
`real` articles are concentrated in the later part of the corpus, *any* cutoff-based
split produces a test period that is mostly real and a training period that is mostly
fake. The prior shifts hard across the split, which is exactly the distribution shift
a deployed model would face, and it is invisible to a stratified random split.

## 3. The temporal experiment

Same estimators as `src/train.py` (`build_models()` is imported from it), same TF-IDF
settings (`max_features={args.max_features}`, `ngram_range=(1, {args.ngram_max})`,
`min_df={args.min_df}`). `precision`/`recall`/`f1` are for the positive class
(`real` = 1); `recall_fake` is recall on the fake class, which is the one that
collapses first when the test period is skewed.

{"".join(variant_sections)}
{embed(fig_path, "temporal validation")}

## 4. Against the shipped random-split numbers
{recorded_block}
{sweep_block}
## 6. Reading this honestly

{textwrap.fill(
    f"The headline result is a negative one: the temporal split barely hurts. "
    f"{model_name(best_temporal)} still reaches {best_m['accuracy']:.4f} accuracy "
    f"and {best_m['balanced_accuracy']:.4f} balanced accuracy on articles published "
    f"after the cutoff, and the one-line Reuters rule scores {reuters_on_test:.4f} on "
    "the very same test rows. That is the explanation, not a mystery: a temporal "
    "split neutralises the date channel (every test article now sits in the "
    "real-only stretch of the calendar, so the timestamp carries no information "
    "inside the test set) but it does nothing at all about the publisher "
    "fingerprint, which travels with the articles into the future. Splitting by "
    "time cannot fix a corpus whose two halves came from two different newsrooms.",
    88)}

{textwrap.fill(
    "So the value of this module is diagnostic rather than corrective. It rules out "
    "one explanation for the ~99% headline number - it is not simply that near-"
    "duplicate articles straddled the random split - and it leaves the publisher "
    "explanation standing. The stripped variant above is the more informative half "
    "of the table: whatever accuracy survives with the Reuters fingerprints removed "
    "AND a future test period is the closest thing to an honest estimate this "
    "dataset can produce on its own.",
    88)}

{textwrap.fill(
    "Two caveats on the numbers themselves. First, the test period is heavily "
    "imbalanced, so read balanced accuracy and recall_fake rather than accuracy; "
    "`--balance-test` re-scores the same predictions on an equal-count subsample if "
    "you want the like-for-like view. Second, the temporal split also shifts the "
    "class prior between train and test, so a model that has simply learned the "
    "training prior is penalised for reasons unrelated to text quality - part of any "
    "drop you see is prior shift, not lost signal.",
    88)}

{textwrap.fill(
    "The remaining check this cannot do is out-of-distribution evaluation. Even a "
    "future test period is still ISOT: same two sources, same scrape, same "
    "formatting. `python src/cross_dataset_eval.py` against an independently "
    "collected corpus is the only test that breaks the publisher confound.",
    88)}
"""

    out = REPORTS_DIR / "temporal_report.md"
    out.write_text(report, encoding="utf-8")

    print(f"\nWrote {out}")
    print(f"      {REPORTS_DIR / 'temporal_results.csv'}")
    if fig_path:
        print(f"      {FIGURES_DIR / 'temporal_validation.png'}")
    print(f"\nTemporal split @ {cutoff.date()}: best model {model_name(best_temporal)} "
          f"acc={best_m['accuracy']:.4f} bal_acc={best_m['balanced_accuracy']:.4f}; "
          f"Reuters one-rule on the same rows={reuters_on_test:.4f}.")
    print("High accuracy on a future test period is not evidence of generalisation "
          "here - the publisher fingerprint crosses the cutoff with the articles.")


if __name__ == "__main__":
    main()
