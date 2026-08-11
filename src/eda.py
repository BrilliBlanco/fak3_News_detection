"""
Exploratory data analysis for the news dataset.

Answers the questions you actually need for the report:
  - How balanced are the classes, and how much of the data is duplicated?
  - Do fake and real articles differ in length, punctuation, ALL-CAPS use?
  - Which words separate the two classes most strongly?
  - **Is the dataset leaking the label?** (spoiler: ISOT does, badly)
  - Do the two classes even cover the same topics and time period?

Usage:
    python src/eda.py
    python src/eda.py --sample 5000 --no-figures

Writes:
    reports/eda_report.md
    reports/figures/*.png
    reports/eda_tables/*.csv
"""

import argparse
import re
import textwrap

import matplotlib

matplotlib.use("Agg")  # headless-safe: never tries to open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from config import FAKE, FIGURES_DIR, REAL, REPORTS_DIR, ensure_dirs
from preprocessing import add_text_features, clean_text, load_data

TABLES_DIR = REPORTS_DIR / "eda_tables"

# Phrases whose presence is suspiciously class-specific. If any of these is
# near-perfectly correlated with a label, the model isn't learning "fake vs
# real" - it's learning "which newsroom published this".
LEAKAGE_MARKERS = {
    "(Reuters) tag": r"\(\s*Reuters\s*\)",
    "word 'reuters'": r"\breuters\b",
    "wire dateline (CITY (Agency) -)": r"^[A-Z][A-Za-z.\-/'’ ]{0,40}\(\s*\w+\s*\)\s*[-–—]",
    "'Reporting by ...'": r"\bReporting by\b",
    "'Featured image via'": r"featured image via",
    "'READ MORE:'": r"READ MORE:",
    "pic.twitter.com link": r"pic\.twitter\.com",
    "'21st Century Wire'": r"21st Century Wire",
    "@mention": r"(?:^|\s)@\w+",
    "ALL-CAPS run (4+ words)": r"(?:\b[A-Z]{2,}\b[ ,]+){3}\b[A-Z]{2,}\b",
}

STYLE_COLS = [
    "n_words", "avg_word_len", "n_sentences", "exclamation_count",
    "question_count", "quote_count", "allcaps_words", "allcaps_ratio",
    "digit_ratio", "upper_char_ratio",
]

FAKE_COLOR, REAL_COLOR = "#d1495b", "#2e86ab"


# --- plumbing --------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Exploratory data analysis on the news dataset.")
    p.add_argument("--data-dir", default=None, help="Folder with Fake.csv / True.csv")
    p.add_argument("--sample", type=int, default=None,
                   help="Analyse a random sample of N rows (faster while iterating)")
    p.add_argument("--top-n", type=int, default=20, help="How many top terms per class to show")
    p.add_argument("--no-figures", action="store_true", help="Skip PNG generation, tables only")
    p.add_argument("--keep-duplicates", action="store_true",
                   help="Do not drop duplicate articles (shows the raw scale of the problem)")
    return p.parse_args()


def save_fig(fig, name: str, enabled: bool = True) -> str:
    """Save a figure into reports/figures and return its relative path."""
    if not enabled:
        plt.close(fig)
        return ""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}"


def save_table(df: pd.DataFrame, name: str) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / name, index=True)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
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


# --- individual analyses ---------------------------------------------------

def overview(df: pd.DataFrame, df_with_dupes: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    counts = df["label"].value_counts().sort_index()
    n = len(df)
    table = pd.DataFrame({
        "articles": counts,
        "share": counts / n,
        "median_words": df.groupby("label")["n_words"].median(),
        "mean_words": df.groupby("label")["n_words"].mean(),
        "empty_titles": df.groupby("label")["title"].apply(lambda s: (s.str.len() == 0).sum()),
    })
    table.index = table.index.map({FAKE: "fake", REAL: "real"})
    table.index.name = "class"

    dupes = len(df_with_dupes) - len(df_with_dupes.drop_duplicates(subset="content"))
    note = (
        f"{len(df_with_dupes):,} rows loaded, **{dupes:,} exact duplicate articles** "
        f"({dupes / max(len(df_with_dupes), 1):.1%}). Duplicates must be dropped *before* "
        "the train/test split - otherwise the same article lands on both sides and "
        "test accuracy is inflated."
    )
    return table, note


def plot_class_balance(df, enabled=True):
    counts = df["label"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.bar(["fake", "real"], counts.values, color=[FAKE_COLOR, REAL_COLOR])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("articles")
    ax.set_title("Class balance")
    ax.spines[["top", "right"]].set_visible(False)
    return save_fig(fig, "class_balance.png", enabled)


def subject_breakdown(df: pd.DataFrame, enabled=True):
    ct = pd.crosstab(df["subject"], df["label"])
    ct.columns = [{FAKE: "fake", REAL: "real"}.get(c, c) for c in ct.columns]
    for col in ("fake", "real"):
        if col not in ct:
            ct[col] = 0
    ct = ct[["fake", "real"]].sort_values("fake", ascending=False)
    ct["exclusive_to"] = np.where(ct["real"] == 0, "fake only",
                                  np.where(ct["fake"] == 0, "real only", "mixed"))

    fig, ax = plt.subplots(figsize=(7, max(3, 0.42 * len(ct))))
    y = np.arange(len(ct))
    ax.barh(y, ct["fake"], color=FAKE_COLOR, label="fake")
    ax.barh(y, ct["real"], left=ct["fake"], color=REAL_COLOR, label="real")
    ax.set_yticks(y, ct.index)
    ax.invert_yaxis()
    ax.set_xlabel("articles")
    ax.set_title("Subject category by class")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return ct, save_fig(fig, "subject_by_class.png", enabled)


def plot_length_distribution(df, enabled=True):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    fake_len = df.loc[df.label == FAKE, "n_words"].clip(upper=2000)
    real_len = df.loc[df.label == REAL, "n_words"].clip(upper=2000)

    axes[0].hist(fake_len, bins=60, alpha=0.65, color=FAKE_COLOR, label="fake")
    axes[0].hist(real_len, bins=60, alpha=0.65, color=REAL_COLOR, label="real")
    axes[0].set_xlabel("words per article (clipped at 2000)")
    axes[0].set_ylabel("articles")
    axes[0].set_title("Article length distribution")
    axes[0].legend(frameon=False)

    axes[1].boxplot([fake_len, real_len], showfliers=False)
    axes[1].set_xticks([1, 2], ["fake", "real"])
    axes[1].set_ylabel("words per article")
    axes[1].set_title("Length spread")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    return save_fig(fig, "length_distribution.png", enabled)


def timeline(df: pd.DataFrame, enabled=True):
    parsed = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    ok = parsed.notna()
    coverage = ok.mean()
    tmp = pd.DataFrame({"date": parsed[ok], "label": df.loc[ok, "label"]})
    if tmp.empty:
        return None, coverage, ""

    monthly = (
        tmp.groupby([tmp["date"].dt.to_period("M"), "label"]).size()
        .unstack(fill_value=0).rename(columns={FAKE: "fake", REAL: "real"})
    )
    monthly.index = monthly.index.to_timestamp()

    fig, ax = plt.subplots(figsize=(9, 3.6))
    for col, color in (("fake", FAKE_COLOR), ("real", REAL_COLOR)):
        if col in monthly:
            ax.plot(monthly.index, monthly[col], color=color, label=col, linewidth=1.8)
    ax.set_ylabel("articles per month")
    ax.set_title(f"Publication timeline ({coverage:.0%} of dates parsed)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return monthly, coverage, save_fig(fig, "articles_over_time.png", enabled)


def style_comparison(df: pd.DataFrame, enabled=True):
    grouped = df.groupby("label")[STYLE_COLS].mean().T
    grouped.columns = [{FAKE: "fake", REAL: "real"}.get(c, c) for c in grouped.columns]
    grouped.index.name = "feature"
    # ratio of fake-mean to real-mean, so big divergences are obvious
    grouped["fake/real"] = grouped["fake"] / grouped["real"].replace(0, np.nan)

    plot_cols = ["exclamation_count", "question_count", "quote_count",
                 "allcaps_words", "avg_word_len", "n_sentences"]
    sub = grouped.loc[plot_cols, ["fake", "real"]]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(sub))
    ax.bar(x - 0.2, sub["fake"], 0.4, color=FAKE_COLOR, label="fake")
    ax.bar(x + 0.2, sub["real"], 0.4, color=REAL_COLOR, label="real")
    ax.set_xticks(x, [c.replace("_", "\n") for c in sub.index], fontsize=8)
    ax.set_ylabel("mean per article")
    ax.set_title("Writing-style markers by class")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return grouped, save_fig(fig, "style_features.png", enabled)


def top_discriminative_terms(clean_series: pd.Series, labels: pd.Series,
                             top_n: int = 20, enabled=True):
    """
    Log-count ratio (the NB-SVM statistic): for each term, how much more
    often does it appear in one class than the other, smoothed.
    Far more informative than raw frequency, which just returns stopwords.
    """
    vec = CountVectorizer(min_df=5, max_features=40000, ngram_range=(1, 2), binary=True)
    X = vec.fit_transform(clean_series)
    terms = np.array(vec.get_feature_names_out())

    is_fake = (labels == FAKE).to_numpy()
    p = np.asarray(X[is_fake].sum(axis=0)).ravel() + 1.0
    q = np.asarray(X[~is_fake].sum(axis=0)).ravel() + 1.0
    ratio = np.log((p / p.sum()) / (q / q.sum()))
    doc_freq = np.asarray(X.sum(axis=0)).ravel()

    order = np.argsort(ratio)
    fake_idx, real_idx = order[::-1][:top_n], order[:top_n]

    table = pd.DataFrame({
        "fake_term": terms[fake_idx],
        "fake_log_ratio": ratio[fake_idx],
        "fake_docs": doc_freq[fake_idx],
        "real_term": terms[real_idx],
        "real_log_ratio": ratio[real_idx],
        "real_docs": doc_freq[real_idx],
    })

    fig, axes = plt.subplots(1, 2, figsize=(11, max(4, 0.32 * top_n)))
    axes[0].barh(range(top_n), ratio[fake_idx][::-1], color=FAKE_COLOR)
    axes[0].set_yticks(range(top_n), terms[fake_idx][::-1], fontsize=8)
    axes[0].set_title("Most fake-indicative terms")
    axes[1].barh(range(top_n), -ratio[real_idx][::-1], color=REAL_COLOR)
    axes[1].set_yticks(range(top_n), terms[real_idx][::-1], fontsize=8)
    axes[1].set_title("Most real-indicative terms")
    for ax in axes:
        ax.set_xlabel("|log count ratio|")
        ax.spines[["top", "right"]].set_visible(False)
    return table, save_fig(fig, "top_terms.png", enabled)


def leakage_audit(df: pd.DataFrame, enabled=True):
    """
    For each marker phrase: what share of each class contains it, and how
    accurate would a one-rule classifier ("contains marker -> real") be?
    """
    rows = []
    n_fake = int((df.label == FAKE).sum())
    n_real = int((df.label == REAL).sum())

    for name, pattern in LEAKAGE_MARKERS.items():
        rx = re.compile(pattern, re.MULTILINE)
        hits = df["content"].str.contains(rx, na=False)
        f_rate = hits[df.label == FAKE].mean()
        r_rate = hits[df.label == REAL].mean()

        # One-rule accuracy: predict REAL when the marker is present.
        # Flip the rule if the marker is actually a fake-side tell.
        predict_real = hits if r_rate >= f_rate else ~hits
        rule_acc = (predict_real.astype(int) == df["label"]).mean()

        rows.append({
            "marker": name,
            "in_fake": f_rate,
            "in_real": r_rate,
            "gap": abs(r_rate - f_rate),
            "one_rule_accuracy": rule_acc,
        })

    table = pd.DataFrame(rows).set_index("marker").sort_values("gap", ascending=False)

    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45 * len(table))))
    y = np.arange(len(table))
    ax.barh(y - 0.2, table["in_fake"], 0.4, color=FAKE_COLOR, label="% of fake articles")
    ax.barh(y + 0.2, table["in_real"], 0.4, color=REAL_COLOR, label="% of real articles")
    ax.set_yticks(y, table.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("share of class containing the marker")
    ax.set_title("Label-leakage audit: publisher fingerprints")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    meta = {"n_fake": n_fake, "n_real": n_real}
    return table, meta, save_fig(fig, "leakage_audit.png", enabled)


def vocabulary_overlap(clean_series: pd.Series, labels: pd.Series) -> dict:
    fake_vocab = set(" ".join(clean_series[labels == FAKE]).split())
    real_vocab = set(" ".join(clean_series[labels == REAL]).split())
    shared = fake_vocab & real_vocab
    union = fake_vocab | real_vocab
    return {
        "fake_vocab": len(fake_vocab),
        "real_vocab": len(real_vocab),
        "shared": len(shared),
        "jaccard": len(shared) / len(union) if union else 0.0,
        "fake_only": len(fake_vocab - real_vocab),
        "real_only": len(real_vocab - fake_vocab),
    }


# --- report ----------------------------------------------------------------

def main():
    args = parse_args()
    ensure_dirs()
    figs = not args.no_figures

    print("[1/8] Loading data...")
    raw = load_data(args.data_dir, drop_duplicates=False)
    df = raw if args.keep_duplicates else raw.drop_duplicates(subset="content").reset_index(drop=True)
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        print(f"    sampled down to {len(df):,} rows")
    print(f"    {len(df):,} articles")

    print("[2/8] Computing text statistics...")
    df = add_text_features(df, "content")

    print("[3/8] Class balance + overview...")
    ov_table, dupe_note = overview(df, raw)
    balance_fig = plot_class_balance(df, figs)
    save_table(ov_table, "overview.csv")

    print("[4/8] Subjects, lengths, timeline...")
    subj_table, subj_fig = subject_breakdown(df, figs)
    len_fig = plot_length_distribution(df, figs)
    monthly, date_coverage, time_fig = timeline(df, figs)
    save_table(subj_table, "subject_by_class.csv")

    print("[5/8] Writing-style comparison...")
    style_table, style_fig = style_comparison(df, figs)
    save_table(style_table, "style_features.csv")

    print("[6/8] Cleaning text for term analysis (this is the slow part)...")
    clean_series = df["content"].map(clean_text)

    print("[7/8] Top discriminative terms + vocabulary overlap...")
    terms_table, terms_fig = top_discriminative_terms(clean_series, df["label"], args.top_n, figs)
    vocab = vocabulary_overlap(clean_series, df["label"])
    save_table(terms_table, "top_terms.csv")

    print("[8/8] Leakage audit...")
    leak_table, leak_meta, leak_fig = leakage_audit(df, figs)
    save_table(leak_table, "leakage_audit.csv")

    worst = leak_table.iloc[0]
    exclusive_subjects = int((subj_table["exclusive_to"] != "mixed").sum())

    def embed(path, caption):
        return f"\n![{caption}]({path})\n" if path else ""

    report = f"""# Exploratory Data Analysis - Fake News Dataset

Generated by `python src/eda.py`. Tables live in `reports/eda_tables/`,
figures in `reports/figures/`.

Rows analysed: **{len(df):,}**{" (random sample)" if args.sample else ""}

## 1. Class balance and data quality

{md_table(ov_table)}

{dupe_note}
{embed(balance_fig, "class balance")}

## 2. Topic coverage

{exclusive_subjects} of {len(subj_table)} subject categories appear in **only one class**.
Where that is true, `subject` alone is a perfect predictor - which tells you the
two halves of the dataset were scraped from different places, not sampled from
one population.

{md_table(subj_table, floatfmt="{:.0f}")}
{embed(subj_fig, "subjects by class")}

## 3. Article length

{embed(len_fig, "length distribution")}
Median length: fake {ov_table.loc['fake', 'median_words']:.0f} words,
real {ov_table.loc['real', 'median_words']:.0f} words.

## 4. Timeline

{date_coverage:.0%} of `date` values parsed cleanly.
{embed(time_fig, "articles over time")}

## 5. Writing style

Mean per-article values. `fake/real` > 1 means the marker is more common in fake articles.

{md_table(style_table)}
{embed(style_fig, "style features")}

## 6. Most discriminative terms

Log count ratio - how much more often a term appears in one class than the other.

{md_table(terms_table, floatfmt="{:.2f}")}
{embed(terms_fig, "top terms")}

Vocabulary: {vocab['fake_vocab']:,} distinct tokens in fake articles,
{vocab['real_vocab']:,} in real, {vocab['shared']:,} shared
(Jaccard {vocab['jaccard']:.3f}).

## 7. Label leakage audit  <- read this one

{md_table(leak_table)}
{embed(leak_fig, "leakage audit")}

**Headline finding:** the marker `{worst.name}` appears in
{worst['in_real']:.1%} of real articles and {worst['in_fake']:.1%} of fake ones.
A one-line rule - "contains this marker, therefore real" - scores
**{worst['one_rule_accuracy']:.1%} accuracy** on its own, with no machine
learning involved.

{textwrap.fill(
    "That number is the honest baseline any model here must be compared against. "
    "A TF-IDF classifier reporting ~99% accuracy on this dataset has most likely "
    "learned the publisher fingerprint, not the difference between truthful and "
    "fabricated reporting. Two things follow: (1) retrain with "
    "`python src/train.py --strip-boilerplate` and compare - the accuracy drop is "
    "the portion of performance that was leakage; (2) evaluate on a second, "
    "independently-sourced dataset with `python src/cross_dataset_eval.py`, "
    "because in-distribution accuracy cannot detect this failure at all.", 88)}
"""

    out = REPORTS_DIR / "eda_report.md"
    out.write_text(report, encoding="utf-8")

    print(f"\nWrote {out}")
    print(f"      {TABLES_DIR} ({len(list(TABLES_DIR.glob('*.csv')))} tables)")
    if figs:
        print(f"      {FIGURES_DIR} ({len(list(FIGURES_DIR.glob('*.png')))} figures)")
    print(f"\nKey finding: '{worst.name}' alone classifies "
          f"{worst['one_rule_accuracy']:.1%} of the dataset correctly.")


if __name__ == "__main__":
    main()
