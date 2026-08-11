"""
Test how well models trained on one dataset (ISOT) generalize to a
*different* fake-news dataset (e.g. WELFake), whose articles come from
different sources and have a different writing style.

This is the single most important number in the project. In-distribution
test accuracy on ISOT is ~99%, but `src/eda.py` shows a one-keyword rule
scores the same - so that 99% mostly measures publisher fingerprinting.
Accuracy on an independently-sourced corpus is the only evaluation that
cannot be gamed that way. A drop from 99% to ~60% here does not mean the
code is broken; it means the original number was never real.

WELFake: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
Expected columns: title, text, label   (label: 0 = fake, 1 = real - same as ISOT)

Usage:
    python src/cross_dataset_eval.py --data-file data/WELFake_Dataset.csv
    python src/cross_dataset_eval.py --data-file data/other.csv --sample 5000
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from config import (DATA_DIR, FIGURES_DIR, METRICS_JSON, MODEL_KEYS,
                    MODELS_DIR, REPORTS_DIR, RUN_META, TARGET_NAMES,
                    VECTORIZER_FILE, ensure_dirs, model_name, model_path)
from preprocessing import clean_text, strip_source_boilerplate


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate saved models on a second dataset.")
    p.add_argument("--data-file", required=True,
                   help="CSV of the second dataset (e.g. data/WELFake_Dataset.csv)")
    p.add_argument("--models-dir", default=str(MODELS_DIR), help="Folder with saved models")
    p.add_argument("--sample", type=int, default=None,
                   help="Evaluate on a random sample of N rows (faster for a quick check)")
    p.add_argument("--label-column", default=None, help="Override label column detection")
    p.add_argument("--text-column", default=None, help="Override text column detection")
    return p.parse_args()


def load_second_dataset(path: str, sample: int = None, label_col: str = None,
                        text_col: str = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise SystemExit(
            f"Dataset not found: {path}\n\n"
            "Download WELFake from Kaggle and put the CSV in data/:\n"
            "  https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification\n\n"
            f"Any CSV with title/text/label columns works - point --data-file at it.\n"
            f"CSVs currently in {DATA_DIR}: "
            f"{sorted(p.name for p in DATA_DIR.glob('*.csv')) or 'none'}"
        )

    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    title_col = cols.get("title")
    text_col = text_col or cols.get("text") or cols.get("content")
    label_col = label_col or cols.get("label") or cols.get("class") or cols.get("target")

    if label_col is None:
        raise SystemExit(
            f"No 'label' column in {path.name}. Found: {list(df.columns)}\n"
            "Pass --label-column to name it explicitly."
        )
    if text_col is None and title_col is None:
        raise SystemExit(
            f"No text column in {path.name}. Found: {list(df.columns)}\n"
            "Pass --text-column to name it explicitly."
        )

    parts = []
    if title_col:
        parts.append(df[title_col].fillna("").astype(str))
    if text_col:
        parts.append(df[text_col].fillna("").astype(str))
    df["content"] = parts[0] if len(parts) == 1 else (parts[0] + " " + parts[1])
    df["content"] = df["content"].str.strip()

    df = df.rename(columns={label_col: "label"})
    df = df[df["content"].str.len() > 0].dropna(subset=["label"])
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    valid = df["label"].isin([0, 1])
    if not valid.all():
        print(f"  dropping {int((~valid).sum())} rows with labels outside {{0, 1}}")
        df = df[valid]

    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=42)

    return df[["content", "label"]].reset_index(drop=True)


def in_distribution_scores(models_dir: Path) -> dict:
    """Original ISOT test-set accuracy, for the side-by-side comparison."""
    path = models_dir / METRICS_JSON
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v["accuracy"] for k, v in data.get("models", {}).items()}


def main():
    args = parse_args()
    ensure_dirs()
    models_dir = Path(args.models_dir)

    if not (models_dir / VECTORIZER_FILE).exists():
        raise SystemExit(f"No trained models in {models_dir}. Run `python src/train.py` first.")

    print(f"Loading second dataset from {args.data_file} ...")
    df = load_second_dataset(args.data_file, args.sample, args.label_column, args.text_column)
    print(f"  {len(df):,} rows (label counts: {df['label'].value_counts().to_dict()})")

    # Match whatever preprocessing the models were trained with
    meta_path = models_dir / RUN_META
    stripped = False
    if meta_path.exists():
        stripped = json.loads(meta_path.read_text(encoding="utf-8")).get("strip_boilerplate", False)
    print(f"Cleaning text (same preprocessing as training; "
          f"boilerplate stripping: {stripped})...")
    content = df["content"].map(strip_source_boilerplate) if stripped else df["content"]
    df["clean_content"] = content.map(clean_text)

    usable = df["clean_content"].str.len() > 0
    if not usable.all():
        print(f"  {int((~usable).sum())} rows had no usable text after cleaning - dropped")
        df = df[usable].reset_index(drop=True)

    vectorizer = joblib.load(models_dir / VECTORIZER_FILE)
    X = vectorizer.transform(df["clean_content"])
    y_true = df["label"]

    # How much of this corpus does the ISOT vocabulary even cover?
    coverage = np.asarray((X > 0).sum(axis=1)).ravel()
    print(f"  vocabulary coverage: median {np.median(coverage):.0f} features per article, "
          f"{(coverage < 5).mean():.1%} of articles match fewer than 5")

    original = in_distribution_scores(models_dir)
    rows, report_sections = [], []

    print("\n=== Cross-dataset generalization results ===")
    for key in MODEL_KEYS:
        path = model_path(key, models_dir)
        if not path.exists():
            print(f"  Skipping {model_name(key)}: {path.name} not found (train it first).")
            continue

        model = joblib.load(path)
        preds = model.predict(X)
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, zero_division=0)
        before = original.get(key)

        rows.append({
            "model": model_name(key),
            "isot_accuracy": before,
            "cross_dataset_accuracy": acc,
            "drop": (before - acc) if before is not None else np.nan,
            "cross_dataset_f1": f1,
        })

        print(f"\n--- {model_name(key)} ---")
        if before is not None:
            print(f"ISOT test accuracy:  {before:.4f}")
        print(f"This dataset:        {acc:.4f}"
              + (f"   ({acc - before:+.4f})" if before is not None else ""))
        print(classification_report(y_true, preds, target_names=TARGET_NAMES, zero_division=0))
        cm = confusion_matrix(y_true, preds)
        print(f"Confusion matrix:\n{cm}")

        report_sections.append(
            f"### {model_name(key)}\n\n"
            f"```\n{classification_report(y_true, preds, target_names=TARGET_NAMES, zero_division=0)}\n"
            f"Confusion matrix:\n{cm}\n```\n"
        )

    if not rows:
        raise SystemExit("No trained models found. Run `python src/train.py`.")

    table = pd.DataFrame(rows).set_index("model")

    # Figure: in-distribution vs out-of-distribution accuracy
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(table))
    if table["isot_accuracy"].notna().any():
        ax.bar(x - 0.2, table["isot_accuracy"], 0.4, color="#2e86ab", label="ISOT test split")
    ax.bar(x + 0.2, table["cross_dataset_accuracy"], 0.4, color="#d1495b",
           label=Path(args.data_file).stem)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="coin flip")
    ax.set_xticks(x, [i.replace(" (", "\n(") for i in table.index], fontsize=8)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("In-distribution vs out-of-distribution accuracy")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "cross_dataset.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    worst_drop = table["drop"].max() if table["drop"].notna().any() else None
    md_rows = "\n".join(
        f"| {i} | " + " | ".join(
            "n/a" if pd.isna(v) else f"{v:.4f}" for v in r) + " |"
        for i, r in zip(table.index, table.values)
    )
    report = f"""# Cross-Dataset Generalization Report

Models trained on ISOT, evaluated on `{Path(args.data_file).name}`
({len(df):,} articles{" sampled" if args.sample else ""}).

| model | ISOT accuracy | this dataset | drop | F1 here |
|---|---|---|---|---|
{md_rows}

![cross-dataset accuracy](figures/cross_dataset.png)

## How to read this

The ISOT column is what the models score on data drawn from the same two
sources they were trained on. The second column is what they score on articles
they have never seen the newsroom of.

{"The largest drop is **%.1f percentage points**. " % (worst_drop * 100) if worst_drop is not None else ""}A large drop is the expected outcome, not a bug: `reports/eda_report.md` shows
that a single-keyword rule reproduces almost the entire ISOT accuracy, so most
of what these models learned was which publisher wrote the article. That
signal does not exist outside ISOT, and what remains is the genuinely
transferable part of the model.

Quote **this** number when describing what the system can do.
"""
    out = REPORTS_DIR / "cross_dataset_report.md"
    out.write_text(report, encoding="utf-8")

    print(f"\nWrote {out} and figures/cross_dataset.png")
    print("\nCompare against models/metrics.txt. A big drop means the models leaned on "
          "ISOT-specific writing style and sources rather than general fake-vs-real signal.")


if __name__ == "__main__":
    main()
