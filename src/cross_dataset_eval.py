"""
Test how well models trained on one dataset (ISOT) generalize to a
different fake-news dataset (e.g. WELFake), whose articles come from
different sources and have a different writing style.

This is the key overfitting check: a model that scores 95%+ on ISOT's
own test split but drops to ~60% here is mostly memorizing ISOT's
specific writing style/sources, not learning genuine fake-vs-real signal.

WELFake dataset: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification
Expected columns: title, text, label   (label: 0 = fake, 1 = real - same convention as ISOT)

Usage:
    python src/cross_dataset_eval.py --data-file data/WELFake_Dataset.csv
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import clean_text


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate saved models on a second dataset.")
    parser.add_argument("--data-file", required=True, help="Path to the second dataset's CSV (e.g. WELFake_Dataset.csv)")
    parser.add_argument("--models-dir", default="models", help="Folder with saved models from train.py")
    parser.add_argument("--sample", type=int, default=None,
                         help="Optional: evaluate on a random sample of N rows (faster for a quick check)")
    return parser.parse_args()


def load_second_dataset(path: str, sample: int = None) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Be forgiving about column naming across dataset versions
    cols = {c.lower(): c for c in df.columns}
    title_col = cols.get("title")
    text_col = cols.get("text")
    label_col = cols.get("label")

    if label_col is None:
        raise ValueError(
            f"Couldn't find a 'label' column in {path}. Found columns: {list(df.columns)}"
        )

    df["content"] = (
        (df[title_col].fillna("") if title_col else "") + " " +
        (df[text_col].fillna("") if text_col else "")
    ).str.strip()

    df = df.rename(columns={label_col: "label"})
    df = df[df["content"].str.len() > 0].dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=42)

    return df[["content", "label"]].reset_index(drop=True)


def main():
    args = parse_args()
    models_dir = Path(args.models_dir)

    print(f"Loading second dataset from {args.data_file} ...")
    df = load_second_dataset(args.data_file, args.sample)
    print(f"  {len(df)} rows (label counts: {df['label'].value_counts().to_dict()})")

    print("Cleaning text (same preprocessing as training)...")
    df["clean_content"] = df["content"].apply(clean_text)

    vectorizer = joblib.load(models_dir / "tfidf_vectorizer.joblib")
    X = vectorizer.transform(df["clean_content"])
    y_true = df["label"]

    model_files = {
        "Naive Bayes": "naive_bayes_model.joblib",
        "SVM (Linear, calibrated)": "svm_model.joblib",
        "Logistic Regression": "logistic_regression_model.joblib",
    }

    print("\n=== Cross-dataset generalization results ===")
    for name, filename in model_files.items():
        model_path = models_dir / filename
        if not model_path.exists():
            print(f"  Skipping {name}: {filename} not found (train it first).")
            continue

        model = joblib.load(model_path)
        preds = model.predict(X)
        acc = accuracy_score(y_true, preds)

        print(f"\n--- {name} ---")
        print(f"Accuracy on second dataset: {acc:.4f}")
        print(classification_report(y_true, preds, target_names=["fake", "real"]))
        print(f"Confusion matrix:\n{confusion_matrix(y_true, preds)}")

    print(
        "\nCompare these numbers to the accuracy in models/metrics.txt (the original "
        "test-set accuracy). A big drop here means the model overfit to ISOT's specific "
        "writing style/sources rather than learning general fake-vs-real signal."
    )


if __name__ == "__main__":
    main()
