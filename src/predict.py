"""
Predict fake/real for a piece of news text, with a confidence score.

Usage:
    python src/predict.py --text "Some headline or article body..."
    python src/predict.py --file path/to/article.txt
    python src/predict.py --text "..." --model svm      (default: nb)
"""

import argparse
from pathlib import Path

import joblib

from preprocessing import clean_text


def parse_args():
    parser = argparse.ArgumentParser(description="Predict fake/real news with confidence score.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Raw text to classify")
    group.add_argument("--file", type=str, help="Path to a .txt file to classify")
    parser.add_argument("--models-dir", default="models", help="Folder with saved models")
    parser.add_argument("--model", choices=["nb", "svm", "lr"], default="nb", help="Which model to use")
    return parser.parse_args()


def main():
    args = parse_args()
    models_dir = Path(args.models_dir)

    raw_text = args.text if args.text else Path(args.file).read_text(encoding="utf-8")

    vectorizer = joblib.load(models_dir / "tfidf_vectorizer.joblib")
    model_files = {
        "nb": "naive_bayes_model.joblib",
        "svm": "svm_model.joblib",
        "lr": "logistic_regression_model.joblib",
    }
    model_file = model_files[args.model]
    model = joblib.load(models_dir / model_file)

    cleaned = clean_text(raw_text)
    if not cleaned:
        print("Input text had no usable content after cleaning - can't classify.")
        return

    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]

    label = "REAL" if pred == 1 else "FAKE"
    confidence = proba[pred] * 100

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}%")


if __name__ == "__main__":
    main()
