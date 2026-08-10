"""
Train TF-IDF + Naive Bayes and TF-IDF + SVM models for fake news detection.

Usage:
    python src/train.py --data-dir data --models-dir models

Produces (in --models-dir):
    tfidf_vectorizer.joblib
    naive_bayes_model.joblib
    svm_model.joblib
    metrics.txt   (accuracy / precision / recall / F1 for both models)
"""

import argparse
import time
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from preprocessing import clean_text, load_data


def parse_args():
    parser = argparse.ArgumentParser(description="Train fake news detection models.")
    parser.add_argument("--data-dir", default="data", help="Folder containing Fake.csv and True.csv")
    parser.add_argument("--models-dir", default="models", help="Where to save trained models")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading data...")
    df = load_data(args.data_dir)
    print(f"    {len(df)} articles loaded (label counts: {df['label'].value_counts().to_dict()})")

    print("[2/5] Cleaning text...")
    t0 = time.time()
    df["clean_content"] = df["content"].apply(clean_text)
    print(f"    done in {time.time() - t0:.1f}s")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_content"], df["label"],
        test_size=args.test_size, random_state=args.random_state, stratify=df["label"],
    )

    print("[3/5] Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("[4/5] Training models...")

    # --- Naive Bayes (gives predict_proba out of the box) ---
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    nb_preds = nb_model.predict(X_test_tfidf)

    # --- SVM (LinearSVC has no predict_proba, so wrap it in a calibrator
    #     to get real confidence scores instead of raw decision-function values) ---
    base_svm = LinearSVC(random_state=args.random_state, max_iter=5000)
    svm_model = CalibratedClassifierCV(base_svm, cv=3)
    svm_model.fit(X_train_tfidf, y_train)
    svm_preds = svm_model.predict(X_test_tfidf)

    # --- Logistic Regression (3rd baseline for comparison in the report) ---
    lr_model = LogisticRegression(max_iter=1000, random_state=args.random_state)
    lr_model.fit(X_train_tfidf, y_train)
    lr_preds = lr_model.predict(X_test_tfidf)

    print("[5/5] Evaluating + saving...")
    report_lines = []
    for name, preds in [
        ("Naive Bayes", nb_preds),
        ("SVM (Linear, calibrated)", svm_preds),
        ("Logistic Regression", lr_preds),
    ]:
        acc = accuracy_score(y_test, preds)
        report_lines.append(f"\n=== {name} ===")
        report_lines.append(f"Accuracy: {acc:.4f}")
        report_lines.append(classification_report(y_test, preds, target_names=["fake", "real"]))
        report_lines.append(f"Confusion matrix:\n{confusion_matrix(y_test, preds)}")
        print(f"    {name} accuracy: {acc:.4f}")

    (models_dir / "metrics.txt").write_text("\n".join(report_lines), encoding="utf-8")

    joblib.dump(vectorizer, models_dir / "tfidf_vectorizer.joblib")
    joblib.dump(nb_model, models_dir / "naive_bayes_model.joblib")
    joblib.dump(svm_model, models_dir / "svm_model.joblib")
    joblib.dump(lr_model, models_dir / "logistic_regression_model.joblib")

    print(f"\nSaved models + metrics.txt to {models_dir.resolve()}")


if __name__ == "__main__":
    main()
