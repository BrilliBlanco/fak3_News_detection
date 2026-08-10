"""
Text preprocessing utilities for the Fake News Detection project.

Deliberately avoids nltk downloads (nltk.download(...)) so the pipeline
runs identically on every teammate's machine without needing an internet
connection at run time - just `pip install -r requirements.txt` and go.
"""

import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = ENGLISH_STOP_WORDS

_URL_RE = re.compile(r"http\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip urls/html/punctuation/digits, drop stopwords + short tokens."""
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)


def load_data(data_dir: str) -> pd.DataFrame:
    """
    Load Fake.csv and True.csv (ISOT-style dataset:
    https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
    from `data_dir`, label them, and return one combined dataframe.

    Expected columns in each CSV: title, text, subject, date
    Label convention: 0 = fake, 1 = real
    """
    data_dir = Path(data_dir)
    fake_path = data_dir / "Fake.csv"
    real_path = data_dir / "True.csv"

    if not fake_path.exists() or not real_path.exists():
        raise FileNotFoundError(
            f"Expected {fake_path.name} and {real_path.name} inside {data_dir}.\n"
            "Download the dataset from Kaggle and place both CSVs there:\n"
            "https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset"
        )

    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)

    fake_df["label"] = 0
    real_df["label"] = 1

    df = pd.concat([fake_df, real_df], ignore_index=True)

    # Combine title + body text into one field to classify on
    df["title"] = df.get("title", "").fillna("")
    df["text"] = df.get("text", "").fillna("")
    df["content"] = (df["title"] + " " + df["text"]).str.strip()

    # Drop empty/duplicate rows
    df = df[df["content"].str.len() > 0]
    df = df.drop_duplicates(subset="content").reset_index(drop=True)

    return df[["content", "label"]]
