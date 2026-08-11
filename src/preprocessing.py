"""
Text preprocessing + data loading for the Fake News Detection project.

Deliberately avoids nltk downloads (nltk.download(...)) so the pipeline
runs identically on every teammate's machine without needing an internet
connection at run time - just `pip install -r requirements.txt` and go.

Two things worth knowing before you touch this file:

1. `clean_text()` is what the models were trained on. If you change it,
   you MUST retrain - a vectorizer fitted on differently-cleaned text will
   silently produce garbage features at prediction time.

2. `strip_source_boilerplate()` exists because of a real leakage problem in
   the ISOT dataset: ~99% of the "real" articles carry a Reuters dateline
   ("WASHINGTON (Reuters) - ...") and essentially none of the fake ones do.
   A model can hit 99% accuracy by learning the single token "reuters".
   Run `python src/eda.py` to see the numbers for your copy of the data.
"""

import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOPWORDS = ENGLISH_STOP_WORDS

# --- cleaning regexes ------------------------------------------------------

_URL_RE = re.compile(r"http\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")

# --- source-boilerplate regexes (leakage guard) ----------------------------

_AGENCIES = r"Reuters|Reuters Staff|AP|AFP|Bloomberg|CNN|BBC"

# "WASHINGTON (Reuters) - " / "UNITED NATIONS (Reuters) - " style datelines
_DATELINE_RE = re.compile(
    rf"^\s*[A-Za-z][A-Za-z.\-/'’ ]{{0,60}}?\(\s*(?:{_AGENCIES})\s*\)\s*[-–—]+\s*"
)
# Any leftover "(Reuters)" tag mid-article
_AGENCY_TAG_RE = re.compile(rf"\(\s*(?:{_AGENCIES})\s*\)")
# The bare agency name as a word
_AGENCY_WORD_RE = re.compile(
    r"\b(?:reuters|thomson\s+reuters|associated\s+press|agence\s+france[-\s]presse)\b",
    re.IGNORECASE,
)
# Wire-service sign-offs: "Reporting by X; Editing by Y"
_EDITOR_NOTE_RE = re.compile(
    r"\((?:reporting|writing|editing|additional reporting)[^)]{0,300}\)|"
    r"\b(?:reporting|writing|editing|additional reporting) by [^.;]{0,120}[.;]",
    re.IGNORECASE,
)
# Blog/aggregator tells that are just as leaky on the fake side
_SOCIAL_BOILERPLATE_RE = re.compile(
    r"featured image via[^.]{0,120}\.?|"
    r"image via[^.]{0,120}\.?|"
    r"read more:.*?(?=\s[A-Z]|$)|"
    r"pic\.twitter\.com/\S+|"
    r"via\s+@\w+|"
    r"\b21st century wire\b",
    re.IGNORECASE,
)

# --- stylometric feature regexes (used by EDA / the app's signals panel) ---

_WORD_RE = re.compile(r"\b\w+\b")
_UPPER_WORD_RE = re.compile(r"\b[A-Z]{2,}\b")


def strip_source_boilerplate(text: str) -> str:
    """
    Remove publisher/wire-service fingerprints that leak the label.

    This is not general "cleaning" - it targets the specific artifacts that
    make ISOT trivially separable (Reuters datelines and agency tags on the
    real side, blog-footer cruft on the fake side). Used by
    `train.py --strip-boilerplate` for the honest-accuracy experiment.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = _DATELINE_RE.sub("", text)
    text = _AGENCY_TAG_RE.sub(" ", text)
    text = _EDITOR_NOTE_RE.sub(" ", text)
    text = _SOCIAL_BOILERPLATE_RE.sub(" ", text)
    text = _AGENCY_WORD_RE.sub(" ", text)
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def clean_text(text: str, strip_boilerplate: bool = False) -> str:
    """Lowercase, strip urls/html/punctuation/digits, drop stopwords + short tokens."""
    if not isinstance(text, str) or not text.strip():
        return ""

    if strip_boilerplate:
        text = strip_source_boilerplate(text)

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(tokens)


def text_stats(text: str) -> dict:
    """
    Cheap stylometric descriptors of a raw (uncleaned) article.

    Not fed to the classifier - these are for exploratory analysis and for
    showing the user *why* an input might be unusual (all-caps headlines and
    exclamation marks are classic sensationalism markers).
    """
    if not isinstance(text, str):
        text = ""

    words = _WORD_RE.findall(text)
    n_words = len(words)
    n_chars = len(text)

    return {
        "n_chars": n_chars,
        "n_words": n_words,
        "n_sentences": max(text.count(".") + text.count("!") + text.count("?"), 1),
        "avg_word_len": (sum(len(w) for w in words) / n_words) if n_words else 0.0,
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "quote_count": text.count('"'),
        "allcaps_words": len(_UPPER_WORD_RE.findall(text)),
        "allcaps_ratio": (len(_UPPER_WORD_RE.findall(text)) / n_words) if n_words else 0.0,
        "digit_ratio": (sum(c.isdigit() for c in text) / n_chars) if n_chars else 0.0,
        "upper_char_ratio": (sum(c.isupper() for c in text) / n_chars) if n_chars else 0.0,
    }


def add_text_features(df: pd.DataFrame, column: str = "content") -> pd.DataFrame:
    """Attach one column per `text_stats` key. Returns a new dataframe."""
    stats = pd.DataFrame(list(df[column].map(text_stats)), index=df.index)
    return pd.concat([df, stats], axis=1)


# --- data loading ----------------------------------------------------------

def _safe_column(df: pd.DataFrame, name: str) -> pd.Series:
    """Return df[name] as a string series, or an empty series if absent."""
    if name in df.columns:
        return df[name].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype=str)


def load_data(
    data_dir: str = None,
    strip_boilerplate: bool = False,
    drop_duplicates: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Load Fake.csv and True.csv (ISOT-style dataset:
    https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
    from `data_dir`, label them, and return one combined dataframe.

    Expected columns in each CSV: title, text, subject, date
    Label convention: 0 = fake, 1 = real

    Returns columns: content, label, title, body, subject, date
    (`content` is title + body, which is what the models are trained on.)
    """
    from config import DATA_DIR, FAKE, REAL  # local import keeps this module standalone

    data_dir = Path(data_dir) if data_dir else DATA_DIR
    fake_path = data_dir / "Fake.csv"
    real_path = data_dir / "True.csv"

    missing = [p.name for p in (fake_path, real_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing {', '.join(missing)} inside {data_dir}.\n"
            "Either run `python src/setup_data.py` (extracts the bundled archive.zip),\n"
            "or download the dataset from Kaggle and place both CSVs there:\n"
            "https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset"
        )

    fake_df = pd.read_csv(fake_path)
    real_df = pd.read_csv(real_path)

    fake_df["label"] = FAKE
    real_df["label"] = REAL

    df = pd.concat([fake_df, real_df], ignore_index=True)
    n_raw = len(df)

    df["title"] = _safe_column(df, "title")
    df["body"] = _safe_column(df, "text")
    df["subject"] = _safe_column(df, "subject")
    df["date"] = _safe_column(df, "date").str.strip()
    df["content"] = (df["title"] + " " + df["body"]).str.strip()

    if strip_boilerplate:
        df["content"] = df["content"].map(strip_source_boilerplate)

    # Drop empty rows
    df = df[df["content"].str.len() > 0]
    n_after_empty = len(df)

    # Drop exact duplicates. ISOT has a few thousand - leaving them in leaks
    # train rows into the test split and inflates accuracy.
    if drop_duplicates:
        df = df.drop_duplicates(subset="content")
    n_after_dupes = len(df)

    df = df.reset_index(drop=True)

    if verbose:
        print(f"    raw rows:        {n_raw}")
        print(f"    after empty drop:{n_after_empty:>7}  (-{n_raw - n_after_empty})")
        print(f"    after dedupe:    {n_after_dupes:>7}  (-{n_after_empty - n_after_dupes})")
        print(f"    label counts:    {df['label'].value_counts().to_dict()}")

    return df[["content", "label", "title", "body", "subject", "date"]]
