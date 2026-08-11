"""
Data quality and profiling framework for the news dataset.

The modelling side of this project is well covered; the *inputs* were not
checked at all. Everything downstream - the vectorizer, the three models, the
accuracy numbers in the report - inherits whatever is wrong with
`data/Fake.csv` and `data/True.csv`. This module is the missing layer: it
profiles both source files, runs a fixed set of named checks against an
explicit schema, and produces a pass/warn/fail verdict per check plus one
overall score.

It answers five questions a reviewer would actually ask:

  - Does the data match the schema we think it matches?   (SCHEMA)
  - How much of it is missing, empty, or duplicated?       (COMPLETENESS, UNIQUENESS)
  - How much of it is malformed?                           (VALIDITY)
  - Do the labels agree with the metadata?                 (CONSISTENCY)
  - Are Fake.csv and True.csv samples of one population?   (DRIFT)
  - Which exact bytes were used for this run?              (INTEGRITY)

The last two matter most here. This is the ISOT dataset, and the two halves
were scraped from different places at different times: they share a schema and
almost nothing else. That is not a modelling nuance, it is a data defect, and
it is why the trained models look far better than they are. The drift section
quantifies it instead of asserting it.

`--strict` makes the script exit non-zero when any check FAILs, so it can sit
in front of `train.py` in CI as a gate. On the dataset as shipped it *will*
exit 1 - that is the correct behaviour, not a bug.

Usage:
    python src/data_quality.py
    python src/data_quality.py --strict
    python src/data_quality.py --no-figures --near-dup-sample 3000

Writes:
    reports/data_quality_report.md   (human)
    reports/data_quality.json        (machine-readable, for a pipeline gate)
    reports/figures/data_quality.png
"""

import argparse
import hashlib
import json
import re
import sys
import textwrap
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: never tries to open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from sklearn.feature_extraction.text import CountVectorizer

from config import DATA_DIR, FAKE, FIGURES_DIR, REAL, REPORTS_DIR, ensure_dirs
from preprocessing import add_text_features, clean_text, load_data

# --- the contract ----------------------------------------------------------

# Data dictionary. This is the thing the CSVs are checked *against*; if the
# upstream file ever changes shape, this is the single place to update, and
# the schema check will tell you it happened rather than the models silently
# training on something else.
EXPECTED_SCHEMA = {
    "title": {
        "dtype": "string",
        "nullable": False,
        "description": "Article headline as scraped.",
        "constraints": {"min_len": 1, "max_len": 600, "no_urls": True},
    },
    "text": {
        "dtype": "string",
        "nullable": False,
        "description": "Article body. Concatenated with `title` to form the "
                       "`content` field the models are trained on.",
        "constraints": {"min_words": 20, "max_words": 6000},
    },
    "subject": {
        "dtype": "string",
        "nullable": False,
        "description": "Publisher-assigned topic category. Metadata only - "
                       "never used as a feature, because it leaks the label.",
        "constraints": {"categorical": True, "max_cardinality": 12},
    },
    "date": {
        "dtype": "string (free-text date)",
        "nullable": False,
        "description": "Publication date. Stored as free text in several "
                       "formats ('December 31, 2017', '31-Dec-17'), so it is "
                       "parsed leniently rather than typed.",
        "constraints": {"parseable": True, "min": "2015-01-01", "max": "2018-12-31"},
    },
}

# filename -> label it is supposed to carry. Used for the per-file profile and
# to check that the file/label mapping is what the loader assumes.
SOURCE_FILES = {"Fake.csv": FAKE, "True.csv": REAL}

# Subject values seen when this module was written. A new value is not
# automatically wrong, but it should be noticed.
KNOWN_SUBJECTS = {
    "News", "politics", "left-news", "Government News", "US_News", "Middle-east",
    "politicsNews", "worldnews",
}

# Every threshold that turns a number into a verdict, in one place, and copied
# into the JSON output so a reader can disagree with the calibration without
# reverse-engineering the code.
THRESHOLDS = {
    "empty_rate_warn": 0.001,
    "empty_rate_fail": 0.05,
    "dup_rate_warn": 0.01,
    "dup_rate_fail": 0.05,
    "dup_title_rate_warn": 0.02,
    "dup_title_rate_fail": 0.10,
    "near_dup_rate_warn": 0.01,
    "near_dup_rate_fail": 0.05,
    "unparseable_date_rate_warn": 0.0,
    "unparseable_date_rate_fail": 0.01,
    "encoding_bad_rate_warn": 0.001,
    "encoding_bad_rate_fail": 0.01,
    "short_article_rate_warn": 0.01,
    "short_article_rate_fail": 0.05,
    "ks_stat_warn": 0.10,
    "ks_stat_fail": 0.25,
    "js_distance_warn": 0.30,
    "js_distance_fail": 0.50,
    "subject_purity_warn": 0.80,   # share of subjects exclusive to one class
    "subject_purity_fail": 0.99,
    "single_class_month_warn": 0.05,
    "single_class_month_fail": 0.20,
    "marker_class_gap_warn": 0.20,  # |P(marker|real) - P(marker|fake)|
    "marker_class_gap_fail": 0.50,
}

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"
STATUS_SCORE = {PASS: 1.0, WARN: 0.5, FAIL: 0.0}  # INFO is excluded from scoring
STATUS_COLOR = {PASS: "#2a9d8f", WARN: "#e9c46a", FAIL: "#d1495b", INFO: "#8d99ae"}
STATUS_ORDER = {FAIL: 0, WARN: 1, INFO: 2, PASS: 3}

FAKE_COLOR, REAL_COLOR = "#d1495b", "#2e86ab"

# Characters that indicate a decoding accident rather than a real character.
# Written as \u escapes on purpose: the literal glyphs are invisible in an
# editor, which is exactly the property that makes them worth checking for.
_REPLACEMENT_RE = re.compile("\ufffd")  # the "could not decode this byte" glyph
# UTF-8 bytes re-read as latin-1 - "don't" becomes "don\u00e2\u20ac\u2122t".
_MOJIBAKE_RE = re.compile("\u00c3[\u0080-\u00bf]"
                          "|\u00e2\u20ac[\u0098-\u009d\u2122\u00a6]"
                          "|\u00c2[\u00a0-\u00bf]")
_C0_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
# zero-width space/joiners, bidi marks and overrides, soft hyphen, BOM
_INVISIBLE_RE = re.compile("[\u00ad\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_URL_RE = re.compile(r"https?://|www\.")
_LETTER_RE = re.compile(r"[A-Za-z]")
# The publisher fingerprint that carries this dataset's label. Kept here rather
# than imported from eda.py so the two modules stay independent.
_REUTERS_TAG_RE = re.compile(r"\(\s*Reuters\s*\)")


# --- plumbing --------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Profile data/Fake.csv and data/True.csv and score their quality.")
    p.add_argument("--data-dir", default=None, help="Folder with Fake.csv / True.csv")
    p.add_argument("--sample", type=int, default=None,
                   help="Profile a random sample of N combined rows (faster while iterating)")
    p.add_argument("--near-dup-sample", type=int, default=6000,
                   help="How many unique articles to MinHash for near-duplicate detection "
                        "(0 = skip). Pairwise comparison of all 39k is not attempted.")
    p.add_argument("--near-dup-threshold", type=float, default=0.80,
                   help="Estimated Jaccard similarity above which two articles count as "
                        "near-duplicates")
    p.add_argument("--vocab-sample", type=int, default=8000,
                   help="Rows used for the vocabulary drift comparison (text cleaning is slow)")
    p.add_argument("--no-figures", action="store_true", help="Skip PNG generation")
    p.add_argument("--strict", action="store_true",
                   help="Exit 1 if any check FAILs, so this can gate a training pipeline")
    return p.parse_args()


@dataclass
class Check:
    """One named quality check and its verdict."""
    id: str
    category: str
    name: str
    status: str
    summary: str
    weight: float = 1.0
    metrics: dict = field(default_factory=dict)
    remedy: str = ""

    @property
    def score(self):
        return STATUS_SCORE.get(self.status)  # None for INFO


def verdict(value: float, warn_at: float, fail_at: float, higher_is_worse: bool = True) -> str:
    """Turn a measured rate into PASS/WARN/FAIL against two thresholds."""
    if higher_is_worse:
        if value >= fail_at:
            return FAIL
        return WARN if value > warn_at else PASS
    if value <= fail_at:
        return FAIL
    return WARN if value < warn_at else PASS


def worst(*statuses: str) -> str:
    """The most severe of several verdicts. Used where one metric can hide what
    another one exposes, so the check reports both and grades on the worse."""
    return min(statuses, key=lambda s: STATUS_ORDER[s])


def save_fig(fig, name: str, enabled: bool = True) -> str:
    """Save a figure into reports/figures and return its relative path."""
    if not enabled:
        plt.close(fig)
        return ""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}"


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
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


def jsonable(obj):
    """numpy/pandas scalars are not JSON-serialisable; unwrap them."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


# --- 1. integrity ----------------------------------------------------------

def file_fingerprint(path: Path) -> dict:
    """SHA-256 + size + row count, so a training run can be tied to exact bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return {
        "file": path.name,
        "sha256": h.hexdigest(),
        "bytes": path.stat().st_size,
        "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
    }


def integrity_checks(fingerprints: list[dict], previous: dict | None) -> list[Check]:
    checks = [Check(
        id="integrity.fingerprint",
        category="integrity",
        name="Source files fingerprinted",
        status=PASS,
        summary=f"{len(fingerprints)} source files hashed (SHA-256) and recorded.",
        metrics={"files": fingerprints},
        remedy="Quote these hashes next to any reported accuracy figure.",
    )]

    # Compare against the previous run's JSON, if there is one. This is what
    # turns the checksum from decoration into an actual control.
    prev_files = {}
    if previous:
        for entry in previous.get("dataset", {}).get("files", []):
            prev_files[entry.get("file")] = entry.get("sha256")

    if not prev_files:
        checks.append(Check(
            id="integrity.unchanged",
            category="integrity",
            name="Data unchanged since last run",
            status=INFO,
            summary="No previous reports/data_quality.json - this run is the baseline.",
            metrics={"previous_run": None},
        ))
        return checks

    changed = [f["file"] for f in fingerprints
               if prev_files.get(f["file"]) not in (None, f["sha256"])]
    new = [f["file"] for f in fingerprints if f["file"] not in prev_files]
    status = FAIL if changed else (WARN if new else PASS)
    checks.append(Check(
        id="integrity.unchanged",
        category="integrity",
        name="Data unchanged since last run",
        status=status,
        summary=("All source files byte-identical to the previous run."
                 if status == PASS else
                 f"Changed since last run: {', '.join(changed) or '-'}; "
                 f"new: {', '.join(new) or '-'}."),
        metrics={"changed": changed, "new": new},
        remedy="If the data changed, retrain and re-evaluate; cached models are stale.",
    ))
    return checks


# --- 2. schema -------------------------------------------------------------

def read_source(path: Path) -> pd.DataFrame:
    """
    Read one source CSV as strings, letting pandas mark missing cells NaN so
    nulls and empty strings can be counted separately.
    """
    return pd.read_csv(path, dtype=str)


def schema_checks(frames: dict[str, pd.DataFrame]) -> list[Check]:
    expected = list(EXPECTED_SCHEMA)
    rows, checks = [], []
    problems = []

    for fname, df in frames.items():
        cols = list(df.columns)
        missing = [c for c in expected if c not in cols]
        unexpected = [c for c in cols if c not in expected]
        wrong_dtype = [c for c in expected if c in cols and df[c].dtype != object]
        rows.append({
            "file": fname,
            "columns": len(cols),
            "rows": len(df),
            "missing": ", ".join(missing) or "-",
            "unexpected": ", ".join(unexpected) or "-",
            "wrong_dtype": ", ".join(wrong_dtype) or "-",
            "column_order_matches": cols[:len(expected)] == expected,
        })
        problems += [(fname, "missing", c) for c in missing]
        problems += [(fname, "unexpected", c) for c in unexpected]
        problems += [(fname, "dtype", c) for c in wrong_dtype]

    status = FAIL if any(p[1] in ("missing", "dtype") for p in problems) else (
        WARN if problems else PASS)
    checks.append(Check(
        id="schema.columns",
        category="schema",
        name="Column set and dtypes match EXPECTED_SCHEMA",
        status=status,
        weight=1.5,
        summary=("Both files carry exactly the four expected columns "
                 "(title, text, subject, date), all object/string dtype."
                 if status == PASS else
                 f"{len(problems)} schema deviations: "
                 + "; ".join(f"{f}:{k}:{c}" for f, k, c in problems[:6])),
        metrics={"per_file": rows, "expected_columns": expected},
        remedy="Update EXPECTED_SCHEMA deliberately, or fix the extract.",
    ))

    # The two files have no label column at all - the label is carried by the
    # filename. That is a real weakness worth stating rather than glossing over.
    checks.append(Check(
        id="schema.label_provenance",
        category="schema",
        name="Label is implicit in the filename",
        status=WARN,
        summary="Neither CSV contains a label column; the label comes from which file "
                "a row was read from. Any row that moves between files silently "
                "changes class, and there is no way to detect that from the data.",
        metrics={"mapping": {k: int(v) for k, v in SOURCE_FILES.items()}},
        remedy="Materialise an explicit `label` column (and a source_file column) when "
               "the two CSVs are merged, so provenance survives the join.",
    ))
    return checks


# --- 3. completeness -------------------------------------------------------

def completeness_checks(frames: dict[str, pd.DataFrame]) -> tuple[list[Check], pd.DataFrame]:
    rows = []
    for fname, df in frames.items():
        for col in EXPECTED_SCHEMA:
            if col not in df.columns:
                continue
            s = df[col]
            n_null = int(s.isna().sum())
            n_empty = int((s.fillna("").str.strip() == "").sum())
            rows.append({
                "file": fname,
                "column": col,
                "rows": len(df),
                "nulls": n_null,
                "empty_or_blank": n_empty,
                "empty_rate": n_empty / max(len(df), 1),
            })
    table = pd.DataFrame(rows)

    worst = table.loc[table["empty_rate"].idxmax()]
    total_rows = int(table.groupby("file")["rows"].first().sum())
    total_empty = int(table["empty_or_blank"].sum())
    status = verdict(float(worst["empty_rate"]),
                     THRESHOLDS["empty_rate_warn"], THRESHOLDS["empty_rate_fail"])

    checks = [Check(
        id="completeness.nulls",
        category="completeness",
        name="No missing or blank required fields",
        status=status,
        weight=1.5,
        summary=(f"{total_empty:,} blank cells across {total_rows:,} rows. Worst column: "
                 f"`{worst['column']}` in {worst['file']} at {worst['empty_rate']:.2%} "
                 f"({int(worst['empty_or_blank']):,} rows)."),
        metrics={"per_column": rows, "total_blank_cells": total_empty},
        remedy="Rows with an empty `text` still carry a title, so `content` is non-empty "
               "and they survive load_data(). Decide explicitly whether a headline-only "
               "row is a valid training example - currently it is treated as one.",
    )]
    return checks, table


# --- 4. uniqueness ---------------------------------------------------------

_NORMALISE_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalise(series: pd.Series) -> pd.Series:
    """Lowercase, drop punctuation, collapse whitespace. Used for fuzzy matching."""
    return (series.str.lower()
            .str.replace(_NORMALISE_PUNCT_RE, " ", regex=True)
            .str.replace(_MULTI_SPACE_RE, " ", regex=True)
            .str.strip())


def minhash_signatures(docs, n_hashes: int = 64, shingle: int = 5,
                       max_words: int = 600, seed: int = 42) -> np.ndarray:
    """
    MinHash signature per document, computed in one linear pass.

    Pairwise Jaccard over 39k documents is ~760M comparisons and is not
    attempted anywhere in this file. MinHash reduces each document to a fixed
    64-int sketch whose agreement rate estimates Jaccard similarity, and the
    banding step below turns lookup into a dictionary join.
    """
    prime = (1 << 31) - 1
    rng = np.random.default_rng(seed)
    a = rng.integers(1, prime, n_hashes, dtype=np.int64)[:, None]
    b = rng.integers(0, prime, n_hashes, dtype=np.int64)[:, None]

    sigs = np.full((len(docs), n_hashes), prime, dtype=np.int64)
    for i, doc in enumerate(docs):
        words = doc.split()[:max_words]
        if not words:
            continue
        if len(words) <= shingle:
            grams = {" ".join(words)}
        else:
            grams = {" ".join(words[j:j + shingle])
                     for j in range(len(words) - shingle + 1)}
        # crc32 rather than hash(): Python's str hash is salted per process,
        # so results would not be reproducible between runs.
        h = np.fromiter((zlib.crc32(g.encode("utf-8")) & 0x7FFFFFFF for g in grams),
                        dtype=np.int64, count=len(grams))[None, :]
        sigs[i] = ((a * h + b) % prime).min(axis=1)
    return sigs


def near_duplicate_pairs(sigs: np.ndarray, threshold: float,
                         bands: int = 16, bucket_cap: int = 40) -> list[tuple[int, int, float]]:
    """LSH banding over MinHash signatures -> candidate pairs -> estimated Jaccard."""
    n, n_hashes = sigs.shape
    rows_per_band = n_hashes // bands
    candidates = set()
    truncated = 0

    for band in range(bands):
        block = sigs[:, band * rows_per_band:(band + 1) * rows_per_band]
        buckets: dict[bytes, list[int]] = {}
        for i in range(n):
            buckets.setdefault(block[i].tobytes(), []).append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            if len(members) > bucket_cap:
                # A huge bucket means a boilerplate-identical family; taking
                # every pair would blow up. Keep a slice and record that.
                truncated += 1
                members = members[:bucket_cap]
            for x in range(len(members)):
                for y in range(x + 1, len(members)):
                    candidates.add((members[x], members[y]))

    pairs = []
    for i, j in candidates:
        sim = float((sigs[i] == sigs[j]).mean())
        if sim >= threshold:
            pairs.append((i, j, sim))
    pairs.sort(key=lambda t: -t[2])
    return pairs


def uniqueness_checks(df: pd.DataFrame, near_dup_sample: int,
                      threshold: float) -> tuple[list[Check], pd.DataFrame]:
    n = len(df)
    checks = []

    dup_mask = df.duplicated(subset="content", keep="first")
    n_dupes = int(dup_mask.sum())
    dup_rate = n_dupes / max(n, 1)

    # Duplicates that appear under both labels are the dangerous kind: the same
    # text is in the corpus as fake *and* as real.
    groups = df.groupby("content")["label"].nunique()
    cross_label = int((groups > 1).sum())

    status = verdict(dup_rate, THRESHOLDS["dup_rate_warn"], THRESHOLDS["dup_rate_fail"])
    if cross_label:
        status = FAIL
    checks.append(Check(
        id="uniqueness.exact_duplicates",
        category="uniqueness",
        name="No exact duplicate articles",
        status=status,
        weight=2.0,
        summary=(f"{n_dupes:,} of {n:,} rows ({dup_rate:.1%}) are exact duplicates of an "
                 f"earlier row. {cross_label} article texts appear under both labels."),
        metrics={"rows": n, "duplicate_rows": n_dupes, "duplicate_rate": dup_rate,
                 "texts_with_both_labels": cross_label},
        remedy="load_data(drop_duplicates=True) removes these, and it must run BEFORE "
               "train_test_split - otherwise the same article lands on both sides of "
               "the split and test accuracy is inflated.",
    ))

    title_dupes = int(df.duplicated(subset="title", keep="first").sum())
    title_rate = title_dupes / max(n, 1)
    unique_after_content_dedupe = df.drop_duplicates(subset="content")
    title_dupes_after = int(unique_after_content_dedupe.duplicated(subset="title").sum())
    checks.append(Check(
        id="uniqueness.duplicate_titles",
        category="uniqueness",
        name="Headlines are distinct",
        status=verdict(title_rate, THRESHOLDS["dup_title_rate_warn"],
                       THRESHOLDS["dup_title_rate_fail"]),
        summary=(f"{title_dupes:,} repeated headlines ({title_rate:.1%}). After exact-body "
                 f"deduplication {title_dupes_after:,} repeated headlines remain, i.e. "
                 f"different bodies published under the same headline."),
        metrics={"duplicate_titles": title_dupes, "duplicate_title_rate": title_rate,
                 "duplicate_titles_after_content_dedupe": title_dupes_after},
        remedy="Same-headline/different-body pairs are usually a story and its update; "
               "they are near-duplicates for training purposes.",
    ))

    # Normalised-text duplicates, over the whole corpus and in linear time.
    # Two articles differing only in curly quotes, casing or spacing survive
    # drop_duplicates(subset="content") untouched; this counts how many.
    norm_all = normalise(df["content"])
    norm_dupes = int(norm_all.duplicated(keep="first").sum())
    extra_from_normalising = max(norm_dupes - n_dupes, 0)
    norm_rate = extra_from_normalising / max(n, 1)

    near_table = pd.DataFrame()
    if near_dup_sample and near_dup_sample > 1:
        unique_df = unique_after_content_dedupe.reset_index(drop=True)
        take = min(near_dup_sample, len(unique_df))
        sample = unique_df.sample(n=take, random_state=42).reset_index(drop=True)
        norm = normalise(sample["content"])
        sigs = minhash_signatures(norm.tolist())
        pairs = near_duplicate_pairs(sigs, threshold)

        involved = {i for p in pairs for i in p[:2]}
        cross = sum(1 for i, j, _ in pairs
                    if sample.at[i, "label"] != sample.at[j, "label"])
        identical_normalised = sum(1 for i, j, _ in pairs if norm.iat[i] == norm.iat[j])
        rate = len(involved) / max(take, 1)
        # A sample of s from N sees only (s/N)^2 of the pairs, so the sampled
        # rate is a floor. The normalised-duplicate count above is exhaustive,
        # so the check is graded on the worse of the two rather than on the
        # sampled number alone.
        pair_coverage = (take / max(len(unique_df), 1)) ** 2

        near_status = worst(
            verdict(rate, THRESHOLDS["near_dup_rate_warn"],
                    THRESHOLDS["near_dup_rate_fail"]),
            verdict(norm_rate, THRESHOLDS["near_dup_rate_warn"],
                    THRESHOLDS["near_dup_rate_fail"]),
        )
        if cross:
            near_status = FAIL
        checks.append(Check(
            id="uniqueness.near_duplicates",
            category="uniqueness",
            name="No near-duplicate articles",
            status=near_status,
            weight=1.5,
            summary=(f"Exhaustive, whole corpus: normalising case, punctuation and "
                     f"whitespace turns {extra_from_normalising:,} further rows "
                     f"({norm_rate:.1%}) into duplicates that `drop_duplicates(\"content\")` "
                     f"leaves in place. Sampled, MinHash+LSH over {take:,} of the "
                     f"{len(unique_df):,} unique articles: {len(pairs):,} pairs at "
                     f"estimated Jaccard >= {threshold:.2f}, touching {len(involved):,} "
                     f"articles ({rate:.1%} of the sample), of which "
                     f"{identical_normalised} are the punctuation-only kind and {cross} "
                     f"straddle the fake/real boundary. That sample covers only "
                     f"{pair_coverage:.1%} of all possible pairs, so the sampled rate is "
                     f"a lower bound, not an estimate - the exhaustive number above is "
                     f"the one to act on."),
            metrics={"normalised_extra_duplicates": extra_from_normalising,
                     "normalised_extra_rate": norm_rate,
                     "sample_size": take, "unique_articles": int(len(unique_df)),
                     "threshold": threshold, "pairs": len(pairs),
                     "articles_involved": len(involved), "sampled_rate": rate,
                     "pair_coverage_of_corpus": pair_coverage,
                     "identical_after_normalisation": identical_normalised,
                     "cross_label_pairs": cross, "n_hashes": 64, "shingle_words": 5},
            remedy="Exact deduplication is not enough: near-duplicates split across the "
                   "train/test boundary leak just as effectively. Deduplicate on a hash "
                   "of the normalised text - that is O(n), needs no threshold, and "
                   "catches the punctuation-only pairs for free. Raise --near-dup-sample "
                   "to search harder for the genuinely fuzzy ones.",
        ))

        if pairs:
            near_table = pd.DataFrame([{
                "rank": r + 1,
                "similarity": f"{sim:.3f}",
                "same_after_normalising": "yes" if norm.iat[i] == norm.iat[j] else "no",
                "labels": ("fake" if sample.at[i, "label"] == FAKE else "real") + "/"
                          + ("fake" if sample.at[j, "label"] == FAKE else "real"),
                "title_a": sample.at[i, "title"][:55],
                "title_b": sample.at[j, "title"][:55],
            } for r, (i, j, sim) in enumerate(pairs[:15])]).set_index("rank")

    # Cheap O(n) complement to MinHash: two articles that share the first 120
    # normalised characters are almost always the same story.
    prefix = norm_all.str.slice(0, 120)
    prefix_hash = prefix.map(
        lambda s: hashlib.blake2b(s.encode("utf-8"), digest_size=8).hexdigest())
    # Count the same way as duplicated(): rows that repeat an *earlier* row, so
    # this is directly comparable with n_dupes rather than off by one per group.
    repeat_prefix = int(prefix_hash.duplicated(keep="first").sum())
    beyond_exact = max(repeat_prefix - n_dupes, 0)
    checks.append(Check(
        id="uniqueness.shared_openings",
        category="uniqueness",
        name="Articles do not share identical openings",
        status=verdict(beyond_exact / max(n, 1), 0.01, 0.05),
        summary=(f"{repeat_prefix:,} rows repeat an earlier row's first 120 normalised "
                 f"characters. {n_dupes:,} of those are exact duplicates already counted "
                 f"above, leaving {beyond_exact:,} rows ({beyond_exact / max(n, 1):.1%}) "
                 f"that open identically but are not identical texts."),
        metrics={"rows_repeating_a_prefix": repeat_prefix,
                 "exact_duplicates_among_them": n_dupes,
                 "beyond_exact_duplicates": beyond_exact,
                 "prefix_chars": 120},
        remedy="Cheap linear screen: one hash per row, no pairwise comparison. Run it "
               "before the MinHash join to see whether the expensive step is even needed.",
    ))
    return checks, near_table


# --- 5. validity -----------------------------------------------------------

def validity_checks(df: pd.DataFrame, frames: dict[str, pd.DataFrame]
                    ) -> tuple[list[Check], pd.Series, pd.DataFrame]:
    n = len(df)
    checks = []

    # -- dates
    parsed = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    n_bad = int(parsed.isna().sum())
    bad_rate = n_bad / max(n, 1)
    lo = pd.Timestamp(EXPECTED_SCHEMA["date"]["constraints"]["min"])
    hi = pd.Timestamp(EXPECTED_SCHEMA["date"]["constraints"]["max"])
    out_of_range = int(((parsed < lo) | (parsed > hi)).sum())

    bad_examples = df.loc[parsed.isna(), "date"].head(5).tolist()
    looks_like_url = int(df.loc[parsed.isna(), "date"].str.contains(_URL_RE, na=False).sum())
    checks.append(Check(
        id="validity.dates",
        category="validity",
        name="Dates parse and fall in the declared range",
        status=verdict(bad_rate, THRESHOLDS["unparseable_date_rate_warn"],
                       THRESHOLDS["unparseable_date_rate_fail"]),
        summary=(f"{n_bad} of {n:,} date values ({bad_rate:.3%}) do not parse; "
                 f"{looks_like_url} of those contain a URL instead of a date. "
                 f"{out_of_range} parsed dates fall outside "
                 f"{lo.date()}..{hi.date()}."),
        metrics={"unparseable": n_bad, "unparseable_rate": bad_rate,
                 "url_in_date_field": looks_like_url, "out_of_range": out_of_range,
                 "examples": [e[:80] for e in bad_examples],
                 "min": str(parsed.min()), "max": str(parsed.max())},
        remedy="These rows are scraper failures - the URL landed in both `title` and "
               "`date`. Drop them; they carry no article text worth training on.",
    ))

    # -- rows where the title is a URL: the same failure seen from the other side
    url_titles = int(df["title"].str.contains(_URL_RE, na=False).sum())
    checks.append(Check(
        id="validity.title_is_url",
        category="validity",
        name="Headlines are headlines, not URLs",
        status=verdict(url_titles / max(n, 1), 0.0, 0.005),
        summary=f"{url_titles} rows have a URL in the `title` field.",
        metrics={"url_titles": url_titles, "rate": url_titles / max(n, 1)},
        remedy="Same scraper failure as the unparseable dates - the two overlap.",
    ))

    # -- encoding
    enc_rows = []
    for fname, frame in frames.items():
        for col in ("title", "text"):
            if col not in frame.columns:
                continue
            s = frame[col].fillna("")
            enc_rows.append({
                "file": fname, "column": col,
                "replacement_char": int(s.str.contains(_REPLACEMENT_RE, na=False).sum()),
                "mojibake": int(s.str.contains(_MOJIBAKE_RE, na=False).sum()),
                "control_char": int(s.str.contains(_C0_CONTROL_RE, na=False).sum()),
                "invisible_char": int(s.str.contains(_INVISIBLE_RE, na=False).sum()),
                "non_ascii": int(s.str.contains(r"[^\x00-\x7f]", na=False, regex=True).sum()),
            })
    enc = pd.DataFrame(enc_rows)
    hard_bad = int(enc[["replacement_char", "mojibake", "control_char"]].to_numpy().sum())
    soft_bad = int(enc["invisible_char"].sum())
    total_cells = sum(len(f) for f in frames.values()) * 2
    enc_status = verdict((hard_bad + soft_bad) / max(total_cells, 1),
                         THRESHOLDS["encoding_bad_rate_warn"],
                         THRESHOLDS["encoding_bad_rate_fail"])
    checks.append(Check(
        id="validity.encoding",
        category="validity",
        name="Text decodes cleanly as UTF-8",
        status=enc_status,
        summary=(f"No U+FFFD replacement characters, no mojibake sequences and no C0 "
                 f"control characters ({hard_bad} total). {soft_bad} cells contain "
                 f"zero-width or bidi format characters, and "
                 f"{int(enc['non_ascii'].sum()):,} contain non-ASCII characters - almost "
                 f"all of them curly quotes, which clean_text() strips anyway."
                 if hard_bad == 0 else
                 f"{hard_bad} cells contain replacement characters, mojibake or control "
                 f"characters - the file was decoded with the wrong codec somewhere."),
        metrics={"per_column": enc_rows, "hard_failures": hard_bad,
                 "invisible_chars": soft_bad},
        remedy="Normalise Unicode (NFKC) and strip format characters at load time if the "
               "vectorizer is ever changed to keep punctuation.",
    ))

    # -- length
    words = df["n_words"]
    min_w = EXPECTED_SCHEMA["text"]["constraints"]["min_words"]
    max_w = EXPECTED_SCHEMA["text"]["constraints"]["max_words"]
    n_short = int((words < min_w).sum())
    n_long = int((words > max_w).sum())
    n_empty_body = int((df["body"].str.strip() == "").sum())
    short_rate = n_short / max(n, 1)
    checks.append(Check(
        id="validity.length",
        category="validity",
        name="Article length is plausible",
        status=verdict(short_rate, THRESHOLDS["short_article_rate_warn"],
                       THRESHOLDS["short_article_rate_fail"]),
        weight=1.5,
        summary=(f"{n_short:,} articles ({short_rate:.1%}) are under {min_w} words, "
                 f"{n_long:,} are over {max_w:,}. {n_empty_body:,} rows have an empty "
                 f"body and consist of the headline alone. Median {words.median():.0f} "
                 f"words, max {words.max():,}."),
        metrics={"short": n_short, "short_rate": short_rate, "long": n_long,
                 "empty_body": n_empty_body, "median_words": float(words.median()),
                 "max_words": int(words.max()), "min_words": int(words.min())},
        remedy="A headline-only row teaches the model headline style, not article "
               "content. Either drop them or record that the effective corpus is mixed.",
    ))

    # -- shouting / punctuation soup
    n_chars = df["content"].str.len().replace(0, np.nan)
    letters = df["content"].str.count(_LETTER_RE)
    alpha_ratio = (letters / n_chars).fillna(0.0)
    shouting = df["upper_char_ratio"] > 0.60
    soup = alpha_ratio < 0.50
    n_shout, n_soup = int(shouting.sum()), int(soup.sum())
    checks.append(Check(
        id="validity.formatting",
        category="validity",
        name="Articles are not all-caps or mostly punctuation",
        status=verdict((n_shout + n_soup) / max(n, 1), 0.002, 0.02),
        summary=(f"{n_shout:,} articles are more than 60% uppercase characters and "
                 f"{n_soup:,} are less than 50% letters. Both skew heavily fake "
                 f"({int((shouting & (df.label == FAKE)).sum())} and "
                 f"{int((soup & (df.label == FAKE)).sum())} respectively), so they are a "
                 f"style signal as much as a defect."),
        metrics={"all_caps": n_shout, "mostly_punctuation": n_soup,
                 "all_caps_fake_share": float((shouting & (df.label == FAKE)).sum()
                                              / max(n_shout, 1))},
        remedy="Do not silently delete these - they are real examples of the fake style. "
               "Flag them so the effect on the model can be measured.",
    ))
    return checks, parsed, enc


# --- 6. consistency --------------------------------------------------------

def consistency_checks(df: pd.DataFrame, parsed_dates: pd.Series
                       ) -> tuple[list[Check], pd.DataFrame]:
    n = len(df)
    checks = []

    ct = pd.crosstab(df["subject"], df["label"])
    for col in (FAKE, REAL):
        if col not in ct:
            ct[col] = 0
    ct = ct[[FAKE, REAL]]
    ct.columns = ["fake", "real"]
    ct["exclusive_to"] = np.where(ct["real"] == 0, "fake only",
                                  np.where(ct["fake"] == 0, "real only", "mixed"))
    exclusive = int((ct["exclusive_to"] != "mixed").sum())
    purity = exclusive / max(len(ct), 1)
    # A one-rule classifier on `subject` alone: predict each subject's majority class.
    subject_rule_acc = float(ct[["fake", "real"]].max(axis=1).sum() / max(n, 1))

    checks.append(Check(
        id="consistency.subject_label",
        category="consistency",
        name="Subject categories are shared across classes",
        status=verdict(purity, THRESHOLDS["subject_purity_warn"],
                       THRESHOLDS["subject_purity_fail"]),
        weight=2.0,
        summary=(f"{exclusive} of {len(ct)} subject values appear in exactly one class. "
                 f"Predicting the majority class of an article's `subject` scores "
                 f"{subject_rule_acc:.2%} accuracy on the whole corpus, with no text and "
                 f"no model involved."),
        metrics={"subjects": int(len(ct)), "exclusive_subjects": exclusive,
                 "purity": purity, "subject_only_rule_accuracy": subject_rule_acc,
                 "unknown_subjects": sorted(set(ct.index) - KNOWN_SUBJECTS)},
        remedy="`subject` must never be used as a feature, and its presence means the two "
               "files came from different scrapes rather than one sampling frame. Treat "
               "it as evidence about provenance, not as metadata.",
    ))

    # Date range disjointness: how much of the corpus is separable on date alone?
    ok = parsed_dates.notna()
    fake_dates = parsed_dates[ok & (df["label"] == FAKE)]
    real_dates = parsed_dates[ok & (df["label"] == REAL)]
    overlap_lo = max(fake_dates.min(), real_dates.min())
    overlap_hi = min(fake_dates.max(), real_dates.max())
    outside = (parsed_dates < overlap_lo) | (parsed_dates > overlap_hi)
    outside = outside & ok
    n_outside = int(outside.sum())
    outside_fake_share = float((df.loc[outside, "label"] == FAKE).mean()) if n_outside else np.nan

    checks.append(Check(
        id="consistency.date_label",
        category="consistency",
        name="Publication dates overlap across classes",
        status=FAIL if n_outside and outside_fake_share > 0.99 else (
            WARN if n_outside else PASS),
        weight=1.5,
        summary=(f"Fake articles span {fake_dates.min().date()}..{fake_dates.max().date()}, "
                 f"real articles {real_dates.min().date()}..{real_dates.max().date()}. "
                 f"{n_outside:,} articles ({n_outside / max(n, 1):.1%}) fall outside the "
                 f"shared window, and {outside_fake_share:.1%} of them are fake - the date "
                 f"alone classifies them."),
        metrics={"fake_range": [str(fake_dates.min().date()), str(fake_dates.max().date())],
                 "real_range": [str(real_dates.min().date()), str(real_dates.max().date())],
                 "shared_window": [str(overlap_lo.date()), str(overlap_hi.date())],
                 "articles_outside_shared_window": n_outside,
                 "fake_share_outside": outside_fake_share},
        remedy="Restrict both classes to the shared window before drawing any conclusion "
               "about model skill, or accept that part of the accuracy is calendar "
               "arithmetic.",
    ))

    # The single most important check in this file. `src/eda.py` runs a full
    # marker sweep; this is the one-line version, here because a quality gate
    # that cannot see the dataset's central defect is not a gate.
    hits = df["content"].str.contains(_REUTERS_TAG_RE, na=False)
    in_fake = float(hits[df.label == FAKE].mean())
    in_real = float(hits[df.label == REAL].mean())
    gap = abs(in_real - in_fake)
    rule_acc = float(((hits if in_real >= in_fake else ~hits).astype(int)
                      == df["label"]).mean())
    checks.append(Check(
        id="consistency.publisher_fingerprint",
        category="consistency",
        name="No single token identifies a class",
        status=verdict(gap, THRESHOLDS["marker_class_gap_warn"],
                       THRESHOLDS["marker_class_gap_fail"]),
        weight=2.0,
        summary=(f"The literal string `(Reuters)` appears in {in_real:.1%} of real "
                 f"articles and {in_fake:.1%} of fake ones. \"Contains it, therefore "
                 f"real\" scores {rule_acc:.2%} on the whole corpus with no model. "
                 f"The two classes are not fake vs real news; they are one wire service "
                 f"vs a set of blogs."),
        metrics={"marker": "(Reuters)", "in_fake": in_fake, "in_real": in_real,
                 "gap": gap, "one_rule_accuracy": rule_acc},
        remedy="Every accuracy figure from this dataset must be quoted next to this "
               "number. `python src/train.py --strip-boilerplate` retrains without the "
               "marker; the drop is the leakage share. See `reports/eda_report.md` for "
               "the full marker sweep.",
    ))

    # Label balance - genuinely fine here, and worth saying so.
    counts = df["label"].value_counts()
    minority = float(counts.min() / counts.sum())
    checks.append(Check(
        id="consistency.class_balance",
        category="consistency",
        name="Classes are reasonably balanced",
        status=PASS if minority > 0.35 else (WARN if minority > 0.2 else FAIL),
        summary=(f"fake {int(counts.get(FAKE, 0)):,} / real {int(counts.get(REAL, 0)):,}; "
                 f"minority class is {minority:.1%} of the corpus. No resampling needed."),
        metrics={"fake": int(counts.get(FAKE, 0)), "real": int(counts.get(REAL, 0)),
                 "minority_share": minority},
    ))
    return checks, ct


# --- 7. drift --------------------------------------------------------------

def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon *distance* (base 2, so it lands in [0, 1]); 0 = identical."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    return float(jensenshannon(p / p.sum(), q / q.sum(), base=2))


def drift_checks(df: pd.DataFrame, parsed_dates: pd.Series,
                 vocab_sample: int) -> tuple[list[Check], dict]:
    checks = []
    fake_len = df.loc[df.label == FAKE, "n_words"].to_numpy()
    real_len = df.loc[df.label == REAL, "n_words"].to_numpy()

    # -- length
    ks = ks_2samp(fake_len, real_len)
    bins = np.histogram_bin_edges(np.clip(np.concatenate([fake_len, real_len]), 0, 2000), bins=60)
    hist_f = np.histogram(np.clip(fake_len, 0, 2000), bins=bins)[0]
    hist_r = np.histogram(np.clip(real_len, 0, 2000), bins=bins)[0]
    len_js = js_distance(hist_f, hist_r)

    checks.append(Check(
        id="drift.length",
        category="drift",
        name="Article length distributions agree",
        status=verdict(float(ks.statistic), THRESHOLDS["ks_stat_warn"],
                       THRESHOLDS["ks_stat_fail"]),
        summary=(f"KS statistic {ks.statistic:.3f} (p={ks.pvalue:.2e}), "
                 f"Jensen-Shannon distance {len_js:.3f} on the length histogram. "
                 f"Median {np.median(fake_len):.0f} words fake vs "
                 f"{np.median(real_len):.0f} real - a {abs(np.median(fake_len) - np.median(real_len)):.0f}-word "
                 f"difference. A KS statistic of {ks.statistic:.2f} is a small effect: "
                 f"length is the dimension on which these two files come closest to "
                 f"agreeing, and it is also the only one a scraper does not control."),
        metrics={"ks_statistic": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
                 "js_distance": len_js,
                 "median_fake": float(np.median(fake_len)),
                 "median_real": float(np.median(real_len))},
        remedy="With n>17,000 per class the KS p-value is effectively zero for any "
               "difference at all, so read the statistic (effect size), not the p-value.",
    ))

    # -- vocabulary
    take = min(vocab_sample, len(df))
    sample = df.sample(n=take, random_state=42)
    cleaned = sample["content"].map(clean_text)
    vec = CountVectorizer(min_df=3, max_features=20000)
    X = vec.fit_transform(cleaned)
    is_fake = (sample["label"] == FAKE).to_numpy()
    p = np.asarray(X[is_fake].sum(axis=0)).ravel()
    q = np.asarray(X[~is_fake].sum(axis=0)).ravel()
    vocab_js = js_distance(p, q)
    fake_terms = set(np.asarray(vec.get_feature_names_out())[p > 0])
    real_terms = set(np.asarray(vec.get_feature_names_out())[q > 0])
    jaccard = len(fake_terms & real_terms) / max(len(fake_terms | real_terms), 1)

    checks.append(Check(
        id="drift.vocabulary",
        category="drift",
        name="Vocabulary distributions agree",
        status=verdict(vocab_js, THRESHOLDS["js_distance_warn"],
                       THRESHOLDS["js_distance_fail"]),
        summary=(f"Jensen-Shannon distance {vocab_js:.3f} between the two unigram "
                 f"distributions on a {take:,}-article sample; term-set Jaccard overlap "
                 f"{jaccard:.3f} over the {len(vec.vocabulary_):,} terms that survive "
                 f"min_df=3. (`src/eda.py` reports Jaccard 0.34 for the same corpus - "
                 f"that figure includes the long tail of once-seen tokens, which is where "
                 f"the two files diverge most. Neither number is wrong; they measure "
                 f"different vocabularies.) Some vocabulary difference is expected "
                 f"between real and fabricated news; the question is whether it is "
                 f"topical or publisher-specific, and the leakage audit in `src/eda.py` "
                 f"says largely the latter."),
        metrics={"js_distance": vocab_js, "jaccard": jaccard, "sample": take,
                 "vocab_size": int(len(vec.vocabulary_))},
        remedy="Re-run this after `strip_source_boilerplate()` to see how much of the "
               "gap is Reuters datelines rather than content.",
    ))

    # -- dates
    ok = parsed_dates.notna()
    months = parsed_dates[ok].dt.to_period("M")
    monthly = (pd.DataFrame({"month": months, "label": df.loc[ok, "label"]})
               .groupby(["month", "label"]).size().unstack(fill_value=0))
    for col in (FAKE, REAL):
        if col not in monthly:
            monthly[col] = 0
    monthly = monthly[[FAKE, REAL]]
    monthly.columns = ["fake", "real"]
    date_js = js_distance(monthly["fake"].to_numpy(), monthly["real"].to_numpy())
    months_one_class = int(((monthly["fake"] == 0) | (monthly["real"] == 0)).sum())
    single_class_share = months_one_class / max(len(monthly), 1)

    # Two metrics, and they disagree - so take the worse. JS distance is
    # dominated by the months where both classes are dense, which is most of
    # them, so on its own it understates a hard boundary at either end of the
    # calendar. The share of single-class months catches exactly that.
    date_status = worst(
        verdict(date_js, THRESHOLDS["js_distance_warn"], THRESHOLDS["js_distance_fail"]),
        verdict(single_class_share, THRESHOLDS["single_class_month_warn"],
                THRESHOLDS["single_class_month_fail"]),
    )
    checks.append(Check(
        id="drift.dates",
        category="drift",
        name="Publication date distributions agree",
        status=date_status,
        weight=1.5,
        summary=(f"Jensen-Shannon distance {date_js:.3f} on the monthly publication "
                 f"histogram - which on its own looks unremarkable, because both classes "
                 f"are dense through the middle of the period. The decisive number is "
                 f"that {months_one_class} of {len(monthly)} months "
                 f"({single_class_share:.0%}) contain articles from one class only. "
                 f"A single divergence statistic hides a hard calendar boundary; this "
                 f"check reports both and takes the worse verdict."),
        metrics={"js_distance": date_js, "months": int(len(monthly)),
                 "single_class_months": months_one_class,
                 "single_class_month_share": single_class_share},
        remedy="Sample both classes from the same calendar window, or state plainly that "
               "the model has a date confound it cannot separate.",
    ))

    payload = {"monthly": monthly, "bins": bins, "hist_fake": hist_f, "hist_real": hist_r,
               "ks": ks}
    return checks, payload


# --- scoring, figure, reporting -------------------------------------------

def score_checks(checks: list[Check]) -> dict:
    scored = [c for c in checks if c.score is not None]
    total_weight = sum(c.weight for c in scored)
    earned = sum(c.weight * c.score for c in scored)
    counts = {s: sum(1 for c in checks if c.status == s) for s in (PASS, WARN, FAIL, INFO)}
    return {
        "score": 100.0 * earned / total_weight if total_weight else float("nan"),
        "weighted_points": earned,
        "weight_total": total_weight,
        "counts": counts,
        "n_checks": len(checks),
    }


def checks_frame(checks: list[Check]) -> pd.DataFrame:
    table = pd.DataFrame([{
        "check": c.id,
        "category": c.category,
        "status": c.status,
        "weight": c.weight,
        "what it means": c.summary,
    } for c in checks]).set_index("check")
    table.index.name = "check"
    return table


def plot_dashboard(checks: list[Check], summary: dict, drift_payload: dict,
                   defect_counts: dict, enabled: bool = True) -> str:
    ordered = sorted(checks, key=lambda c: (STATUS_ORDER[c.status], c.id))
    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.25, 1], hspace=0.6, wspace=0.3)

    # (1) per-check verdicts. Bar length is the check's weight, so a reader can
    # see at a glance which failures actually move the score.
    ax = fig.add_subplot(gs[:, 0])
    y = np.arange(len(ordered))
    weights = [c.weight for c in ordered]
    ax.barh(y, weights, color=[STATUS_COLOR[c.status] for c in ordered])
    ax.set_yticks(y, [c.id for c in ordered], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(weights) * 1.55)
    ax.set_xlabel("check weight in the score", fontsize=8)
    for i, c in enumerate(ordered):
        ax.text(weights[i] + 0.06, i, c.status, va="center", fontsize=7.5,
                color=STATUS_COLOR[c.status], fontweight="bold")
    ax.set_title(f"Quality checks - score {summary['score']:.0f}/100  "
                 f"({summary['counts'][FAIL]} fail, {summary['counts'][WARN]} warn, "
                 f"{summary['counts'][PASS]} pass)", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    # (2) how many rows each defect touches. Log scale, because the counts span
    # single rows (scraper failures) to five figures (duplicates).
    ax = fig.add_subplot(gs[0, 1])
    labels = list(defect_counts)
    vals = [max(defect_counts[k], 0.7) for k in labels]  # 0.7 keeps zeros visible on a log axis
    ax.barh(np.arange(len(labels)), vals, color="#6c757d")
    ax.set_yticks(np.arange(len(labels)), labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("rows affected (log scale)", fontsize=8)
    for i, k in enumerate(labels):
        ax.text(vals[i] * 1.15, i, f"{defect_counts[k]:,}", va="center", fontsize=7)
    ax.set_xlim(0.7, max(vals) * 6)
    ax.set_title("Defects by rows affected", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    # (3) length distributions + KS
    ax = fig.add_subplot(gs[1, 1])
    bins, hf, hr = drift_payload["bins"], drift_payload["hist_fake"], drift_payload["hist_real"]
    centres = (bins[:-1] + bins[1:]) / 2
    ax.plot(centres, hf / max(hf.sum(), 1), color=FAKE_COLOR, label="fake", linewidth=1.6)
    ax.plot(centres, hr / max(hr.sum(), 1), color=REAL_COLOR, label="real", linewidth=1.6)
    ax.set_xlabel("words per article (clipped at 2000)", fontsize=8)
    ax.set_ylabel("density")
    ax.set_title(f"Length drift - KS={drift_payload['ks'].statistic:.3f}", fontsize=10)
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    # (4) monthly coverage - the disjoint windows
    ax = fig.add_subplot(gs[2, 1])
    monthly = drift_payload["monthly"]
    idx = monthly.index.to_timestamp()
    ax.fill_between(idx, monthly["fake"], color=FAKE_COLOR, alpha=0.65, label="fake")
    ax.fill_between(idx, monthly["real"], color=REAL_COLOR, alpha=0.55, label="real")
    ax.set_ylabel("articles/month")
    ax.set_title("Date drift - the classes do not cover the same period", fontsize=10)
    ax.legend(frameon=False, fontsize=7)
    ax.tick_params(axis="x", labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    return save_fig(fig, "data_quality.png", enabled)


def build_report(checks: list[Check], summary: dict, fingerprints: list[dict],
                 completeness: pd.DataFrame, subject_ct: pd.DataFrame,
                 near_table: pd.DataFrame, encoding: pd.DataFrame,
                 n_rows: int, fig_path: str, args) -> str:
    by_id = {c.id: c for c in checks}
    fails = [c for c in checks if c.status == FAIL]

    def embed(path, caption):
        return f"\n![{caption}]({path})\n" if path else ""

    def section(ids):
        return "\n".join(
            f"**{by_id[i].name}** - `{by_id[i].status}`  \n{by_id[i].summary}"
            + (f"  \n*What to do:* {by_id[i].remedy}\n" if by_id[i].remedy else "\n")
            for i in ids if i in by_id)

    dict_rows = pd.DataFrame([{
        "dtype": spec["dtype"],
        "nullable": spec["nullable"],
        "constraints": ", ".join(f"{k}={v}" for k, v in spec["constraints"].items()),
        "description": spec["description"],
    } for spec in EXPECTED_SCHEMA.values()], index=list(EXPECTED_SCHEMA))
    dict_rows.index.name = "column"

    fp = pd.DataFrame(fingerprints).set_index("file")
    fp["sha256"] = fp["sha256"].str.slice(0, 16) + "..."
    fp["bytes"] = fp["bytes"].map(lambda v: f"{v:,}")

    comp = completeness.copy()
    comp["empty_rate"] = comp["empty_rate"].map("{:.3%}".format)
    comp = comp.set_index("file")

    near_section = "No near-duplicate pairs above the threshold in the sample.\n"
    if not near_table.empty:
        near_section = md_table(near_table.set_index("similarity"), floatfmt="{:.2f}")

    fail_list = "\n".join(f"- `{c.id}` - {c.name}" for c in fails) or "- (none)"

    # Sampling is fine for distributions and useless for duplicate rates: a
    # duplicate pair only survives if both of its rows are drawn. Say so rather
    # than letting a reader compare a sampled run against a full one.
    sample_warning = (
        f"\n> **Run on a {n_rows:,}-row random sample (`--sample`).** Distribution and "
        f"validity figures below are reliable; the uniqueness figures are not. A "
        f"duplicate pair only appears in a sample if both of its rows are drawn, so "
        f"every duplicate rate here is biased downwards. Re-run without `--sample` "
        f"before quoting a uniqueness number.\n" if args.sample else "")

    return f"""# Data Quality Report

Generated by `python src/data_quality.py`. Machine-readable copy:
`reports/data_quality.json`. Figure: `reports/figures/data_quality.png`.

## Verdict

**Quality score: {summary['score']:.1f}/100** over {summary['n_checks']} checks
({summary['counts'][FAIL]} FAIL, {summary['counts'][WARN]} WARN,
{summary['counts'][PASS]} PASS, {summary['counts'][INFO]} INFO).

Checks are weighted (2.0 for anything that can invalidate a training run, 1.0
for the rest); a PASS earns full weight, a WARN half, a FAIL none. The score is
not a grade for the modelling work - it is a statement about the inputs.

Failing checks:

{fail_list}

Running `python src/data_quality.py --strict` exits **{1 if fails else 0}** on this
data, so it can be used as a pre-training gate.
{sample_warning}

{embed(fig_path, "data quality dashboard")}

## 1. Integrity - which bytes were profiled

{md_table(fp)}

{section(["integrity.fingerprint", "integrity.unchanged"])}

## 2. Schema conformance

Expected schema (the contract these files are checked against, defined as
`EXPECTED_SCHEMA` in `src/data_quality.py`):

{md_table(dict_rows)}

{section(["schema.columns", "schema.label_provenance"])}

## 3. Completeness

{md_table(comp)}

{section(["completeness.nulls"])}

## 4. Uniqueness

{section(["uniqueness.exact_duplicates", "uniqueness.duplicate_titles",
          "uniqueness.near_duplicates", "uniqueness.shared_openings"])}

Highest-similarity near-duplicate pairs found in the sample:

{near_section}

## 5. Validity

{section(["validity.dates", "validity.title_is_url", "validity.encoding",
          "validity.length", "validity.formatting"])}

Encoding profile per column:

{md_table(encoding.set_index("file"))}

## 6. Consistency

{md_table(subject_ct, floatfmt="{:.0f}")}

{section(["consistency.subject_label", "consistency.publisher_fingerprint",
          "consistency.date_label", "consistency.class_balance"])}

## 7. Distribution drift - are these two files one population?

The question this section answers is not "do fake and real articles differ" -
of course they do. It is the narrower and more damaging one: **were these two
files drawn from the same sampling frame?** If they were not, a classifier can
separate them by provenance instead of by truthfulness, and in-distribution
accuracy cannot tell the difference.

{section(["drift.length", "drift.vocabulary", "drift.dates"])}

{textwrap.fill(
    "The pattern across the three drift checks is the informative part. Length "
    "is the one dimension on which the two files agree, and it is also the one "
    "dimension a scraper cannot control. Everything that carries provenance - "
    "topic taxonomy, publication window, vocabulary - separates almost "
    "perfectly. That is the signature of two independently collected corpora "
    "that were concatenated and relabelled, not of one population split by a "
    "truth label.", 88)}

## 8. All checks

{md_table(checks_frame(checks), floatfmt="{:.1f}")}

## 9. Using this as a pipeline gate

```
python src/data_quality.py --strict && python src/train.py
```

`--strict` returns 1 when any check FAILs, so `train.py` never runs on data
that has not been profiled. On the dataset as shipped that gate is closed, and
correctly so: {len(fails)} checks fail.

`reports/data_quality.json` carries the same verdicts in machine-readable form -
`gate.strict_exit_code`, `score`, and a `metrics` block per check - so a CI job
can read a specific number rather than grep this file.

{textwrap.fill(
    "Two honest ways forward, and one dishonest one. Fix the data: deduplicate "
    "before splitting rather than after, drop the scraper-failure rows, "
    "restrict both classes to the shared date window, and retrain with "
    "`--strip-boilerplate`. Or keep the data as it is and report every accuracy "
    "figure next to the one-rule baseline it has to beat. What is not "
    "defensible is quoting the headline accuracy on its own, because on this "
    "dataset that number is mostly a measure of how well the model recognises "
    "a wire service.", 88)}

Rows profiled: {n_rows:,}{" (random sample)" if args.sample else ""}.
Every threshold used is recorded in `reports/data_quality.json` under
`thresholds`, so the calibration can be argued with directly.
"""


# --- main ------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    ensure_dirs()
    figs = not args.no_figures
    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    json_path = REPORTS_DIR / "data_quality.json"

    previous = None
    if json_path.exists():
        try:
            previous = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            previous = None

    print("[1/8] Fingerprinting source files...")
    frames, fingerprints = {}, []
    for fname in SOURCE_FILES:
        path = data_dir / fname
        if not path.exists():
            raise SystemExit(f"{path} not found. Run `python src/setup_data.py` first.")
        fp = file_fingerprint(path)
        frames[fname] = read_source(path)
        fp["rows"] = len(frames[fname])
        fingerprints.append(fp)
        print(f"    {fname:<10} {fp['rows']:>6,} rows  {fp['bytes'] / 1e6:>6.1f} MB  "
              f"sha256={fp['sha256'][:16]}...")

    checks: list[Check] = []
    checks += integrity_checks(fingerprints, previous)

    print("[2/8] Schema conformance...")
    checks += schema_checks(frames)

    print("[3/8] Completeness...")
    comp_checks, completeness = completeness_checks(frames)
    checks += comp_checks

    print("[4/8] Loading combined corpus + text statistics...")
    df = load_data(args.data_dir, drop_duplicates=False)
    if args.sample and args.sample < len(df):
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        print(f"    sampled down to {len(df):,} rows")
    df = add_text_features(df, "content")
    print(f"    {len(df):,} rows profiled")

    print("[5/8] Uniqueness (exact, titles, MinHash near-duplicates)...")
    uniq_checks, near_table = uniqueness_checks(df, args.near_dup_sample,
                                                args.near_dup_threshold)
    checks += uniq_checks

    print("[6/8] Validity (dates, encoding, length, formatting)...")
    val_checks, parsed_dates, encoding = validity_checks(df, frames)
    checks += val_checks

    print("[7/8] Consistency + drift...")
    cons_checks, subject_ct = consistency_checks(df, parsed_dates)
    checks += cons_checks
    dr_checks, drift_payload = drift_checks(df, parsed_dates, args.vocab_sample)
    checks += dr_checks

    print("[8/8] Scoring, figure, report...")
    summary = score_checks(checks)

    # One number per defect, for the figure: "how many rows would I touch if I
    # actually fixed this?" is the question that decides what to fix first.
    m = {c.id: c.metrics for c in checks}
    defect_counts = {
        "duplicate titles": m["uniqueness.duplicate_titles"]["duplicate_titles"],
        "exact duplicate articles": m["uniqueness.exact_duplicates"]["duplicate_rows"],
        "identical opening, different text":
            m["uniqueness.shared_openings"]["beyond_exact_duplicates"],
        "blank article body": m["validity.length"]["empty_body"],
        "under 20 words": m["validity.length"]["short"],
        "over 6000 words": m["validity.length"]["long"],
        "all-caps article": m["validity.formatting"]["all_caps"],
        "URL in the title field": m["validity.title_is_url"]["url_titles"],
        "unparseable date": m["validity.dates"]["unparseable"],
    }
    fig_path = plot_dashboard(checks, summary, drift_payload, defect_counts, figs)

    n_fail = summary["counts"][FAIL]
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": "src/data_quality.py",
        "dataset": {"dir": str(data_dir), "files": fingerprints,
                    "rows_profiled": int(len(df))},
        "score": round(summary["score"], 2),
        "status_counts": summary["counts"],
        "gate": {"strict_exit_code": 1 if n_fail else 0, "n_fail": n_fail},
        "thresholds": THRESHOLDS,
        "expected_schema": EXPECTED_SCHEMA,
        "checks": [{
            "id": c.id, "category": c.category, "name": c.name, "status": c.status,
            "weight": c.weight, "summary": c.summary, "remedy": c.remedy,
            "metrics": jsonable(c.metrics),
        } for c in checks],
    }
    json_path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")

    md_path = REPORTS_DIR / "data_quality_report.md"
    md_path.write_text(
        build_report(checks, summary, fingerprints, completeness, subject_ct,
                     near_table, encoding, len(df), fig_path, args),
        encoding="utf-8")

    print(f"\nWrote {md_path}")
    print(f"      {json_path}")
    if fig_path:
        print(f"      {FIGURES_DIR / 'data_quality.png'}")

    print(f"\nQuality score: {summary['score']:.1f}/100  "
          f"({summary['counts'][FAIL]} FAIL, {summary['counts'][WARN]} WARN, "
          f"{summary['counts'][PASS]} PASS, {summary['counts'][INFO]} INFO)")
    for c in sorted(checks, key=lambda c: (STATUS_ORDER[c.status], c.id)):
        if c.status in (FAIL, WARN):
            print(f"  {c.status:<4} {c.id:<34} {c.name}")

    if args.strict and n_fail:
        print(f"\n--strict: {n_fail} checks FAILED, exiting 1. "
              "Do not train on this data without recording why.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
