# Data Management Plan

How data enters this project, what shape it is in, where it goes, how long it
stays, and what is known to be wrong with it.

The analysis side of this project is documented in [reports/eda_report.md](../reports/eda_report.md)
and [reports/evaluation_report.md](../reports/evaluation_report.md). This
document covers the *management* side: provenance, schemas, quality gates,
versioning, retention and governance.

---

## 1. Data asset inventory

| Asset | Location | Format | Size | Version control | Owner |
|---|---|---|---|---|---|
| Raw corpus (bundled) | `archive.zip` | ZIP of 2 CSVs | 41 MB | Committed | Project |
| Fake articles | `data/Fake.csv` | CSV, UTF-8 | 62.8 MB | Ignored, derived | Derived |
| Real articles | `data/True.csv` | CSV, UTF-8 | 53.6 MB | Ignored, derived | Derived |
| Second corpus (optional) | `data/WELFake_Dataset.csv` | CSV | ~230 MB | Ignored, external | Kaggle |
| Fitted vectorizer | `models/tfidf_vectorizer.joblib` | joblib | ~1 MB | Ignored, derived | Derived |
| Trained models | `models/*_model.joblib` | joblib | ~1-15 MB | Ignored, derived | Derived |
| Run metadata | `models/run_metadata.json` | JSON | <5 KB | Ignored, derived | Derived |
| Metrics | `models/metrics.json`, `metrics.txt` | JSON / text | <20 KB | Ignored, derived | Derived |
| Analysis outputs | `reports/**` | MD / CSV / PNG | ~1 MB | **Committed** | Derived |
| Prediction log | `reports/predictions.db` | SQLite | grows | Ignored, runtime | Runtime |

**Why `reports/` is committed but `data/` and `models/` are not.** Analysis
outputs are small, are the actual deliverable, and let a reader inspect the
findings without a 116 MB download and a training run. Datasets and model
binaries are large, regenerable, and would bloat git history — they are
reproducible from `archive.zip` plus a seeded command.

---

## 2. Lineage

```
archive.zip                                   (committed, immutable, SHA-256 verified)
    │
    │  src/setup_data.py           extract + checksum
    ▼
data/Fake.csv, data/True.csv                  (raw layer - never modified in place)
    │
    │  src/data_quality.py         schema / completeness / validity gate
    ▼
quality report + pass|warn|fail               (reports/data_quality.json)
    │
    │  preprocessing.load_data()   label, concat title+body, drop empty, dedupe
    ▼
in-memory canonical frame                     (content, label, title, body, subject, date)
    │
    │  preprocessing.clean_text()  lowercase, strip URLs/HTML/digits, stopwords
    ▼
cleaned text ──► train_test_split(seed) ──► TfidfVectorizer.fit(TRAIN ONLY)
    │                                              │
    │                                              ▼
    │                                    models/*.joblib + run_metadata.json
    ▼
reports/*.md, reports/figures/*.png, reports/eda_tables/*.csv
    │
    │  app.py / src/predict.py --log
    ▼
reports/predictions.db                        (serving layer + human feedback)
    │
    │  src/db.py --export-corrections
    ▼
corrections CSV ──────────────────────────────► back into training (feedback loop)
```

**Transformation boundary rule.** The raw layer is read-only. Every derived
artifact is reproducible by re-running a documented command with a fixed seed;
nothing is edited by hand. If a derived file disagrees with its source, the
source wins and the derived file is regenerated.

---

## 3. Data dictionary — source corpus

Both `Fake.csv` and `True.csv` share this schema. The `label` column does not
exist in the files; it is assigned at load time by file of origin.

| Column | Type | Null? | Description | Constraints / notes |
|---|---|---|---|---|
| `title` | string | no | Article headline | May be empty string; not unique |
| `text` | string | no | Article body | May be empty; ~5,795 exact duplicates across the corpus |
| `subject` | categorical | no | Publisher's own topic tag | 7 values, **disjoint across classes** — see §8 |
| `date` | string | no | Publication date, free text | 99.98% parse; formats vary; trailing whitespace in `True.csv` |
| `label` | int | no | **Assigned at load**: 0 = fake, 1 = real | From source file, not from the data |

Derived at load time by `preprocessing.load_data()`:

| Column | Derivation |
|---|---|
| `content` | `title + " " + text`, stripped — this is what models are trained on |
| `body` | alias of `text` |

Subject values by class:

| Class | Subjects |
|---|---|
| fake (0) | `News`, `politics`, `left-news`, `Government News`, `US_News`, `Middle-east` |
| real (1) | `politicsNews`, `worldnews` |

The sets do not intersect. `subject` is therefore a perfect predictor of the
label and must never be used as a feature. It is retained for analysis and
error slicing only.

---

## 4. Prediction store — schema and rationale

SQLite (`reports/predictions.db`), chosen because it is in the Python standard
library: no server, no install, one file that can be copied between machines or
deleted to reset. For a single-user analytical app with modest write volume,
a client-server RDBMS would add operational cost for no benefit.

### DDL

```sql
CREATE TABLE predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT    NOT NULL,   -- ISO-8601 UTC, second precision
    source        TEXT    NOT NULL,   -- text | url | image | csv | cli
    origin        TEXT,               -- the URL / filename, when there is one
    model         TEXT    NOT NULL,   -- registry key: nb | svm | lr
    input_hash    TEXT    NOT NULL,   -- sha256 of the raw input
    text_preview  TEXT,               -- first 500 chars only
    char_count    INTEGER,
    word_count    INTEGER,
    prediction    INTEGER NOT NULL,   -- 0 fake, 1 real
    label         TEXT    NOT NULL,   -- 'FAKE' | 'REAL', denormalised
    confidence    REAL,
    proba_fake    REAL,
    proba_real    REAL,
    top_terms     TEXT,               -- JSON array of [term, contribution]
    user_feedback TEXT,               -- NULL | 'correct' | 'incorrect'
    true_label    INTEGER             -- set when a human supplies the answer
);
CREATE INDEX idx_pred_created ON predictions(created_at);
CREATE INDEX idx_pred_model   ON predictions(model);
CREATE INDEX idx_pred_hash    ON predictions(input_hash);
```

### Design decisions

**Single table, deliberately denormalised.** A textbook normalisation would
split out `model` and `source` into lookup tables. That is not done here, and
the reason is explicit: this is an append-mostly analytical log, not a
transactional system. The cardinality of `model` is 3 and of `source` is 5;
lookup tables would add joins to every query to save a few kilobytes and would
introduce referential-integrity maintenance for no operational benefit. The
denormalisation is a considered choice, not an oversight.

**`label` is redundant with `prediction`.** Kept because every read path wants
the human-readable form, and a 4-byte string beats a CASE expression in each
query. The redundancy is safe because both are written in the same statement
and neither is ever updated.

**`input_hash` rather than the full text.** Storing SHA-256 of the raw input
allows duplicate-submission detection and lets a prediction be traced back to a
specific input without retaining the full document indefinitely. `text_preview`
is capped at 500 characters — enough for a human to recognise the item in the
History tab, short of retaining whole third-party articles.

**`top_terms` as JSON in a TEXT column.** The explanation is variable-length and
is only ever read back as an opaque blob for display. A separate
`prediction_terms` child table would be the normalised form; it would triple the
row count for data that is never queried relationally.

**Indexes.** `created_at` supports the History tab's recency ordering,
`model` supports per-model filtering, `input_hash` supports duplicate detection.
No index on `user_feedback` — that column is low cardinality and the corrections
query scans a small table.

### Access paths

| Operation | Function | Notes |
|---|---|---|
| Insert | `db.log_prediction()` | One row per classification |
| Read recent | `db.fetch_predictions(limit, model, label, since)` | Parameterised, newest first |
| Aggregate | `db.summary_stats()` | Counts, mean confidence, review rate |
| Annotate | `db.record_feedback(id, correct=, true_label=)` | Raises if no verdict given |
| Export all | `db.export_csv(path)` | Full dump |
| Export training data | `db.export_corrections(path)` | Human-labelled rows only |
| Reset | `db.clear()` | Interactive confirmation in the CLI |

All queries use parameter binding (`?` placeholders). No string interpolation of
user values into SQL anywhere in `db.py`.

---

## 5. Quality gates

`src/data_quality.py` profiles the raw corpus and emits `reports/data_quality.json`
with a per-check `pass` / `warn` / `fail` verdict. Run with `--strict` it exits
non-zero on any failure, so it can gate training in CI:

```bash
python src/data_quality.py --strict && python src/train.py
```

Checks cover schema conformance, completeness, uniqueness (exact and near
duplicates), validity (dates, encoding, degenerate text), cross-class
consistency, distributional drift between the two source files, and file
integrity (size + SHA-256).

Two failures are **expected and permanent** on this corpus — they are properties
of the dataset, not bugs to fix:

1. `subject` values are disjoint across classes.
2. The two source files are not samples from one population (see §8).

---

## 6. Versioning and reproducibility

Every training run writes `models/run_metadata.json` recording the dataset
options, split parameters, seed, feature settings, package versions, platform,
and the exact command line. `src/evaluate.py`, `src/significance.py`,
`src/error_taxonomy.py` and `src/alt_models.py` all read it to rebuild the
identical test split, so every reported number refers to the same rows.

| Dimension | Mechanism |
|---|---|
| Data version | SHA-256 of each source CSV, recorded by `setup_data.py` and `data_quality.py` |
| Code version | git commit |
| Split | `random_state` + `test_size` in `run_metadata.json`, stratified |
| Model params | full parameter set in `run_metadata.json` and `model_card.md` |
| Environment | Python, scikit-learn version and platform string in `run_metadata.json` |

Seeds are fixed at 42 throughout. `Date.now()`-style nondeterminism is confined
to the `trained_at` timestamp, which is metadata only and never feeds a
computation.

---

## 7. Retention, privacy and access

**Source corpus.** Public news articles published 2015–2018, redistributed on
Kaggle. No personal data beyond what appeared in published journalism. Retained
for the life of the project.

**Prediction log.** Contains whatever users submit. This is the only place the
project stores data it did not ship with, so it is the only genuine privacy
surface.

| Concern | Handling |
|---|---|
| What is stored | First 500 characters of input, a SHA-256 of the full input, the prediction, and the explanation terms |
| What is not stored | Full document text, any user identifier, IP address, or session token |
| Location | Local file only; never transmitted anywhere |
| Retention | Indefinite by default — this is a **known gap**, see below |
| Deletion | `python src/db.py --clear` wipes the store; individual rows can be deleted by `id` via SQL |
| Access control | Filesystem permissions only; the app has no authentication |

**Known gaps, stated rather than hidden:**

- No automatic retention window. A production deployment should expire rows
  older than a defined period; here they accumulate until manually cleared.
- No authentication on the Streamlit app. Anyone who can reach the port can read
  the whole prediction history, including other people's submitted text. Acceptable
  for local/demo use, not for a shared deployment.
- URL scraping fetches third-party pages server-side. Respect the target site's
  terms; the scraper sends a identifying User-Agent and does not attempt to
  bypass paywalls or bot protection.

---

## 8. Known data issues register

These are properties of the corpus. They are documented rather than silently
worked around, because they determine how every result must be read.

| ID | Issue | Severity | Evidence | Handling |
|---|---|---|---|---|
| DQ-1 | **Publisher leakage.** 99.5% of real articles contain `(Reuters)`; ~0% of fake ones do | **Critical** | `reports/eda_report.md` §7; a one-rule classifier scores 99.4%, beating every trained model | `strip_source_boilerplate()` + `train.py --strip-boilerplate`; leakage baseline reported next to every headline metric |
| DQ-2 | **Disjoint subjects.** 7 of 7 subject categories appear in exactly one class | **Critical** | `reports/eda_tables/subject_by_class.csv` | `subject` excluded from features; used for error slicing only |
| DQ-3 | **Disjoint date ranges.** Fake spans 2015-03-31 to 2018-02-19, real only 2016-01-13 to 2017-12-31, so every article before 2016-01-13 is fake | **High** | `src/temporal_eval.py` | Quantified and reported; date excluded from features |
| DQ-4 | **Duplicates.** 5,795 exact duplicate articles (12.9%) | High | `load_data(verbose=True)` | Dropped **before** the train/test split, so a document cannot appear on both sides |
| DQ-5 | **Single-source classes.** The two halves were scraped from different places, so the task is partly "which newsroom wrote this" | **Critical** | Drift checks in `data_quality.py` | Cross-dataset evaluation (§9) is the only valid generalisation estimate |
| DQ-6 | Free-text dates in inconsistent formats, trailing whitespace in `True.csv` | Low | 99.98% parse rate | Parsed with `errors="coerce"`; unparseable rows excluded from time analysis only |
| DQ-7 | Corpus is US/world politics, 2015–2018 only | Medium | `reports/figures/articles_over_time.png` | Stated as a scope limit in the model card and the app |

**The combined implication of DQ-1, DQ-2, DQ-3 and DQ-5:** in-distribution
accuracy on this dataset is close to meaningless. Three independent columns
(`text`, `subject`, `date`) each leak the label, because the two classes are two
different corpora rather than two samples from one. Any figure quoted from the
random test split is an upper bound on a task substantially easier than the real
one.

---

## 9. Outstanding work

**Cross-dataset evaluation is not yet run.** This is the single most important
missing number. `src/cross_dataset_eval.py` is implemented and tested but needs a
second, independently-sourced corpus:

```bash
# download WELFake_Dataset.csv from Kaggle into data/, then:
python src/cross_dataset_eval.py --data-file data/WELFake_Dataset.csv
```

Until that runs, the honest statement of what this system can do is bounded by
the temporal and stylometry-only results in `reports/temporal_report.md` and
`reports/alt_models_report.md`, not by the 99% on the random split.
