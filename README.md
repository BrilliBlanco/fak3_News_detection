# Fake News Detection

A TF-IDF + classical ML pipeline (Naive Bayes / Linear SVM / Logistic
Regression) that classifies a news article as **FAKE** or **REAL**, returns a
confidence score, and — the part that matters — **shows the exact terms behind
every decision**.

Runs identically on Windows, macOS and Linux (tested on Kali). Plain Python +
scikit-learn, no OS-specific code, no downloads needed at run time.

> **New here?** Read [PROJECT_GUIDE.md](PROJECT_GUIDE.md) — it walks through
> running every command, what each output means, and how to debug the things
> that commonly break.

---

## The headline finding

The models score **99.3% accuracy** on the ISOT test split. That number is
close to meaningless, and the project now proves why:

> 99.5% of the "real" articles contain the string `(Reuters)`. Almost none of
> the fake ones do. A one-line rule — *"mentions Reuters ⇒ real"* — scores
> **99.4%** on the same test split, **beating every trained model.**

So most of the apparent skill is publisher fingerprinting, not fake-vs-real
detection. `src/eda.py` quantifies it, `src/explain.py` shows it in the model
weights (`reuters` carries ~4× the weight of any other feature), and
`src/train.py --strip-boilerplate` lets you retrain without it.

Being able to *find, measure and explain* that is the actual result of this
project. Reporting 99% and stopping would have been the failure.

### Five independent lines of evidence

Every number below is reproducible from a command in the table further down.

| Evidence | Result | Source |
|---|---|---|
| The keyword rule is **not statistically distinguishable** from the best model | 0.9932 vs 0.9940; 78 discordant articles split 36/42; Holm-adjusted **p = 0.57** | `significance.py` |
| **50 features are as good as 5,000** — the ablation curve is flat, so there is nothing to learn | SVM 0.9898 @ 50 features vs 0.9932 @ 5,000. Strip the boilerplate and the same curve climbs 0.9105 → 0.9847, i.e. the model finally has to work | `tune.py` |
| `reuters` is the **6th term** the vectorizer keeps when the vocabulary is squeezed | rank 6 of 50 | `tune.py` |
| A model that **cannot see** the fingerprint scores far lower — and is unmoved by removing it | stylometry-only 0.8832 → 0.8841 (−0.0009), while the keyword stump collapses 0.9940 → 0.5421 (chance) | `alt_models.py` |
| The label leaks through **three** independent columns, not one | `text` (Reuters tag), `subject` (7/7 categories single-class), `date` (every article before 2016-01-13 is fake) | `eda.py`, `temporal_eval.py` |

**The honest headline: ~0.88, not 0.99.** The stylometry-only model is the only
one here structurally incapable of reading the publisher fingerprint, so its
accuracy is the closest thing this dataset offers to a real difficulty estimate
— pending the cross-dataset test below.

---

## Quick start

```bash
pip install -r requirements.txt
python src/setup_data.py     # extracts Fake.csv / True.csv from archive.zip
python src/train.py          # trains all three models (~35s)
python -m streamlit run app.py   # the demo UI
```

> Commands are written as `python -m streamlit` / `python -m pytest` rather
> than bare `streamlit` / `pytest`. When pip installs into a *user* directory
> — which it does whenever the active Python (e.g. conda `base`) isn't
> writable — the `.exe` launchers land somewhere that isn't on PATH, and the
> bare commands fail with *"is not recognized"*. The module form always works.
> See [PROJECT_GUIDE.md §5](PROJECT_GUIDE.md#5-debugging-common-errors).

Full setup, per-OS notes and troubleshooting: [PROJECT_GUIDE.md](PROJECT_GUIDE.md).

---

## What each script does

| Command | Purpose |
|---|---|
| `python src/setup_data.py` | Extract + verify the dataset from `archive.zip` into `data/` |
| `python src/data_quality.py` | 21 schema / completeness / validity / drift checks; `--strict` gates training |
| `python src/eda.py` | Exploratory analysis → `reports/eda_report.md`, figures, tables, **leakage audit** |
| `python src/train.py` | Train NB / SVM / LR → `models/` + metrics + model card |
| `python src/evaluate.py` | ROC, PR, calibration, threshold sweep, error analysis |
| `python src/significance.py` | McNemar + paired bootstrap CIs — **are the model differences real?** |
| `python src/temporal_eval.py` | Train on older articles, test on newer ones |
| `python src/tune.py` | GridSearchCV + the feature-count ablation |
| `python src/alt_models.py` | Char n-gram, stylometry-only, non-linear, and leakage-only baselines |
| `python src/error_taxonomy.py` | Categorised error analysis, not just a dump |
| `python src/predict.py` | Classify text, a file, a URL, or a whole CSV |
| `python src/explain.py` | Per-prediction and global feature attributions |
| `python src/cross_dataset_eval.py` | Score the trained models on a *second* dataset — the only honest generalization test |
| `python src/db.py` | Inspect / export the prediction log |
| `python -m pytest -q` | 78 tests |
| `python -m streamlit run app.py` | Demo UI: Classify, Batch, Data, Evidence, Model, History |

Data management — provenance, schemas, retention, governance and the
known-issues register — is documented in [docs/DATA_MANAGEMENT.md](docs/DATA_MANAGEMENT.md).

Every script supports `--help`.

---

## 1. Get the dataset

The repo ships `archive.zip` (the Kaggle
["Fake and Real News Dataset"](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)),
so one command is enough:

```bash
python src/setup_data.py
```

It extracts `Fake.csv` and `True.csv` into `data/`, prints sizes and SHA-256
prefixes, and refuses to clobber existing files unless you pass `--force`.
`data/*.csv` is git-ignored so the ~116 MB of CSVs never gets committed.

## 2. Set up the environment

**Linux / Kali / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> If PowerShell blocks the activation script, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Everyone should use the same `requirements.txt` so we get identical package
versions — avoids "works on my machine" bugs between Kali and Windows.

## 3. Explore the data first

```bash
python src/eda.py
```

Writes `reports/eda_report.md` plus figures and CSV tables covering class
balance, duplicate rate (12.9% of ISOT is duplicated), subject overlap, length
distributions, publication timeline, writing-style markers, the most
discriminative terms per class, and the **label-leakage audit**.

Options: `--sample 5000` (faster while iterating), `--no-figures`,
`--keep-duplicates`, `--top-n 30`.

## 4. Train the models

```bash
python src/train.py
```

Loads and cleans the data, drops duplicates *before* splitting, vectorizes
with TF-IDF, trains all three models, and saves to `models/`:
`tfidf_vectorizer.joblib`, the three `*_model.joblib` files, `metrics.txt`,
`metrics.json`, `run_metadata.json` and `model_card.md`.

It also reports two baselines on the same test split — majority class, and the
one-keyword Reuters rule — so the headline accuracy always arrives with
context.

Useful flags:

```bash
python src/train.py --strip-boilerplate   # remove publisher fingerprints, then compare
python src/train.py --cv 5                # add 5-fold cross-validation
python src/train.py --max-features 20000 --ngram-max 3 --min-df 3
```

Measured on this dataset (SVM): **99.32% → 98.49%** after stripping
boilerplate, while the Reuters one-rule baseline collapses from 99.40% to
45.79%. The residual 98.5% is *not* proof the model is fine — the two halves
of ISOT still come from entirely different newsrooms (their `subject`
categories don't overlap at all). Only step 6 settles it.

## 5. Evaluate properly

```bash
python src/evaluate.py --cv 3 --learning-curve
```

Rebuilds the exact test split from `models/run_metadata.json` (so the numbers
match `metrics.json`) and produces ROC/PR curves, **calibration curves** — is
"90% confident" right 90% of the time? — confusion matrices, a decision
threshold sweep, a learning curve, and per-model CSVs of the mistakes the
model was *most confident* about.

## 6. Test generalization — the number that counts

Download **WELFake** from
[Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification),
put the CSV in `data/`, then:

```bash
python src/cross_dataset_eval.py --data-file data/WELFake_Dataset.csv
```

Different sources, different newsrooms, different writing style. In-distribution
accuracy cannot detect publisher leakage; this can. Whatever it reports is the
number to quote when describing what the system actually does.

## 7. Predict

```bash
# single input
python src/predict.py --text "Breaking: scientists confirm the moon is made of cheese"
python src/predict.py --file article.txt --model svm --explain
python src/predict.py --url https://example.com/story --json
python src/predict.py --text "..." --model all        # compare all three

# batch — score an entire CSV
python src/predict.py --csv inbox.csv --text-column content --out scored.csv
python src/predict.py --csv labelled.csv --label-column label   # also reports accuracy
```

Add `--log` to record predictions in the local store, `--threshold 0.7` to
require more evidence before calling something REAL.

Output:
```
Prediction: FAKE
Confidence: 99.99%
Matched 7 vocabulary features (from 6 distinct words after cleaning)

Top contributing terms:
  term                     tfidf   weight    effect  pushes
  breaking                 0.338   -2.796   -0.9443  FAKE
  mainstream media         0.419   -0.432   -0.1809  FAKE
  ...
```

## 8. Explain

```bash
python src/explain.py --text "..." --model lr        # why this prediction
python src/explain.py --global --model svm           # what the model leans on overall
python src/explain.py --text "..." --compare         # all three models side by side
```

All three models are linear over TF-IDF, so a prediction decomposes *exactly*:
`score = bias + Σ (tfidf(term) × weight(term))`. Each term's contribution is
real arithmetic, not a LIME/SHAP approximation.

## 9. Manage the prediction log

Every prediction from the CLI (`--log`) or the app is recorded in
`reports/predictions.db` (SQLite, standard library, no server).

```bash
python src/db.py --stats                                   # aggregates
python src/db.py --recent 20                               # latest predictions
python src/db.py --export reports/predictions.csv          # full dump
python src/db.py --feedback 12 --correct-label fake        # human verdict
python src/db.py --export-corrections data/corrections.csv # training-ready
python src/db.py --clear                                   # reset (asks first)
```

The feedback loop is the point: review flagged items, mark them right or
wrong, export the corrections, and you have new labelled data drawn from the
distribution the model is *actually used on* rather than from ISOT.

## 10. Run the demo UI

```bash
python -m streamlit run app.py
```

Five tabs:

- **Classify** — paste text, a URL, or a screenshot (OCR); get the prediction,
  confidence, the contributing terms as a chart, style signals, and thumbs
  up/down feedback
- **Batch** — upload a CSV, score every row, download the result
- **Dataset** — the EDA findings and the leakage banner
- **Model** — metrics, baselines, global feature weights, evaluation figures, model card
- **History** — every prediction made, analytics, CSV export

---

## Project structure

```
fake-news-detection/
├── app.py                      Streamlit UI (6 tabs)
├── archive.zip                 bundled dataset
├── data/                       Fake.csv / True.csv (git-ignored)
├── models/                     trained artifacts (git-ignored)
├── docs/DATA_MANAGEMENT.md     provenance, schemas, retention, issues register
├── reports/                    generated analysis - committed
│   ├── eda_report.md           exploratory analysis + leakage audit
│   ├── evaluation_report.md    curves, calibration, thresholds
│   ├── data_quality_report.md  21 quality checks (+ .json for CI)
│   ├── significance_report.md  McNemar + bootstrap CIs
│   ├── temporal_report.md      older-vs-newer validation
│   ├── tuning_report.md        grid search + feature ablation
│   ├── alt_models_report.md    alternative representations
│   ├── error_taxonomy.md       categorised errors
│   ├── eda_tables/*.csv, error_tables/*.csv, figures/*.png
│   └── predictions.db          runtime log (git-ignored)
├── src/
│   ├── config.py               paths, label convention, model registry
│   ├── setup_data.py           dataset extraction + verification
│   ├── preprocessing.py        cleaning, leakage guard, stats, loading
│   ├── data_quality.py         schema / completeness / validity / drift gate
│   ├── eda.py                  exploratory analysis + leakage audit
│   ├── train.py                training pipeline + model card
│   ├── evaluate.py             curves, calibration, thresholds, errors
│   ├── significance.py         McNemar, Holm, paired bootstrap
│   ├── temporal_eval.py        temporal split + date-leakage analysis
│   ├── tune.py                 GridSearchCV + feature ablation
│   ├── alt_models.py           char n-gram / stylometry / non-linear baselines
│   ├── error_taxonomy.py       error categorisation
│   ├── explain.py              feature attribution
│   ├── predict.py              CLI: single + batch scoring
│   ├── cross_dataset_eval.py   generalization test
│   ├── extract.py              URL scraping + OCR
│   └── db.py                   SQLite prediction store
├── tests/
│   ├── test_pipeline.py        core pipeline
│   └── test_analysis_modules.py  the statistics (McNemar, Holm, bootstrap)
├── PROJECT_GUIDE.md            run + debug guide
├── requirements.txt
└── README.md
```

## Notes for the team

- Don't commit `data/` or `models/` — they're git-ignored (large binaries).
  `reports/` **is** committed so everyone can read the analysis without
  running the pipeline.
- All paths, filenames and the label convention live in `src/config.py`.
  Change them there, not in five places.
- `clean_text()` is what the models were trained on. If you change it, you
  **must** retrain — a vectorizer fitted on differently-cleaned text produces
  silently wrong features.
- Label convention is `0 = fake, 1 = real` everywhere.
- `LinearSVC` has no `predict_proba`, so it's wrapped in `CalibratedClassifierCV`
  for real probabilities instead of raw decision-function distances.
- Run `python -m pytest -q` before pushing. Tests that need trained models skip
  themselves on a fresh clone.
- If git shows every line as changed on Windows:
  `git config --global core.autocrlf true` (Windows) or
  `core.autocrlf input` (Linux/Kali).

## Screenshot input (OCR)

Needs the Tesseract engine installed system-wide, separate from the
`pytesseract` pip package:

- **Linux/Kali:** `sudo apt install tesseract-ocr`
- **Windows:** [UB-Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki),
  then add the install folder to PATH

If it's missing, the Screenshot tab disables itself with an explanation — the
rest of the app keeps working.

## Deployment

Vercel isn't a good fit (it targets JS/serverless, not Python + scikit-learn +
Streamlit with model files). Better:

- **Streamlit Community Cloud** — free, connect the GitHub repo, done. Recommended.
- **Hugging Face Spaces** — also free and Streamlit-native.

Both need `models/` to exist. Since it's git-ignored, either train on first run
or commit the artifacts to a deployment branch. OCR won't work on either host
unless it has Tesseract — leave it as a run-locally feature and demo the text
and URL tabs.

## Limitations — state these before anyone asks

1. **It does not fact-check.** It matches writing style and vocabulary against
   its training corpus. A well-written lie reads REAL; a badly-written truth
   reads FAKE.
2. **The training data leaks the label** (see above). Any accuracy quoted from
   the ISOT split is an upper bound on a task easier than the real one.
3. **Topic- and period-bound.** US/world politics, 2015–2018. Performance
   elsewhere is unmeasured.
4. **In-distribution evaluation proves little.** `cross_dataset_eval.py` is the
   real test.

## Why this instead of just asking ChatGPT?

See [question.txt](question.txt) — and note that this repo now *demonstrates*
the argument rather than just asserting it. The pipeline is inspectable
(`explain.py` prints the weights behind any decision), deterministic (same
input → same output, forever), free and offline at inference, and — most
importantly — it let us **discover a flaw in the dataset that an API call
would have hidden behind a confident-sounding answer.**
