# Project Guide — Running & Debugging

Everything you need to go from a fresh clone to a working demo, plus what to
do when something breaks. For *what the project is*, see
[README.md](README.md).

---

## Table of contents

1. [First-time setup](#1-first-time-setup)
2. [The full pipeline, in order](#2-the-full-pipeline-in-order)
3. [Daily workflows](#3-daily-workflows)
4. [What every output file is](#4-what-every-output-file-is)
5. [Debugging: common errors](#5-debugging-common-errors)
6. [Debugging: wrong results, not crashes](#6-debugging-wrong-results-not-crashes)
7. [Debugging techniques](#7-debugging-techniques)
8. [Architecture — where to change what](#8-architecture--where-to-change-what)
9. [Before you push](#9-before-you-push)
10. [Demo script](#10-demo-script)

---

## 1. First-time setup

### Requirements

- Python 3.9+ (developed on 3.12)
- ~500 MB free disk (dataset 116 MB + models + figures)
- No internet needed after `pip install` — the dataset ships in `archive.zip`

### Steps

**Linux / Kali / macOS**
```bash
git clone <repo-url> && cd fak3_News_detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
git clone <repo-url> ; cd fak3_News_detection
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell refuses to run the activation script:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Verify the install

```bash
python -c "import pandas, sklearn, matplotlib, streamlit; print('ok')"
```

Optional extras — the app degrades gracefully without them:

| Feature | Needs |
|---|---|
| URL input | `requests`, `beautifulsoup4` (in requirements.txt) |
| Screenshot input | `pytesseract` + `pillow` **and** the Tesseract engine installed system-wide |

Tesseract: `sudo apt install tesseract-ocr` (Kali) or the
[UB-Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki) on
Windows (then add it to PATH).

---

## 2. The full pipeline, in order

Run these top to bottom on a fresh clone. Times are from a mid-range laptop.

```bash
python src/setup_data.py                       # ~5s    -> data/Fake.csv, data/True.csv
python src/data_quality.py                     # ~2min  -> reports/data_quality_report.md
python src/eda.py                              # ~90s   -> reports/eda_report.md + figures
python src/train.py --cv 3                     # ~60s   -> models/
python src/evaluate.py --cv 3 --learning-curve # ~5min  -> reports/evaluation_report.md
python -m pytest -q                            # ~15s   -> 78 passed
python -m streamlit run app.py                 # opens http://localhost:8501
```

The deeper analyses. Each rebuilds the same test split from
`models/run_metadata.json`, so their numbers line up with `metrics.json`:

```bash
python src/significance.py                     # ~1min  -> McNemar + bootstrap CIs
python src/temporal_eval.py                    # ~4min  -> train on older, test on newer
python src/alt_models.py                       # ~6min  -> char/stylometry/non-linear baselines
python src/error_taxonomy.py                   # ~2min  -> categorised errors
python src/tune.py                             # ~10min -> grid search + feature ablation
```

Useful flags: `--strict` (data_quality, exits non-zero on FAIL — use it as a CI
gate before training), `--strip-boilerplate` (temporal_eval), `--quick` /
`--full` (tune), `--sweep` (temporal_eval, tries several cutoffs),
`--bootstrap 0` (significance, skips the slow CIs).

Expected output at each step:

| Step | Looks right when… |
|---|---|
| `setup_data.py` | prints two files, ~62.8 MB and ~53.6 MB |
| `eda.py` | ends with `Key finding: '(Reuters) tag' alone classifies 99.5% …` |
| `train.py` | SVM ≈ 0.993 accuracy, `one_rule_mentions_reuters` ≈ 0.994 |
| `evaluate.py` | prints a 3-row table with `roc_auc` ≈ 0.99+ |
| `pytest` | `78 passed` |
| `streamlit` | five tabs, sidebar warns about boilerplate leakage |

> **Ordering rules.** `train.py` needs `data/`. `evaluate.py` and `explain.py`
> need `models/`. The app's Dataset tab needs `eda.py` to have run. Nothing
> else depends on order.

### Optional: the leakage experiment

```bash
python src/train.py --strip-boilerplate --models-dir models_clean
```

Then compare `models/metrics.json` against `models_clean/metrics.json`. On this
dataset: SVM 99.32% → 98.49%, and the Reuters one-rule baseline collapses from
99.40% to 45.79% (confirming the marker really is gone).

---

## 3. Daily workflows

**Changed `clean_text()` or any preprocessing?**
```bash
python src/train.py && python src/evaluate.py && python -m pytest -q
```
You *must* retrain. A vectorizer fitted on differently-cleaned text will
silently produce garbage features at prediction time — no error, just bad
predictions.

**Changed model hyperparameters only?**
```bash
python src/train.py --max-features 20000 --ngram-max 3
python src/evaluate.py
```

**Added a new model?** Edit two places:
1. `src/config.py` → `MODEL_REGISTRY` (key, display name, filename)
2. `src/train.py` → `build_models()`

Everything else — `predict.py`, `explain.py`, `evaluate.py`, the app's model
dropdown — picks it up automatically. If it isn't linear over TF-IDF,
`explain.get_feature_weights()` will raise a clear `TypeError` and the
explanation panel degrades instead of crashing.

**Just want to test a change to the UI?**
```bash
python -m streamlit run app.py
```
Streamlit hot-reloads on save. If a `@st.cache_resource` loader is stale,
press **C** in the browser (clear cache) or restart.

---

## 4. What every output file is

### `models/`
| File | What it is |
|---|---|
| `tfidf_vectorizer.joblib` | The fitted vectorizer. **Must** pair with the models beside it |
| `naive_bayes_model.joblib`, `svm_model.joblib`, `logistic_regression_model.joblib` | Trained classifiers |
| `metrics.txt` | Human-readable classification reports |
| `metrics.json` | Machine-readable metrics — read by `evaluate.py` and the app |
| `run_metadata.json` | Every setting the run used. Lets `evaluate.py` rebuild the identical split |
| `model_card.md` | Intended use, data, results, limitations. Good report material |

### `reports/`
| File | What it is |
|---|---|
| `eda_report.md` | Full exploratory analysis, ending in the leakage audit |
| `evaluation_report.md` | Curves, calibration, thresholds, error analysis |
| `eda_tables/*.csv` | The same tables as raw CSV, for your own plots |
| `figures/*.png` | All generated figures |
| `errors_{nb,svm,lr}.csv` | The mistakes each model was most confident about |
| `predictions.db` | SQLite log of every prediction made (git-ignored) |

---

## 5. Debugging: common errors

### `pytest : The term 'pytest' is not recognized...` / same for `streamlit`

The package **is** installed — only its `.exe` launcher isn't on your PATH.
pip warns about this during install:

```
WARNING: The script streamlit.exe is installed in
'C:\Users\<you>\AppData\Roaming\Python\Python312\Scripts'
which is not on PATH.
```

This happens when the active Python (e.g. conda `base`) isn't writable, so pip
falls back to a **user** install. Reinstalling won't help — pip correctly
reports "Requirement already satisfied".

**Fix — run them as modules.** Works everywhere, no PATH changes:

```bash
python -m pytest -q
python -m streamlit run app.py
```

All docs in this repo use this form for exactly that reason.

**Optional, if you want the bare commands back**, add the Scripts folder to
PATH for the current PowerShell session:

```powershell
$env:Path += ";$env:APPDATA\Python\Python312\Scripts"
```

To make it permanent (new shells only — reopen PowerShell after):

```powershell
[Environment]::SetEnvironmentVariable("Path", $env:Path + ";$env:APPDATA\Python\Python312\Scripts", "User")
```

**Cleaner long-term fix — use a venv**, which puts everything in one writable
place and sidesteps this entirely:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Inside an activated venv, plain `pytest` and `streamlit` work.

### `FileNotFoundError: Missing Fake.csv, True.csv inside .../data`
The dataset isn't extracted.
```bash
python src/setup_data.py
```
If `archive.zip` is missing, download from
[Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
and drop both CSVs in `data/`.

### `No trained models in models/. Run python src/train.py first.`
Exactly what it says. `train.py` needs `data/` first.

### `ModuleNotFoundError: No module named 'config'` (or `preprocessing`)
You ran a `src/` script from a directory it can't resolve imports from. The
scripts import each other as flat modules, so run them **from the project
root**:
```bash
python src/train.py      # correct
cd src && python train.py  # also works
```
The failure mode is importing `src.train` from somewhere else. If you need
that, add the repo root to `PYTHONPATH`.

### `ModuleNotFoundError: No module named 'bs4'` when starting the app
Fixed — the app now degrades instead of dying, and the URL tab disables itself
with an explanation. If you *want* URL input:
```bash
pip install -r requirements.txt
```

### `TesseractNotFoundError` / "OCR failed"
The `pytesseract` pip package is only a wrapper. Install the actual engine
(see §1), then restart your shell so PATH updates. To point at it explicitly,
set `pytesseract.pytesseract.tesseract_cmd` in `src/extract.py`.

### `ValueError: X has N features, but ... is expecting M features`
The vectorizer and the model came from different training runs. They are a
matched pair.
```bash
python src/train.py     # retrains both together
```

### `Vectorizer has N features but the model has M weights — retrain`
Same cause, raised deliberately by `explain.py` with a clearer message.

### Streamlit: `Port 8501 is already in use`
```bash
python -m streamlit run app.py --server.port 8899
```
Or kill the old process. On Windows:
```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*streamlit*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

### App shows "No trained models found"
The app looks in `models/` relative to the **project root**. Run
`python -m streamlit run app.py` from the root, not from `src/`.

### `MemoryError` / machine crawls during training
Lower the feature count:
```bash
python src/train.py --max-features 2000 --ngram-max 1
```
The `--cv` and `--learning-curve` flags multiply the work — drop them first.

### `pytest` collects 0 tests
Run from the project root: `python -m pytest -q`. Tests requiring trained models skip
themselves (`run python src/train.py first`) rather than fail — that's normal
on a fresh clone.

### Git shows every line as changed
Line endings.
```bash
git config --global core.autocrlf true    # Windows
git config --global core.autocrlf input   # Linux/Kali/macOS
```

---

## 6. Debugging: wrong results, not crashes

These are the bugs that don't announce themselves.

### "Accuracy is 99% — great!"
No. That is the *symptom*. See the leakage finding in the README. Always
report the one-rule baseline next to it; `train.py` prints both.

### "It predicts REAL for obvious garbage"
Check how many features matched:
```bash
python src/predict.py --text "your text" --explain
```
If it says `Matched 2 vocabulary features`, the input barely overlaps the
5,000-term vocabulary and the prediction is near-arbitrary. The app shows an
explicit warning below 5. This is expected for short or off-topic inputs, not
a bug.

### "Confidence is 99.99% on a three-word input"
Also expected, and also a real weakness worth mentioning in the report: TF-IDF
normalises per document, so one strongly-weighted term in a short text
dominates. Look at `reports/figures/calibration_curves.png` — the SVM is
well-calibrated *in aggregate*, which is not the same as being trustworthy on
a single short input.

### "The URL tab returns nonsense"
Scraping limitation, not a model problem. Paywalled and JS-rendered sites
yield navigation text instead of article body. Check what was actually
extracted:
```bash
python src/predict.py --url "https://..." --explain
```
The app's "Text used for classification" expander shows the same thing.

### "Two models disagree"
Normal and worth demoing — Naive Bayes assumes feature independence, which
text badly violates. Compare directly:
```bash
python src/predict.py --text "..." --model all --explain
```

### "Results changed between runs"
They shouldn't. Everything is seeded (`--random-state 42`). If they did:
- Did you retrain between the two runs? Check `models/run_metadata.json` →
  `trained_at` and `command`.
- Did the input change? `db.py` stores a SHA-256 of every logged input:
  `python src/db.py --recent 20`.

---

## 7. Debugging techniques

### Inspect a single stage
```bash
python -c "import sys; sys.path.insert(0,'src'); from preprocessing import clean_text; print(clean_text('YOUR TEXT HERE'))"
```
If `clean_text` returns `''`, everything downstream is meaningless — that's the
first thing to check for any "it predicts nonsense" report.

### See what the model actually keys on
```bash
python src/explain.py --global --model lr --top-n 25
```
If a suspicious token (a publisher, a date, a stopword-ish artifact) sits at
the top, you've found a leakage problem, not a modelling one.

### Reproduce a logged prediction
Every prediction made with `--log` or through the app is stored:
```bash
python src/db.py --recent 20
python src/db.py --export reports/predictions.csv
```
`top_terms` holds the exact attributions from that moment.

### Sanity-check on data you control
```bash
python src/predict.py --csv reports/my_test.csv --label-column label --model all
```
Accuracy on a hand-built CSV of 20 articles you personally judged tells you
far more than another decimal place on the ISOT split.

### Isolate with a smaller run
```bash
python src/eda.py --sample 3000 --no-figures     # seconds instead of minutes
python src/train.py --max-features 500           # fast iteration
```

### Verbose data loading
```bash
python -c "import sys; sys.path.insert(0,'src'); from preprocessing import load_data; load_data(verbose=True)"
```
Prints raw rows, rows dropped for being empty, rows dropped as duplicates, and
final label counts — the fastest way to spot a data-loading regression.

---

## 8. Architecture — where to change what

```
config.py ─────────────► every other module
    (paths, labels, model registry)

setup_data.py ──► data/*.csv
                       │
preprocessing.py ◄─────┘   clean_text, strip_source_boilerplate,
    │                      text_stats, load_data
    ├──► eda.py ─────────► reports/ (analysis, leakage audit)
    ├──► train.py ───────► models/ (artifacts, metrics, model card)
    │        │
    │        ├──► evaluate.py ──────► reports/evaluation_report.md
    │        ├──► explain.py ───────► feature attributions
    │        ├──► predict.py ───────► CLI, single + batch
    │        └──► cross_dataset_eval.py
    │
    └──► app.py ──► db.py ──► reports/predictions.db
```

| I want to change… | Edit |
|---|---|
| A path or filename | `src/config.py` — nowhere else |
| Text cleaning | `src/preprocessing.py: clean_text()` → **retrain** |
| What counts as publisher boilerplate | `src/preprocessing.py: strip_source_boilerplate()` |
| Which markers the leakage audit checks | `src/eda.py: LEAKAGE_MARKERS` |
| Add/remove a model | `config.MODEL_REGISTRY` + `train.build_models()` |
| Hyperparameters | `train.build_models()`, or pass CLI flags |
| Prediction-log schema | `src/db.py: SCHEMA` (delete the old `.db` after) |
| UI layout | `app.py` — one `tab_*()` function per tab |

**Invariants to preserve:**
- `0 = fake, 1 = real`, everywhere.
- Never index `predict_proba` output by position — use
  `list(model.classes_).index(REAL)`. Positional indexing works until a model
  reports classes in a different order, then silently inverts every prediction.
- The vectorizer and the models are a matched set. Save and load them together.
- Deduplicate **before** the train/test split, never after.

---

## 9. Before you push

```bash
python -m pytest -q                        # 78 passed
python src/train.py              # artifacts still build
git status                       # no data/*.csv, no *.joblib, no predictions.db
```

`.gitignore` already covers the datasets, model artifacts and the runtime DB.
`reports/` **is** committed on purpose so teammates can read the analysis
without running the pipeline.

---

## 10. Demo script

A five-minute walkthrough that leads with the finding rather than the accuracy.
Run `python src/eda.py`, `evaluate.py`, `significance.py`, `temporal_eval.py`,
`tune.py`, `alt_models.py`, `error_taxonomy.py` and `data_quality.py` first so
every tab is populated.

1. **Start with the flaw.** Open the **Data** tab. The red banner: *"a
   single-rule classifier scores 99.5%"*. Explain that this is why the headline
   accuracy can't be taken at face value.

2. **Prove it four ways in thirty seconds.** **Evidence** tab. The KPI row is
   the whole argument: keyword rule vs best model **p = 0.57, no difference
   detected**; SVM at **50 features scores 98.98%**; the stylometry model that
   cannot see the tag scores **88.32%** and *doesn't move* when you strip the
   boilerplate; the keyword stump **collapses 45 points** when you do. Click
   through the five pills if there's time — each is a separate analysis.

3. **Show the model agreeing.** **Model** tab → strongest REAL indicators.
   `reuters` sits on top with roughly 4× the weight of anything else. The
   audit and the learned weights independently point at the same thing.

4. **Then classify something.** **Classify** tab, paste sensational text. Point
   at the evidence chart: `breaking`, `mainstream media` push FAKE, with exact
   numbers. Contrast with an LLM, which cannot show you this.

5. **Show the honest limitation.** Paste a short, neutral sentence. Watch the
   low-confidence and few-features-matched warnings appear. The system knows
   when it doesn't know.

6. **Close on the management layer.** **History** tab — every prediction
   logged, review verdicts recorded, corrections exportable as new training
   data. Not just a model, a loop.

Expected question: *"Why not just use ChatGPT?"* — see
[question.txt](question.txt), and add: an LLM would have returned a confident
label and we would never have found the Reuters leak.
