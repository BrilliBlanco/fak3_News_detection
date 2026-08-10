# Fake News Detection

TF-IDF + Naive Bayes / SVM pipeline that classifies a news article as
**FAKE** or **REAL** and returns a confidence score.

Works identically on Windows, macOS, and Linux (tested on Kali) — everything
is plain Python + scikit-learn, no OS-specific code, no manual downloads
required at runtime.

## 1. Get the dataset

Download **"Fake and Real News Dataset"** from Kaggle:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Unzip it and place both files here:

```
data/
  Fake.csv
  True.csv
```

(`data/` is git-ignored so the ~120MB dataset never gets committed —
everyone just downloads it once locally.)

## 2. Set up the environment

**Linux / Kali / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> If PowerShell blocks the activation script, run once as admin:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

Everyone should use the **same `requirements.txt`** so we all get identical
package versions — avoids "works on my machine" bugs between Kali and Windows.

## 3. Train the models

```bash
python src/train.py
```

This will:
1. Load and clean the data
2. Vectorize text with TF-IDF
3. Train a Naive Bayes model and a calibrated Linear SVM
4. Print accuracy/precision/recall/F1 for both
5. Save everything to `models/` (`tfidf_vectorizer.joblib`, `naive_bayes_model.joblib`,
   `svm_model.joblib`, `metrics.txt`)

Options: `python src/train.py --help`

## 4. Predict on new text

```bash
python src/predict.py --text "Breaking: scientists confirm the moon is made of cheese"
python src/predict.py --file some_article.txt --model svm
```

Output:
```
Prediction: FAKE
Confidence: 91.23%
```

## Project structure

```
fake-news-detection/
├── data/                  # put Fake.csv / True.csv here (git-ignored)
├── models/                # trained models saved here (git-ignored)
├── src/
│   ├── preprocessing.py   # text cleaning + data loading
│   ├── train.py           # training pipeline (TF-IDF, NB, SVM)
│   └── predict.py         # CLI to classify new text
├── requirements.txt
└── README.md
```

## Notes for the team

- Don't commit `data/` or `models/` — they're in `.gitignore` (large binary/CSV files).
- Naive Bayes gives confidence scores natively. SVM (`LinearSVC`) does not
  by default, so `train.py` wraps it in `CalibratedClassifierCV` to get
  real probabilities instead of raw decision-function scores.
- If `git` shows every line as changed on Windows, run once:
  `git config --global core.autocrlf true` (Windows) or
  `git config --global core.autocrlf input` (Linux/Kali).

## Extra: 3rd baseline (Logistic Regression)

`train.py` now also trains a Logistic Regression model alongside Naive Bayes
and SVM, and reports all three in `models/metrics.txt`. Use it for prediction
with `--model lr`:

```bash
python src/predict.py --text "..." --model lr
```

## Extra: cross-dataset generalization check

Trains happen on ISOT. To check whether the models generalize to a
*different* fake-news dataset's writing style (a real overfitting risk in
this task), download **WELFake** from Kaggle:
https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification

Then run:

```bash
python src/cross_dataset_eval.py --data-file data/WELFake_Dataset.csv
```

This evaluates your already-trained models (from `models/`) on WELFake and
prints accuracy/precision/recall for each. Compare these numbers to
`models/metrics.txt` — a big drop means the models leaned on ISOT-specific
writing style rather than general fake-vs-real signal. Great material for
the report's discussion/limitations section.

## Extra: URL and screenshot input

The demo UI (`app.py`) now has 3 tabs: paste text, paste a URL, or upload
a screenshot. URL and screenshot just extract plain text and feed it into
the same TF-IDF + model pipeline as manual text — the classifier itself
is unchanged.

**URL input** works out of the box (`requests` + `BeautifulSoup`). Some
sites (paywalled or JS-heavy) won't scrape well — that's a scraping
limitation, not a model problem.

**Screenshot input (OCR)** needs the Tesseract OCR engine installed
system-wide (separate from the `pytesseract` pip package):

- **Linux/Kali:** `sudo apt install tesseract-ocr`
- **Windows:** install from https://github.com/UB-Mannheim/tesseract/wiki,
  then add the install folder to your PATH (or set
  `pytesseract.pytesseract.tesseract_cmd` to the full `.exe` path in
  `src/extract.py` if it's not auto-detected).

## Deployment

Vercel isn't a good fit here (it's built for JS/serverless, not
Python+scikit-learn+Streamlit apps with model files). Better options:

- **Streamlit Community Cloud** (streamlit.io/cloud) — free, connect your
  GitHub repo, done. Recommended.
- **Hugging Face Spaces** — also free and Streamlit-native.

Note: OCR won't work on either without the host also having Tesseract
installed — check the platform's docs, or just leave OCR as a "run locally"
feature and demo URL/text input in the deployed version.
