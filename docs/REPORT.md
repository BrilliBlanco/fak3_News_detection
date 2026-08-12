# When 99% Accuracy Means Nothing: Diagnosing Label Leakage in a Fake-News Corpus

**A Data Analysis and Management Project**

| | |
|---|---|
| Author | *[Your Name]* |
| Student ID | *[ID]* |
| Module | *[Module code and title]* |
| Supervisor | *[Supervisor]* |
| Institution | *[Institution]* |
| Submission date | *[Date]* |
| Repository | `https://github.com/BrilliBlanco/fak3_News_detection` |
| Commit | `946e8e9` |

---

## Abstract

This project builds a fake-news text classifier on the ISOT Fake and Real News
Dataset and then subjects it to the evaluation it deserves. A conventional
TF-IDF pipeline with a calibrated linear SVM reaches 99.32% accuracy on a
held-out test split — a figure consistent with much of the published work on
this corpus. This report demonstrates that the figure is an artefact.

A single-keyword decision rule — *"if the article contains the word Reuters,
call it real"* — scores **99.40%** on the same test split, outperforming every
trained model. McNemar's test finds no statistically detectable difference
between that one-line rule and the best model (Holm-adjusted *p* = 0.57; 78
discordant articles, split 36/42). A feature ablation shows 50 vocabulary terms
score 98.98% where 5,000 score 99.32%, so the learning curve is effectively
flat: there is nothing to learn. Four further experiments corroborate the
diagnosis. The label leaks through three independent columns — article text
(a wire-service dateline present in 99.20% of real articles and 0.04% of fake
ones), topic label (all seven subject categories occur in exactly one class),
and publication date (every article before 13 January 2016 is fake).

A stylometry-only model, restricted to ten punctuation and casing counts and
therefore structurally incapable of observing the publisher fingerprint,
achieves 88.32% and is unaffected by removing that fingerprint (−0.09 pp),
while the keyword rule collapses from 99.40% to 54.21% — the majority-class
rate. That 88% is the closest estimate this corpus supports of the genuine task
difficulty.

The contribution is therefore not a classifier but a reproducible methodology
for detecting dataset shortcuts, together with a data-management framework —
21 automated quality checks, documented lineage, a versioned prediction store
and a governance register — that makes such defects visible before they are
mistaken for results.

**Keywords:** fake news detection, label leakage, shortcut learning, dataset
bias, TF-IDF, model evaluation, statistical significance testing, data quality

---

## Table of contents

1. [Introduction](#1-introduction)
2. [Background and related work](#2-background-and-related-work)
3. [Data](#3-data)
4. [Data management](#4-data-management)
5. [Methodology](#5-methodology)
6. [Results](#6-results)
7. [Discussion](#7-discussion)
8. [Limitations and threats to validity](#8-limitations-and-threats-to-validity)
9. [Conclusions and future work](#9-conclusions-and-future-work)
10. [References](#10-references)
11. [Appendices](#appendix-a-reproducing-every-figure-in-this-report)

---

## 1. Introduction

### 1.1 Problem context

Automated detection of fabricated news has attracted sustained attention since
2016, and the ISOT Fake and Real News Dataset (Ahmed et al., 2017; 2018) has
become one of its standard benchmarks. Reported accuracies on this corpus are
routinely above 99%. Taken at face value, that would suggest the problem is
essentially solved by classical text classification.

This project began as a conventional implementation of that pipeline. It became
an investigation into why the reported numbers cannot mean what they appear to
mean.

### 1.2 Aim

To build, evaluate and critically assess a fake-news text classifier, and to
determine what its accuracy actually measures.

### 1.3 Objectives

1. Implement a reproducible TF-IDF classification pipeline with multiple
   classifiers and full artefact versioning.
2. Profile the corpus for quality defects before modelling.
3. Evaluate with appropriate rigour: baselines, calibration, cross-validation
   and statistical significance testing rather than a single accuracy figure.
4. Establish whether measured performance reflects the intended task.
5. Provide a data-management layer covering provenance, quality gating,
   retention and governance.
6. Deliver an inspectable artefact in which every prediction can be traced to
   the features that produced it.

### 1.4 Contributions

- **A quantified diagnosis of label leakage** in a widely used benchmark,
  established through five mutually independent experiments (§6.2–6.7).
- **A statistically grounded negative result**: the best model is not
  distinguishable from a one-word keyword rule (§6.3).
- **Identification of a third leakage channel** — publication date — not
  discussed in the sources consulted for this project (§6.6).
- **An honest difficulty estimate** of ≈88% via a representation that cannot
  observe the leaking feature (§6.5).
- **A reusable data-quality framework** (21 checks, CI-gating) and governance
  documentation (§4).
- **A fully reproducible implementation**: fixed seeds, recorded run metadata,
  78 automated tests, and every figure regenerable by one command.

### 1.5 Report structure

§2 situates the work. §3 describes the corpus and §4 its management. §5 sets
out the methodology. §6 presents results across ten experiments. §7 interprets
them, §8 states the limits of what may be concluded, and §9 concludes.

---

## 2. Background and related work

### 2.1 Fake-news detection as text classification

Surveys of the field (Shu et al., 2017; Zhang and Ghorbani, 2020) distinguish
*content-based* approaches, which classify the article text, from *context-based*
approaches, which use propagation patterns or source credibility. This project
is content-based: it observes only the words in an article. It is important to
state at the outset that such a system performs **stylistic classification, not
verification**. It has no knowledge base, cannot check a claim against evidence,
and will misread a well-written falsehood as genuine.

### 2.2 The ISOT corpus

Ahmed et al. (2017, 2018) assembled ISOT by pairing articles from Reuters with
articles from sites flagged as unreliable by PolitiFact and Wikipedia. This
construction — one class from a single wire service, the other from a
heterogeneous set of low-credibility outlets — is the origin of the defect this
report characterises. The two halves are not two samples from one population;
they are two different corpora, and any feature separating the publishers also
separates the labels.

### 2.3 Shortcut learning and dataset bias

The phenomenon has a substantial literature under several names. Torralba and
Efros (2011) showed that classifiers can identify the *dataset* an image came
from, rather than its content. Gururangan et al. (2018) found annotation
artefacts in natural-language-inference corpora that allowed hypothesis-only
models to succeed without reading the premise. Geirhos et al. (2020) unify
these as *shortcut learning*: models exploit whatever decision rule minimises
loss on the training distribution, which need not be the rule the designer
intended. Kaufman et al. (2012) provide the canonical formulation of leakage in
data mining.

The methodological response common to that literature — and adopted here — is
to compare against deliberately impoverished baselines. If a model that should
be incapable of solving the task nevertheless solves it, the task is not what
it appears to be.

### 2.4 Evaluation methodology

Dietterich (1998) analyses statistical tests for comparing classifiers and
recommends McNemar's test (McNemar, 1947) for paired comparisons on a single
held-out set, which is the design used here. Holm (1979) provides the step-down
correction applied across the resulting family of comparisons. Efron and
Tibshirani (1993) supply the bootstrap used for interval estimation.
Niculescu-Mizil and Caruana (2005) motivate the calibration analysis, and Platt
(1999) the probability calibration applied to the SVM. Gebru et al. (2021) and
Mitchell et al. (2019) inform the documentation practices in §4.

---

## 3. Data

### 3.1 Source and provenance

| Property | Value |
|---|---|
| Corpus | ISOT Fake and Real News Dataset |
| Files | `Fake.csv` (23,481 articles), `True.csv` (21,417 articles) |
| Raw total | 44,898 articles |
| Period | March 2015 – February 2018 |
| Domain | Predominantly US and world politics |
| Distribution | Kaggle; bundled with this repository as `archive.zip` and extracted with SHA-256 verification |

### 3.2 Data dictionary

| Column | Type | Description | Notes |
|---|---|---|---|
| `title` | string | Headline | Never empty in either file |
| `text` | string | Article body | 5,795 exact duplicates corpus-wide |
| `subject` | categorical | Publisher's topic tag | 7 values, **disjoint across classes** |
| `date` | string | Publication date | Free-text; 99.985% parse successfully |
| `label` | int | 0 = fake, 1 = real | **Assigned by source file**, not present in the data |

The final column deserves emphasis. The label is not an annotation of
truthfulness; it is a record of which file an article came from. Every
conclusion in this report follows from taking that distinction seriously.

### 3.3 Preparation

Duplicates are removed **before** the train/test split. Removing them
afterwards, or not at all, would place identical articles on both sides and
inflate measured performance.

| Stage | Articles | Change |
|---|---|---|
| Raw | 44,898 | — |
| After removing empty content | 44,898 | 0 |
| After removing exact duplicates | 39,103 | −5,795 (12.9%) |
| After cleaning to non-empty text | 39,098 | −5 |
| Training split | 31,278 | 80%, stratified |
| Test split | 7,820 | 20%, stratified |

Post-deduplication class balance is 17,907 fake (45.8%) and 21,196 real (54.2%),
giving a majority-class baseline of 54.21%.

### 3.4 Data quality assessment

`src/data_quality.py` applies 21 automated checks across schema conformance,
completeness, uniqueness, validity, cross-class consistency, distributional
drift and file integrity. It emits a machine-readable verdict and, with
`--strict`, a non-zero exit code suitable for gating a pipeline.

**Overall score: 48.15 / 100 — 8 PASS, 7 WARN, 6 FAIL.**

Two failures are permanent properties of the corpus rather than defects to
repair: the disjoint subject categories, and the finding that the two source
files are not samples from a common population. Both are recorded in the issues
register (§4.5) and both are *findings*, not bugs.

### 3.5 Exploratory analysis

Median article length is 397 words (fake) against 376 (real) — a negligible
difference. Writing-style markers separate the classes far more sharply:

| Marker | Mean per fake article | Mean per real article | Ratio |
|---|---|---|---|
| Exclamation marks | 0.855 | 0.063 | **13.56×** |
| Quotation marks | 0.170 | 0.019 | **9.10×** |
| ALL-CAPS words | 8.275 | 3.781 | 2.19× |
| Words | 442.7 | 403.6 | 1.10× |

These are genuine stylistic differences and motivate the stylometry-only model
of §6.5. Whether they reflect *fabrication* or merely *the house style of
low-budget outlets versus a wire service* is precisely the question this corpus
cannot answer.

---

## 4. Data management

### 4.1 Lineage

```
archive.zip  (immutable, SHA-256 verified)
     │  setup_data.py
     ▼
data/*.csv  (raw layer, never modified in place)
     │  data_quality.py ── pass/warn/fail gate ──► reports/data_quality.json
     ▼
canonical frame  (label, dedupe, concatenate title+body)
     │  clean_text() ──► train_test_split(seed=42) ──► TfidfVectorizer.fit(TRAIN ONLY)
     ▼
models/*.joblib + run_metadata.json + metrics.json + model_card.md
     │
     ├──► reports/*.md, figures, tables
     └──► reports/predictions.db  ──► human review ──► corrections CSV ──► retraining
```

The raw layer is read-only. Every downstream artefact is regenerable from a
documented command with a fixed seed; nothing is edited by hand.

### 4.2 Storage design

Runtime predictions are persisted to SQLite (`reports/predictions.db`), chosen
because it is in the Python standard library — no server, no installation, a
single portable file. The schema is a single deliberately denormalised table.

Normalising `model` and `source` into lookup tables was considered and
rejected: cardinality is 3 and 5 respectively, the table is append-mostly and
analytical rather than transactional, and joins would be added to every query
to save a few kilobytes while introducing referential-integrity maintenance.
The denormalisation is a design decision, documented as such, not an oversight.
Full DDL and per-column rationale are in `docs/DATA_MANAGEMENT.md`.

Privacy is handled by storing only a 500-character preview plus a SHA-256 hash
of each input, never the full third-party document, and no user identifiers of
any kind.

### 4.3 Versioning and reproducibility

| Dimension | Mechanism |
|---|---|
| Data version | SHA-256 per source file |
| Code version | Git commit |
| Split | `random_state` and `test_size` recorded in `run_metadata.json` |
| Parameters | Complete set in `run_metadata.json` and `model_card.md` |
| Environment | Python 3.12, scikit-learn 1.6.1, platform string recorded |

Every downstream analysis reconstructs the identical test split by reading
`run_metadata.json`, so all reported figures refer to the same 7,820 articles.
This was verified: independently rebuilding the split reproduced all three
model accuracies to four decimal places.

### 4.4 Feedback loop

Predictions made through the application are logged; a reviewer marks each
correct or incorrect; corrected rows export as labelled training data. This
closes the loop between deployment and training, and — more importantly for
evaluation — measures accuracy on the distribution the system actually
encounters rather than on the benchmark.

### 4.5 Known issues register

| ID | Issue | Severity | Handling |
|---|---|---|---|
| DQ-1 | Wire-service tag in 99.20% of real, 0.04% of fake | Critical | Removal function; leakage baseline reported alongside every metric |
| DQ-2 | 7/7 subject categories single-class | Critical | Excluded from features; used only for error slicing |
| DQ-3 | Disjoint date ranges | High | Quantified (§6.6); excluded from features |
| DQ-4 | 5,795 exact duplicates (12.9%) | High | Removed before splitting |
| DQ-5 | Classes drawn from different sources | Critical | Cross-dataset evaluation identified as the only valid generalisation test |
| DQ-6 | Inconsistent date formats | Low | Coerced; 0.015% unparseable, excluded from time analysis only |
| DQ-7 | Domain and period bound | Medium | Stated as a scope limit in the model card and interface |

---

## 5. Methodology

### 5.1 Preprocessing

Lowercasing; removal of URLs, HTML, punctuation and digits; English stopword
removal; tokens shorter than three characters discarded. A separate
`strip_source_boilerplate()` function removes publisher fingerprints —
wire-service datelines, agency tags, editorial sign-offs and blog-footer
markers — and is applied only in the ablation conditions where its effect is
the object of study.

### 5.2 Feature extraction

TF-IDF over unigrams and bigrams, `max_features=5,000`, `min_df=2`. The
vectorizer is fitted on the **training split only** in every experiment. This
is stated explicitly because fitting a vectorizer before splitting is the most
common silent leak in text-classification pipelines, and because the temporal
experiment (§6.6) would otherwise carry future vocabulary backwards in time.

### 5.3 Models

| Model | Rationale |
|---|---|
| Multinomial Naive Bayes | Standard fast text baseline; native probabilities |
| Linear SVM (calibrated) | Strong on sparse high-dimensional text; `LinearSVC` wrapped in `CalibratedClassifierCV` (Platt, 1999) to obtain probabilities rather than raw margins |
| Logistic Regression | Directly interpretable coefficients in log-odds space |

All three are linear over identical features — a limitation addressed
explicitly in §6.5 by introducing structurally different representations.

### 5.4 Protocol

80/20 stratified split, `random_state=42`, duplicates removed beforehand.
Hyperparameter selection uses 3-fold cross-validation on the **training split
only**; the test split is scored once, at the end, and never consulted during
model selection.

### 5.5 Evaluation

Accuracy, precision, recall and F1 (positive class = real), ROC AUC, and Brier
score for probability quality. Two baselines accompany every headline figure:
majority class, and the single-keyword rule. Reporting a model's accuracy
without the accuracy of the dumbest thing that also works is the practice this
project exists to argue against.

### 5.6 Statistical testing

McNemar's test on the paired correctness vectors, using the exact binomial form
when the discordant count is ≤ 25 and the continuity-corrected chi-square form
otherwise; Holm–Bonferroni correction across the family of pairwise
comparisons; and a **paired** bootstrap (2,000 resamples, seeded) in which
every system is scored on the same resampled items, so that intervals on
*differences* account for the correlation between systems.

### 5.7 Implementation and verification

Approximately 6,800 lines of Python across 16 modules, with 78 automated tests.
The statistical core is tested against hand-worked values rather than against
current behaviour: McNemar discordant counts, Holm monotonicity and
order-preservation, and the paired-bootstrap invariant that two anti-correlated
models must have accuracies summing to exactly 1 in every replicate.

All headline figures in §6 were additionally re-derived by an independent
script that rebuilt the split from first principles. Every value matched
exactly.

---

## 6. Results

### 6.1 Headline classification performance

Test split, 7,820 articles.

| Model | Accuracy | Precision | Recall | F1 | ROC AUC | Brier |
|---|---|---|---|---|---|---|
| Naive Bayes | 0.9422 | 0.9463 | 0.9472 | 0.9467 | 0.9860 | 0.0432 |
| **SVM (calibrated)** | **0.9932** | 0.9936 | 0.9939 | **0.9937** | 0.9995 | **0.0057** |
| Logistic Regression | 0.9867 | 0.9832 | 0.9925 | 0.9878 | 0.9989 | 0.0140 |

Three-fold cross-validated F1 on the training split confirms stability: NB
0.9483 ± 0.0019, SVM 0.9923 ± 0.0002, LR 0.9862 ± 0.0005. The SVM is also
well calibrated (Brier 0.0057), so its confidence estimates are meaningful in
aggregate.

Reported in isolation, this is an excellent result. §6.2 explains why it is not.

### 6.2 Baselines — the central finding

| System | Accuracy |
|---|---|
| Majority class | 0.5421 |
| **One rule: contains "reuters" ⇒ real** | **0.9940** |
| Best trained model (SVM) | 0.9932 |

A single keyword outperforms a 5,000-feature machine-learning pipeline. Across
the full corpus the `(Reuters)` tag appears in **99.20%** of real articles and
**0.04%** of fake ones; used alone it classifies **99.55%** of all articles
correctly.

### 6.3 Statistical significance

McNemar's test with Holm–Bonferroni correction, and paired bootstrap intervals
(2,000 resamples).

| System | Accuracy | 95% CI | Errors |
|---|---|---|---|
| One-rule keyword | 0.9940 | [0.9921, 0.9956] | 47 |
| SVM | 0.9932 | [0.9913, 0.9948] | 53 |
| Logistic Regression | 0.9867 | [0.9839, 0.9891] | 104 |
| Naive Bayes | 0.9422 | [0.9364, 0.9467] | 452 |

**Keyword rule vs SVM:** 78 discordant articles, split *b* = 42 / *c* = 36;
χ² = 0.32; **Holm-adjusted *p* = 0.5713**; difference in accuracy 0.0008,
95% CI **[−0.0014, +0.0028]**.

The interval spans zero. There is no detectable difference between the best
model in this project and a one-line keyword rule. By contrast the rule *is*
distinguishable from Logistic Regression (*p* = 1.6 × 10⁻⁶) and Naive Bayes,
so the test is not simply under-powered.

### 6.4 Feature ablation

Accuracy of the calibrated SVM against vocabulary size, with and without
publisher boilerplate.

| `max_features` | Boilerplate intact | Boilerplate removed |
|---|---|---|
| 50 | **0.9898** | 0.9105 |
| 100 | 0.9898 | 0.9322 |
| 200 | 0.9905 | 0.9527 |
| 500 | 0.9900 | 0.9692 |
| 1,000 | 0.9907 | 0.9766 |
| 2,000 | 0.9919 | 0.9821 |
| 5,000 | 0.9932 | 0.9847 |
| 20,000 | 0.9930 | 0.9867 |

This is the clearest single piece of evidence in the project. With the
fingerprint present the curve is **flat from 50 features**: 400× more
vocabulary buys 0.34 percentage points. A flat learning curve means the
information required is available immediately and nothing further is learned.
Remove the fingerprint and the same curve climbs 7.6 points across the same
range — the ordinary behaviour of a model that must actually generalise.

Corroborating this: when the vocabulary is squeezed to 50 terms, **`reuters` is
the 6th term retained**.

### 6.5 Alternative representations

Because all three primary models are linear over identical features, they share
any defect in that feature space. Four structurally different models test
whether the result survives a change of representation.

| Model | What it reads | Intact | Stripped | Δ |
|---|---|---|---|---|
| Word TF-IDF + SVM (reference) | sparse words | 0.9932 | 0.9847 | −0.0086 |
| Char 3–5-grams + SVM | sparse characters | 0.9996 | 0.9983 | −0.0013 |
| TF-IDF → SVD → gradient boosting | dense topics, non-linear | 0.9683 | 0.9597 | −0.0086 |
| **Stylometry only + Random Forest** | **10 numeric counts** | **0.8832** | **0.8841** | **+0.0009** |
| Control: keyword stump | 1 binary feature | 0.9940 | 0.5421 | **−0.4519** |

Three findings follow.

First, the character n-gram model's 0.9996 is not a success but the strongest
leakage signature in the table: a 5-character window captures the literal
string `(Reuters)` even more readily than a word tokeniser.

Second, the control behaves exactly as a leakage-dependent model should — it
collapses to the majority-class rate once the fingerprint is removed.

Third, and most importantly, the **stylometry-only model is unmoved** (+0.09 pp).
Reading only punctuation and casing counts, it cannot observe the fingerprint,
so removing it changes nothing. Its **88.32%** is therefore the estimate least
contaminated by the defect — and the honest headline for this project.

### 6.6 Temporal validation and the third leakage channel

A random split draws training and test articles from the same weeks. A
deployed detector encounters articles published after everything it has seen.
Retraining on articles before 1 June 2017 and testing on those after gives:

| Split | Condition | Train | Test | Accuracy | Balanced acc. |
|---|---|---|---|---|---|
| Random | intact | 31,277 | 7,820 | 0.9926 | 0.9925 |
| Temporal | intact | 22,337 | 16,760 | 0.9799 | 0.9842 |
| Random | stripped | 31,275 | 7,819 | 0.9835 | 0.9833 |
| Temporal | stripped | 22,335 | 16,759 | 0.9436 | 0.9565 |

(The random/intact figure of 0.9926 differs from the 0.9932 in §6.1 by one
training row: this experiment additionally discards the six articles whose
`date` field cannot be parsed, so that the random control and the temporal
split are computed over exactly the same population.)

Temporal validation alone barely dents performance — because the fingerprint
crosses the cutoff along with the articles. Only removing it *and* splitting
temporally produces a meaningful drop. **High accuracy on a future test period
is not evidence of generalisation when the shortcut is time-invariant**, which
is itself a methodological finding worth stating.

The date analysis also surfaced a channel not previously identified in this
project. Fake articles span 2015-03-31 to 2018-02-19; real articles only
2016-01-13 to 2017-12-31. **Every article published before 13 January 2016 is
fake.** 2,028 articles (5.19% of the corpus) fall outside the overlap and all
are fake; 11 of the 35 covered months contain exactly one class. Publication
date is thus a third independent leak, alongside text and subject.

### 6.7 Error analysis

| Model | Errors | Rate | False REAL | False FAKE | Shared by all three | Unique |
|---|---|---|---|---|---|---|
| Naive Bayes | 452 | 5.78% | 228 | 224 | 39 | 365 (80.8%) |
| SVM | 53 | 0.68% | 27 | 26 | 39 | 4 (7.5%) |
| Logistic Regression | 104 | 1.33% | 72 | 32 | 39 | 13 (12.5%) |

*False REAL* — a fabricated article accepted as genuine — is the costly
direction for a moderation tool, and the models split their errors roughly
evenly between the two directions.

The 39 articles misclassified by all three models are **dataset-driven**: they
are not attributable to any modelling choice, since three different learning
algorithms fail identically on them. For the SVM these constitute 73.6% of all
its errors, meaning that the strongest model has almost no idiosyncratic
failures left — its remaining mistakes are properties of the corpus.

### 6.8 Hyperparameter optimisation

Grid search on the training split only, scored once on the test split:

| Model | Default F1 | Tuned F1 | Δ | 95% CI |
|---|---|---|---|---|
| Naive Bayes | 0.9467 | 0.9613 | +0.0146 | [+0.0114, +0.0179] |
| SVM | 0.9943 | 0.9960 | +0.0016 | [+0.0006, +0.0027] |
| Logistic Regression | 0.9878 | 0.9939 | +0.0061 | [+0.0044, +0.0079] |

Improvements are statistically real but practically negligible for the SVM —
0.16 pp on a task where a keyword scores 99.4%. Sublinear term-frequency
scaling was the single most influential parameter. The honest conclusion is
that hyperparameter tuning is not where the value lies in this problem.

### 6.9 Explainability

All three models are linear over TF-IDF, so a prediction decomposes exactly:
*score = bias + Σ tfidf(term) × weight(term)*. Each term's contribution is
arithmetic, not an approximation such as LIME or SHAP. (One qualification: for
the calibrated SVM the weights are averaged over three calibration folds, so
term *ranking* is exact while magnitudes are close but not identical.)

Inspecting the global weights corroborates the leakage diagnosis independently
of every experiment above. `reuters` is the highest-weighted "real" indicator
in every model: **20.25** in Logistic Regression (against 14.98 for the next
term, `said`, and below 9 for everything else) and **13.94** in the SVM, which
is 2.74× the next feature and 2.95× the strongest term not containing
"reuters". More telling than any single ratio is the composition of the top of
the list: of the SVM's eight strongest real-indicating features, **four are
Reuters-derived** — `reuters`, `washington reuters`, `reuters president` and
`york reuters` — alongside `said`, `president donald`, `nov` and `factbox`
(the last being a Reuters article-format tag). The model states plainly what it
has learned.

### 6.10 Application

A five-tab Streamlit interface provides classification (text, URL or
screenshot via OCR) with per-prediction evidence, batch CSV scoring, the
exploratory findings, an evidence tab presenting the five analyses above, model
metrics, and the prediction history with its review loop. Confidence below 70%
and inputs matching fewer than five vocabulary features trigger explicit
warnings, so the system communicates its own uncertainty rather than projecting
false confidence.

---

## 7. Discussion

### 7.1 What the models actually learned

Five independent lines of evidence converge:

1. A one-word rule matches the best model, with no statistically detectable
   difference (*p* = 0.57).
2. 50 features perform as well as 5,000 — a flat curve, i.e. nothing learned.
3. `reuters` is the 6th term retained under aggressive feature selection, is
   the top-weighted "real" indicator in every model, and Reuters-derived terms
   occupy four of the SVM's eight strongest.
4. A model blind to the fingerprint scores 88.3% and is unaffected by its
   removal, while a model dependent on it collapses by 45 points.
5. The label leaks through three separate columns.

The models learned to identify the publisher. Given how ISOT was assembled —
one class from a single wire service — this is the *optimal* strategy under the
training objective. The models are not defective. The task, as posed by the
corpus, is not the task the field believes it is measuring.

### 7.2 Implications for reported results on this corpus

Any accuracy quoted from a random split of ISOT is an upper bound on a problem
substantially easier than fake-news detection. This does not indict the
published work directly — this project made the same mistake before testing
for it — but it does suggest that results on this benchmark should be
accompanied, at minimum, by a leakage baseline. The cost of computing one is a
single line of code, and it is the difference between reporting a measurement
and reporting an artefact.

### 7.3 The methodological lesson

The generalisable contribution is a procedure, not a number:

1. Profile the data before modelling.
2. Report the simplest possible baseline beside every headline metric.
3. Test whether differences between models are statistically real.
4. Ablate the feature space — a flat curve is diagnostic.
5. Include at least one model structurally incapable of exploiting the
   suspected shortcut.
6. Inspect what the model weights actually say.

Steps 2 and 5 are cheap and would have exposed this defect immediately.

### 7.4 Classical models versus a large language model

A reasonable challenge is why not simply prompt an LLM. Four arguments,
strengthened by the findings above:

- **Inspectability.** Every decision here decomposes into weighted terms. That
  property is what made the leak discoverable at all.
- **Reproducibility.** Identical input yields identical output indefinitely,
  from a versioned artefact.
- **Cost.** Inference is free, offline and millisecond-scale; the classifier
  is ~1 MB.
- **Discovery.** This is the decisive point. An LLM would have returned a
  confident label for each article. It would not have revealed that the
  benchmark is broken. The finding came from the transparency of the method,
  not despite its simplicity.

The honest concession is that an LLM would very likely be *more accurate* on
genuinely unseen news, because it is not restricted to this corpus's
vocabulary. Accuracy, however, was not the deliverable.

---

## 8. Limitations and threats to validity

1. **No cross-dataset evaluation.** The single most significant limitation.
   Evaluating on an independently sourced corpus (e.g. WELFake; Verma et al.,
   2021) is the only test that cannot be confounded by publisher leakage. The
   code is implemented and tested (`src/cross_dataset_eval.py`) but was not run,
   as the second corpus could not be obtained in the working environment. Until
   it is, the 88.3% stylometry figure is the best available estimate of true
   difficulty, and it remains an estimate from a single corpus.

2. **The stylometry estimate is itself corpus-bound.** It shows what is
   achievable without the fingerprint *within ISOT*. It does not establish that
   88% transfers elsewhere.

3. **Style is not veracity.** The system detects sensationalist register. A
   carefully written falsehood reads as real; a poorly written truth reads as
   fake. It must never be deployed as an arbiter of truth.

4. **Domain and period.** US and world politics, 2015–2018. Performance on
   other domains, languages or later events is unmeasured.

5. **Boilerplate removal is imperfect.** `strip_source_boilerplate()` is
   regex-based and may leave residual publisher signal, which would make the
   stripped-condition results optimistic.

6. **Binary framing.** Real classification is a spectrum from accurate
   reporting through misleading framing to fabrication; this reduces it to two
   classes.

7. **Single split.** Although cross-validation and bootstrap intervals are
   reported, all test-set figures derive from one seed. Repeated splits would
   tighten the estimates.

---

## 9. Conclusions and future work

### 9.1 Conclusions

A conventional pipeline achieves 99.32% accuracy on ISOT. That figure does not
measure fake-news detection. A one-word keyword rule achieves 99.40% on the
same data, and no statistically detectable difference separates them
(*p* = 0.57). Fifty features suffice to reach 98.98%. A model that cannot
observe the publisher fingerprint achieves 88.32% and is unaffected by its
removal, while one dependent on it collapses to chance. The label leaks through
article text, topic label and publication date simultaneously.

The classifier is therefore not the contribution. The contribution is a
reproducible methodology for detecting this class of defect, a quantified
diagnosis of it in a widely used benchmark, and a data-management framework
that surfaces such problems before they are mistaken for results.

The objectives in §1.3 are met, with the qualification recorded in §8.1: the
generalisation question is characterised and the instrument built, but the
decisive measurement awaits a second corpus.

### 9.2 Future work

**Immediate.** Run the cross-dataset evaluation against WELFake. This is one
command and would convert the project's principal limitation into its
strongest result.

**Short term.** Retrain on a corpus assembled so that both classes are drawn
from the same publishers, which would eliminate the confound at source rather
than patching it. Report leakage baselines as standard practice.

**Longer term.** Extend to multi-class credibility rather than binary
labelling; combine content features with propagation or source signals
(Shu et al., 2017); and — the natural continuation of §7.4 — evaluate whether
an LLM's advantage survives on a corpus where the shortcut has been removed,
which would isolate genuine semantic capability from shortcut exploitation.

---

## 10. References

Ahmed, H., Traore, I. and Saad, S. (2017) 'Detection of online fake news using
N-gram analysis and machine learning techniques', in *Intelligent, Secure, and
Dependable Systems in Distributed and Cloud Environments (ISDDC)*. Springer,
pp. 127–138.

Ahmed, H., Traore, I. and Saad, S. (2018) 'Detecting opinion spams and fake
news using text classification', *Security and Privacy*, 1(1), e9.

Dietterich, T.G. (1998) 'Approximate statistical tests for comparing supervised
classification learning algorithms', *Neural Computation*, 10(7),
pp. 1895–1923.

Efron, B. and Tibshirani, R.J. (1993) *An Introduction to the Bootstrap*.
New York: Chapman & Hall.

Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J.W., Wallach, H.,
Daumé III, H. and Crawford, K. (2021) 'Datasheets for datasets',
*Communications of the ACM*, 64(12), pp. 86–92.

Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M.
and Wichmann, F.A. (2020) 'Shortcut learning in deep neural networks',
*Nature Machine Intelligence*, 2(11), pp. 665–673.

Gururangan, S., Swayamdipta, S., Levy, O., Schwartz, R., Bowman, S.R. and
Smith, N.A. (2018) 'Annotation artifacts in natural language inference data',
in *Proceedings of NAACL-HLT*, pp. 107–112.

Holm, S. (1979) 'A simple sequentially rejective multiple test procedure',
*Scandinavian Journal of Statistics*, 6(2), pp. 65–70.

Kaufman, S., Rosset, S., Perlich, C. and Stitelman, O. (2012) 'Leakage in data
mining: formulation, detection, and avoidance', *ACM Transactions on Knowledge
Discovery from Data*, 6(4), pp. 1–21.

McNemar, Q. (1947) 'Note on the sampling error of the difference between
correlated proportions or percentages', *Psychometrika*, 12(2), pp. 153–157.

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B.,
Spitzer, E., Raji, I.D. and Gebru, T. (2019) 'Model cards for model reporting',
in *Proceedings of the Conference on Fairness, Accountability, and
Transparency*, pp. 220–229.

Niculescu-Mizil, A. and Caruana, R. (2005) 'Predicting good probabilities with
supervised learning', in *Proceedings of the 22nd International Conference on
Machine Learning*, pp. 625–632.

Pedregosa, F. et al. (2011) 'Scikit-learn: machine learning in Python',
*Journal of Machine Learning Research*, 12, pp. 2825–2830.

Platt, J. (1999) 'Probabilistic outputs for support vector machines and
comparisons to regularized likelihood methods', in *Advances in Large Margin
Classifiers*. MIT Press, pp. 61–74.

Shu, K., Sliva, A., Wang, S., Tang, J. and Liu, H. (2017) 'Fake news detection
on social media: a data mining perspective', *ACM SIGKDD Explorations
Newsletter*, 19(1), pp. 22–36.

Torralba, A. and Efros, A.A. (2011) 'Unbiased look at dataset bias', in
*Proceedings of the IEEE Conference on Computer Vision and Pattern
Recognition*, pp. 1521–1528.

Verma, P.K., Agrawal, P., Amorim, I. and Prodan, R. (2021) 'WELFake: word
embedding over linguistic features for fake news detection', *IEEE Transactions
on Computational Social Systems*, 8(4), pp. 881–893.

Zhang, X. and Ghorbani, A.A. (2020) 'An overview of online fake news:
characterization, detection, and discussion', *Information Processing &
Management*, 57(2), 102025.

---

## Appendix A: Reproducing every figure in this report

```bash
pip install -r requirements.txt
python src/setup_data.py         # extract corpus, verify checksums
python src/data_quality.py       # §3.4  - 21 quality checks
python src/eda.py                # §3.5, §6.2 - profiling + leakage audit
python src/train.py --cv 3       # §6.1  - three models, metrics, model card
python src/evaluate.py --cv 3 --learning-curve   # §6.1 - curves, calibration
python src/significance.py       # §6.3  - McNemar, Holm, bootstrap
python src/tune.py               # §6.4, §6.8 - ablation and grid search
python src/alt_models.py         # §6.5  - alternative representations
python src/temporal_eval.py      # §6.6  - temporal split, date leakage
python src/error_taxonomy.py     # §6.7  - error categorisation
python src/explain.py --global --model lr        # §6.9 - global weights
python -m pytest -q              # 78 tests
python -m streamlit run app.py   # §6.10 - application
```

Not run in this project, pending a second corpus (§8.1):

```bash
python src/cross_dataset_eval.py --data-file data/WELFake_Dataset.csv
```

## Appendix B: Repository structure

Sixteen modules under `src/`, a Streamlit application, 78 tests, and generated
analysis under `reports/`. Full listing in `README.md`; data-management detail
in `docs/DATA_MANAGEMENT.md`; operational guidance in `PROJECT_GUIDE.md`.

## Appendix C: Environment

Python 3.12.x, scikit-learn 1.6.1, pandas 2.3.1, numpy 2.1.3, matplotlib 3.10,
scipy, Streamlit 1.61.1. Windows 11; the codebase is OS-independent and has
been exercised on Linux. All randomness seeded at 42.
