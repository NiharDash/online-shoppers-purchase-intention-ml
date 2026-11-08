# Online Shoppers Purchasing Intention — Classification & Streamlit App

M.Tech (AIML) · Machine Learning · Assignment 2

---

## a. Problem Statement

Online retailers see far more browsing sessions than purchases. Being able to tell,
**from behavioural signals captured during a session, whether that session will end in a
purchase**, lets a site intervene while the visitor is still present — by surfacing a
discount, offering live chat, or prioritising a retargeting budget.

This project frames that as a **supervised binary classification** problem:

> Given 17 features describing a single e-commerce session (pages visited, time spent,
> bounce/exit rates, the Google Analytics "page value", proximity to a special day,
> plus technical and calendar attributes), predict whether the session ended in a
> transaction (`Revenue = True`) or not (`Revenue = False`).

Five classifiers are trained and compared on identical data, and an interactive
Streamlit application loads the saved models and evaluates them on any test CSV the
user uploads.

**Why the metric choice matters here.** Only **15.5%** of sessions convert. A model that
blindly predicts "no purchase" for every session already scores **84.5% accuracy** while
being commercially worthless. Accuracy is therefore reported but *not* used to rank the
models — **MCC** and **AUC** are, because both account for all four confusion-matrix
cells and are robust to the class imbalance.

---

## b. Dataset Description

| Property | Value |
|---|---|
| **Name** | Online Shoppers Purchasing Intention Dataset |
| **Source** | UCI Machine Learning Repository ([ID 468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)) |
| **Instances** | **12,330** sessions (requirement: ≥ 500 ✔) |
| **Features** | **17** input features (requirement: ≥ 12 ✔) |
| **Target** | `Revenue` — boolean, did the session end in a transaction |
| **Class balance** | 10,422 negative (84.5%) · 1,908 positive (**15.5%**) |
| **Missing values** | None |
| **Task type** | Binary classification |

Each row is one user session. The data was collected over a one-year period and
deliberately sampled so that no two sessions belong to the same user, campaign or
period — so sessions can reasonably be treated as independent.

### Feature dictionary

| # | Feature | Type | Description |
|---|---|---|---|
| 1 | `Administrative` | int | Number of account/admin pages visited |
| 2 | `Administrative_Duration` | float | Seconds spent on those pages |
| 3 | `Informational` | int | Number of informational pages visited |
| 4 | `Informational_Duration` | float | Seconds spent on those pages |
| 5 | `ProductRelated` | int | Number of product pages visited |
| 6 | `ProductRelated_Duration` | float | Seconds spent on product pages |
| 7 | `BounceRates` | float | Avg. bounce rate of pages visited |
| 8 | `ExitRates` | float | Avg. exit rate of pages visited |
| 9 | `PageValues` | float | Avg. Google Analytics page value |
| 10 | `SpecialDay` | float | Closeness to a special day (0–1) |
| 11 | `Month` | categorical | Month of the session (10 levels) |
| 12 | `OperatingSystems` | categorical id | Visitor's OS |
| 13 | `Browser` | categorical id | Visitor's browser |
| 14 | `Region` | categorical id | Geographic region |
| 15 | `TrafficType` | categorical id | Traffic source |
| 16 | `VisitorType` | categorical | Returning / New / Other |
| 17 | `Weekend` | boolean | Session occurred at a weekend |
| — | **`Revenue`** | **boolean** | **Target** |

### Preprocessing

- **Stratified 80 / 20 split** (`random_state=42`) → 9,864 train / 2,466 test. Stratifying
  keeps the 15.5% positive rate identical in both halves.
- **Continuous features** → `StandardScaler`. Necessary for the distance- and
  margin-based models (KNN, SVM) and for logistic regression's convergence.
- **Categorical features** → `OneHotEncoder(handle_unknown="ignore")`.
  `OperatingSystems`, `Browser`, `Region` and `TrafficType` are stored as integers but
  are *unordered identifiers* — leaving them numeric would falsely imply
  `Browser 4 > Browser 2`, so they are one-hot encoded too. `handle_unknown="ignore"`
  means a category level absent from training will not crash the deployed app.
- Every model is wrapped in a **`Pipeline`** so preprocessing is fitted on the training
  fold only (**no data leakage**) and travels with the serialised model. The app never
  re-implements preprocessing.

---

## c. GitHub Repository Link

**https://github.com/REPLACE_ME/online-shoppers-purchase-intention-ml**

**Live Streamlit App:** _REPLACE_ME_ (see Deployment section)

```
online-shoppers-ml-app/
├── app.py                 # Streamlit front-end (inference + evaluation only)
├── train_models.py        # Trains all 5 models, writes model/ and test_data.csv
├── requirements.txt       # Pinned dependencies
├── README.md
├── test_data.csv          # Held-out 2,466-row test split (upload this to the app)
├── data/
│   └── online_shoppers_intention.csv   # Full raw UCI dataset
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── support_vector_machine.joblib
    └── metrics.json       # Metrics recorded at training time
```

### Reproducing

```bash
pip install -r requirements.txt
python train_models.py      # retrains, overwrites model/ and test_data.csv
streamlit run app.py
```

---

## d. Models Used

All five classifiers are trained on the **same** stratified training split and evaluated
on the **same** held-out 2,466-row test set.

| Model | Key hyperparameters | Rationale |
|---|---|---|
| Logistic Regression | `max_iter=2000` | Linear baseline; well-calibrated probabilities |
| Decision Tree | `max_depth=8` | Captures non-linear interactions; depth capped to limit overfitting |
| KNN | `n_neighbors=15` | Instance-based; k raised above default to smooth a noisy boundary |
| Naive Bayes | `GaussianNB()` | Continuous features → Gaussian variant |
| Support Vector Machine | `kernel="rbf"`, `probability=True` | Non-linear margin; `probability` needed for AUC |

All are left at otherwise-default settings (in particular **no** `class_weight="balanced"`)
so the comparison between them is like-for-like and the effect of the imbalance stays
visible in the metrics.

### Evaluation Metrics — Comparison Table

Measured on the held-out test set (2,466 sessions), decision threshold 0.50.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8808 | **0.8830** | **0.7366** | 0.3586 | 0.4824 | 0.4592 |
| Decision Tree | **0.8917** | 0.8798 | 0.6936 | 0.5393 | **0.6068** | **0.5509** |
| KNN | 0.8755 | 0.8238 | 0.7246 | 0.3168 | 0.4408 | 0.4243 |
| Naive Bayes | 0.2733 | 0.7333 | 0.1727 | **0.9738** | 0.2934 | 0.1292 |
| Support Vector Machine | 0.8881 | 0.8829 | 0.7246 | 0.4476 | 0.5534 | 0.5122 |

> **Baseline for context:** always predicting "no purchase" gives **0.8451 accuracy**,
> with AUC 0.50 and MCC 0.00. Three of the five models beat that accuracy by less than
> five points — which is exactly why accuracy is the wrong column to read.

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | **Best ranker, poor decider.** Its AUC (0.8830) is the highest of the five, meaning it orders sessions by purchase likelihood better than any other model. Yet at the default 0.50 cut-off it recovers only **35.9% of actual purchases** — it misses roughly two in three. High AUC with low recall is the signature of a *well-ranked but badly-thresholded* classifier: the fix is to move the threshold, not to change the model. Its precision (0.7366) is the best of the five, so the purchases it does flag are usually genuine. |
| **Decision Tree** | **Best overall on this dataset.** Top MCC (0.5509), top F1 (0.6068) and top accuracy (0.8917). It nearly doubles logistic regression's recall (0.5393 vs 0.3586) for only a modest precision cost. The reason is structural: `PageValues` alone carries **65.6% of the tree's total feature importance**, and it acts through a sharp threshold effect — 77.9% of sessions have `PageValues` exactly zero, and almost all conversions sit in the non-zero tail. An axis-aligned split models that step directly; a linear decision boundary can only approximate it. Capping depth at 8 was necessary — an unconstrained tree memorises the training set. |
| **KNN** | **Weakest of the four working models.** AUC 0.8238 and recall 0.3168 are both the lowest among the non-degenerate models. Two causes: (i) one-hot encoding expands the feature space from 17 columns to **74 dimensions**, where Euclidean distances concentrate and neighbourhoods stop being locally meaningful — the curse of dimensionality; (ii) with only 15.5% positives, the 15 nearest neighbours of a true purchase are usually dominated by non-purchases, so the majority vote suppresses the minority class. It is also the most expensive artefact at **5.9 MB**, since a KNN model must store the entire training set. |
| **Naive Bayes** | **Fails badly, and informatively.** Accuracy collapses to **0.2733 — far worse than the 0.8451 always-negative baseline** — because it flags almost everything as a purchase (recall 0.9738, precision 0.1727). Both of its core assumptions are violated, and measurably so: features are *not* conditionally independent (`BounceRates` ↔ `ExitRates` correlate at **r = 0.91**, `ProductRelated` ↔ `ProductRelated_Duration` at **r = 0.86**), and the continuous columns are nothing like Gaussian — `PageValues` has **skew 6.4 with 77.9% of values exactly zero**, and `ProductRelated_Duration` has skew 7.3. Crucially its **AUC is still 0.7333**, well above random: the model retains real ranking signal but its probability estimates are so badly miscalibrated that a 0.50 cut-off sits in the wrong place entirely. This is the cleanest illustration in the set of why accuracy and AUC must be read together. |
| **Support Vector Machine** | **Strong and well-balanced — a close second.** MCC 0.5122, and its AUC (0.8829) is statistically indistinguishable from logistic regression's (0.8830). The RBF kernel buys genuine non-linearity, converting that equal ranking ability into better decisions at the default threshold (recall 0.4476 vs 0.3586). The costs are practical rather than statistical: it was by far the slowest to train (the `probability=True` flag fits Platt scaling over an internal 5-fold CV), and it produces a 1.7 MB artefact. |
| **Overall winner for this dataset** | **Decision Tree.** It leads on MCC (0.5509), F1 (0.6068) and accuracy (0.8917) simultaneously, and it is the cheapest to train and store (31 KB). One honest caveat: logistic regression and SVM edge it on AUC, so they *rank* sessions marginally better. If the threshold were tuned per model rather than fixed at 0.50, the gap would narrow and could plausibly reverse — the app's threshold slider is included precisely so this can be explored interactively. On the stated evaluation protocol, however, the Decision Tree wins. |

**A cross-cutting finding.** Every model except Naive Bayes clusters within four
accuracy points of each other and of the trivial baseline, while their MCC scores range
from 0.42 to 0.55 and their recall from 0.32 to 0.54. On an imbalanced target, accuracy
compresses real differences and MCC exposes them. Reporting accuracy alone here would
have made four genuinely different models look interchangeable — and would have
disguised the fact that none of them catches more than 54% of actual purchases.

---

## Streamlit Application

### Features

| Requirement | Where it is implemented |
|---|---|
| **Dataset upload (CSV)** | Sidebar uploader; falls back to the bundled `test_data.csv` |
| **Model selection dropdown** | Sidebar `selectbox` across all five classifiers |
| **Display of evaluation metrics** | Six metric cards (Accuracy, AUC, Precision, Recall, F1, MCC) |
| **Confusion matrix / classification report** | Both — annotated heatmap *and* a full per-class report |
| **All models on the test data** | "All five compared" tab: sortable table, metric bar chart, overlaid ROC curves |

Additional features beyond the brief: an **adjustable decision threshold** (so the
precision/recall trade-off discussed above can be explored live), **overlaid ROC curves**
for all five models, a **data-preview tab** showing class balance, and a **CSV download**
of the comparison results.

### Important: no retraining happens in the app

The app loads the pre-fitted `.joblib` pipelines and calls `predict_proba` on them.
Uploading a CSV produces a fresh *evaluation*; it never refits a model. The metrics the
app reports at threshold 0.50 on `test_data.csv` reproduce `model/metrics.json` exactly.

### Uploading your own CSV

The file must contain **all 17 input features plus the `Revenue` target column**, in raw
(un-preprocessed) form — scaling and encoding are handled inside each pipeline.
`test_data.csv` in this repo is exactly that format and is the recommended upload.

### Deployment

Deployed on **Streamlit Community Cloud**:

1. Push this repository to GitHub (public).
2. Sign in at [streamlit.io/cloud](https://streamlit.io/cloud) with GitHub.
3. **New App** → select this repo → branch `main` → main file `app.py` → **Deploy**.

`scikit-learn` is pinned to the exact training version in `requirements.txt`; unpickling
a model across a different minor version raises `InconsistentVersionWarning` and can
misbehave silently.

---

## Tech Stack

Python · scikit-learn · pandas · NumPy · Streamlit · Matplotlib · Seaborn · joblib

## Dataset Citation

Sakar, C.O. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset.*
UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q
