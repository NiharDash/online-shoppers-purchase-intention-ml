"""
Online Shoppers Purchasing Intention - training script
======================================================

Trains five classifiers on the UCI "Online Shoppers Purchasing Intention"
dataset and persists one self-contained sklearn Pipeline per model.

Each saved Pipeline bundles its own preprocessing (scaling + one-hot
encoding), so the Streamlit app can feed it a *raw* CSV and call
`predict` directly. No preprocessing logic is duplicated in the UI, and
there is no risk of the app and the training run disagreeing about how a
column should be transformed.

Run:  python train_models.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET = "Revenue"

# Labels are derived by thresholding the positive-class score rather than by
# calling `.predict()`. For SVC the two disagree slightly: `.predict()` uses
# the sign of the decision function, while `predict_proba` comes from Platt
# scaling fitted on a separate internal cross-validation. Thresholding keeps
# the metrics recorded here identical to the ones the Streamlit app computes.
DECISION_THRESHOLD = 0.50

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DATA_RAW = ROOT / "data" / "online_shoppers_intention.csv"

# Columns that are stored as integers but are really unordered category IDs.
# Treating them as numeric would imply Browser 4 > Browser 2, which is
# meaningless, so they are one-hot encoded instead.
CATEGORICAL_ID_COLS = ["OperatingSystems", "Browser", "Region", "TrafficType"]
CATEGORICAL_TEXT_COLS = ["Month", "VisitorType"]
BOOLEAN_COLS = ["Weekend"]


def load_data() -> pd.DataFrame:
    """Read the raw CSV and coerce the boolean columns to 0/1 integers."""
    df = pd.read_csv(DATA_RAW)
    for col in BOOLEAN_COLS + [TARGET]:
        df[col] = df[col].astype(int)
    return df


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Scale the continuous columns, one-hot the categorical ones.

    `handle_unknown="ignore"` matters here: the deployed app accepts an
    arbitrary user-supplied CSV, which may contain a category level that
    never appeared in training. Without it the transform would raise.
    """
    categorical = CATEGORICAL_TEXT_COLS + CATEGORICAL_ID_COLS
    numeric = [c for c in features.columns if c not in categorical]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
    )


def build_models() -> dict[str, object]:
    """The five classifiers required by the assignment.

    All are left at sensible defaults rather than being tuned for the
    imbalanced target. That keeps the comparison between them honest and
    makes the effect of the 15/85 class split visible in the metrics --
    which is the point being analysed in the README observations.
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(
            kernel="rbf", probability=True, random_state=RANDOM_STATE
        ),
    }


def positive_scores(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    """Continuous score for the positive class, used for ROC-AUC.

    Prefers calibrated probabilities; falls back to the raw decision
    function for any estimator fitted without `probability=True`.
    """
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    return pipeline.decision_function(X)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    """The six evaluation metrics required by the assignment."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Stratified so the 15.5% positive rate is preserved in both halves.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    print(f"Positive rate -- train {y_train.mean():.4f}, test {y_test.mean():.4f}\n")

    # The held-out split is written out verbatim (raw, un-preprocessed) so
    # that it can be uploaded to the Streamlit app exactly as-is.
    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(ROOT / "test_data.csv", index=False)
    print(f"Wrote test_data.csv ({len(test_df)} rows)\n")

    results = {}
    for name, estimator in build_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X_train)),
                ("classifier", estimator),
            ]
        )
        print(f"Training {name} ...", end=" ", flush=True)
        pipeline.fit(X_train, y_train)

        y_score = positive_scores(pipeline, X_test)
        y_pred = (y_score >= DECISION_THRESHOLD).astype(int)
        results[name] = evaluate(y_test, y_pred, y_score)

        slug = name.lower().replace(" ", "_")
        joblib.dump(pipeline, MODEL_DIR / f"{slug}.joblib")
        print(f"done  (MCC {results[name]['MCC']:.4f})")

    # Cached so the README table and the app's baseline view do not need a
    # retrain to be regenerated.
    with open(MODEL_DIR / "metrics.json", "w") as fh:
        json.dump(results, fh, indent=2)

    table = pd.DataFrame(results).T.round(4)
    print("\n" + "=" * 78)
    print("HOLD-OUT TEST RESULTS")
    print("=" * 78)
    print(table.to_string())
    print(f"\nBest by MCC: {table['MCC'].idxmax()}")
    print(f"Best by AUC: {table['AUC'].idxmax()}")

    print("\nMarkdown (for README):\n")
    print(as_markdown(table))


def as_markdown(table: pd.DataFrame) -> str:
    """Render the results table as a Markdown table.

    Hand-rolled rather than using `DataFrame.to_markdown`, which needs the
    optional `tabulate` package -- not worth an extra deployment dependency
    for one convenience print.
    """
    header = "| ML Model Name | " + " | ".join(table.columns) + " |"
    rule = "|---" * (len(table.columns) + 1) + "|"
    rows = [
        "| " + name + " | " + " | ".join(f"{v:.4f}" for v in row) + " |"
        for name, row in table.iterrows()
    ]
    return "\n".join([header, rule, *rows])


if __name__ == "__main__":
    main()
