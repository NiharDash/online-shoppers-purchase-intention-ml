"""
Streamlit front-end for the Online Shoppers Purchasing Intention classifiers.

The app is strictly an *inference and evaluation* surface. It loads the
five pipelines that `train_models.py` persisted and scores whatever CSV
the user supplies. Nothing is refitted here -- uploading a file never
changes a model, it only produces a fresh evaluation of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "Revenue"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(
    page_title="Online Shopper Purchase Intention",
    page_icon="🛒",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; }
      .hero {
          background: linear-gradient(90deg, #1f4e79 0%, #2e7bb8 100%);
          color: #fff; padding: 18px 26px; border-radius: 10px; margin-bottom: 18px;
      }
      .hero h1 { margin: 0; font-size: 1.6rem; }
      .hero p  { margin: 6px 0 0 0; opacity: .92; font-size: .95rem; }
      .verdict {
          border-left: 5px solid #2e7d32; background: #f1f8f1;
          padding: 10px 16px; border-radius: 6px; margin-top: 8px;
      }
      .warnbox {
          border-left: 5px solid #c9a227; background: #fdf8e8;
          padding: 10px 16px; border-radius: 6px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🛒 Online Shopper Purchasing Intention</h1>
      <p>Will an e-commerce browsing session end in a purchase? Five classifiers,
         evaluated side by side on data you supply.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(filename: str):
    """Load one persisted pipeline. Cached so a dropdown change is instant."""
    return joblib.load(MODEL_DIR / filename)


@st.cache_data(show_spinner=False)
def load_reference_metrics() -> dict:
    """Metrics recorded at training time, used as the baseline comparison."""
    path = MODEL_DIR / "metrics.json"
    return json.load(open(path)) if path.exists() else {}


@st.cache_data(show_spinner=False)
def load_bundled_test_data() -> pd.DataFrame | None:
    path = ROOT / "test_data.csv"
    return pd.read_csv(path) if path.exists() else None


def score_positive_class(model, X: pd.DataFrame) -> np.ndarray:
    """Continuous score for the positive class, for ROC-AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def compute_metrics(y_true, y_pred, y_score) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# --------------------------------------------------------------------------
# Sidebar: data source and model choice
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader(
        "Upload a test CSV",
        type="csv",
        help=(
            "Must contain every input feature plus the 'Revenue' target column. "
            "Models are only scored, never retrained."
        ),
    )
    use_bundled = st.checkbox(
        "Use the bundled test_data.csv instead", value=uploaded is None
    )

    st.header("2 · Model")
    selected_model = st.selectbox("Classifier", list(MODEL_FILES.keys()))

    st.header("3 · Decision threshold")
    threshold = st.slider(
        "Probability cut-off for predicting a purchase",
        min_value=0.05, max_value=0.95, value=0.50, step=0.05,
        help=(
            "The default 0.50 is rarely optimal when only ~15% of sessions "
            "convert. Lowering it trades precision for recall."
        ),
    )

    st.divider()
    st.caption(
        "UCI Online Shoppers Purchasing Intention · 12,330 sessions · "
        "17 features · 15.5% positive"
    )


# --------------------------------------------------------------------------
# Resolve the dataframe to evaluate on
# --------------------------------------------------------------------------
df: pd.DataFrame | None = None
source_label = ""

if uploaded is not None and not use_bundled:
    df = pd.read_csv(uploaded)
    source_label = f"uploaded file — `{uploaded.name}`"
elif use_bundled:
    df = load_bundled_test_data()
    source_label = "bundled `test_data.csv` (held-out split)"

if df is None:
    st.info("Upload a test CSV in the sidebar, or tick the bundled-data box to begin.")
    st.stop()

if TARGET not in df.columns:
    st.error(
        f"The CSV has no `{TARGET}` column. The target is required in order to "
        f"compute evaluation metrics.\n\nColumns found: {', '.join(df.columns)}"
    )
    st.stop()

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

st.success(f"Loaded **{len(df):,}** rows from {source_label} — {X.shape[1]} features.")

if y.nunique() < 2:
    st.markdown(
        '<div class="warnbox">The target column contains only one class, so AUC '
        "is undefined. Upload a file with both classes present.</div>",
        unsafe_allow_html=True,
    )
    st.stop()


# --------------------------------------------------------------------------
# Score every model once; reused by both tabs
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Scoring models…")
def score_all(frame: pd.DataFrame, cut: float) -> tuple[dict, dict]:
    """Return (metrics per model, positive-class scores per model)."""
    feats = frame.drop(columns=[TARGET])
    truth = frame[TARGET].astype(int)
    metrics, scores = {}, {}
    for name, fname in MODEL_FILES.items():
        model = load_model(fname)
        s = score_positive_class(model, feats)
        preds = (s >= cut).astype(int)
        metrics[name] = compute_metrics(truth, preds, s)
        scores[name] = s
    return metrics, scores


all_metrics, all_scores = score_all(df, threshold)

tab_single, tab_compare, tab_data = st.tabs(
    ["📊 Selected model", "🏁 All five compared", "🔍 Data preview"]
)


# --------------------------------------------------------------------------
# Tab 1 — the selected model in detail
# --------------------------------------------------------------------------
with tab_single:
    st.subheader(selected_model)

    scores = all_scores[selected_model]
    y_pred = (scores >= threshold).astype(int)
    m = all_metrics[selected_model]

    cols = st.columns(6)
    for col, (label, value) in zip(cols, m.items()):
        col.metric(label, f"{value:.4f}")

    reference = load_reference_metrics().get(selected_model)
    if reference and threshold == 0.50:
        delta = m["MCC"] - reference["MCC"]
        st.caption(
            f"Recorded at training time on the held-out split: "
            f"MCC {reference['MCC']:.4f} · AUC {reference['AUC']:.4f}  "
            f"(difference on this data: {delta:+.4f})"
        )

    left, right = st.columns(2)

    with left:
        st.markdown("##### Confusion matrix")
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["No purchase", "Purchase"],
            yticklabels=["No purchase", "Purchase"],
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig, width="stretch")

        tn, fp, fn, tp = cm.ravel()
        st.caption(
            f"{tp:,} purchases correctly caught · {fn:,} missed · "
            f"{fp:,} false alarms · {tn:,} correctly ignored"
        )

    with right:
        st.markdown("##### ROC curve")
        fpr, tpr, _ = roc_curve(y, scores)
        fig2, ax2 = plt.subplots(figsize=(4.2, 3.4))
        ax2.plot(fpr, tpr, lw=2, label=f"AUC = {m['AUC']:.4f}")
        ax2.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
        ax2.set_xlabel("False positive rate")
        ax2.set_ylabel("True positive rate")
        ax2.legend(loc="lower right", fontsize=8)
        st.pyplot(fig2, width="stretch")

    st.markdown("##### Classification report")
    report = classification_report(
        y, y_pred, target_names=["No purchase", "Purchase"],
        output_dict=True, zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")


# --------------------------------------------------------------------------
# Tab 2 — all five side by side
# --------------------------------------------------------------------------
with tab_compare:
    st.subheader("All five classifiers on this dataset")

    table = pd.DataFrame(all_metrics).T.round(4)
    st.dataframe(
        table.style
        .background_gradient(cmap="Greens", subset=["MCC", "AUC", "F1"])
        .format("{:.4f}"),
        width="stretch",
    )

    best_mcc = table["MCC"].idxmax()
    best_auc = table["AUC"].idxmax()
    st.markdown(
        f'<div class="verdict"><b>Best by MCC:</b> {best_mcc} '
        f"({table.loc[best_mcc, 'MCC']:.4f}) &nbsp;·&nbsp; "
        f"<b>Best by AUC:</b> {best_auc} ({table.loc[best_auc, 'AUC']:.4f})"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("##### Metric comparison")
    metric_choice = st.radio(
        "Metric", list(table.columns), horizontal=True, index=5
    )
    fig3, ax3 = plt.subplots(figsize=(8, 3.2))
    ordered = table[metric_choice].sort_values()
    colors = ["#2e7bb8"] * len(ordered)
    colors[-1] = "#2e7d32"
    ax3.barh(ordered.index, ordered.values, color=colors)
    ax3.set_xlabel(metric_choice)
    ax3.set_xlim(0, 1)
    for i, v in enumerate(ordered.values):
        ax3.text(v + 0.012, i, f"{v:.4f}", va="center", fontsize=9)
    st.pyplot(fig3, width="stretch")

    st.markdown("##### ROC curves, all models")
    fig4, ax4 = plt.subplots(figsize=(5.2, 4))
    for name in MODEL_FILES:
        fpr, tpr, _ = roc_curve(y, all_scores[name])
        ax4.plot(fpr, tpr, lw=1.6,
                 label=f"{name} ({all_metrics[name]['AUC']:.3f})")
    ax4.plot([0, 1], [0, 1], "k--", lw=1)
    ax4.set_xlabel("False positive rate")
    ax4.set_ylabel("True positive rate")
    ax4.legend(loc="lower right", fontsize=7.5)
    st.pyplot(fig4, width="content")

    csv = table.to_csv().encode()
    st.download_button(
        "Download these results as CSV", csv,
        file_name="model_comparison.csv", mime="text/csv",
    )


# --------------------------------------------------------------------------
# Tab 3 — what was actually loaded
# --------------------------------------------------------------------------
with tab_data:
    st.subheader("Uploaded data")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Features", X.shape[1])
    c3.metric("Positive rate", f"{y.mean():.2%}")

    st.markdown("##### First 20 rows")
    st.dataframe(df.head(20), width="stretch")

    st.markdown("##### Class balance")
    counts = y.value_counts().sort_index()
    fig5, ax5 = plt.subplots(figsize=(4, 2.4))
    ax5.bar(["No purchase", "Purchase"], counts.values,
            color=["#9db8d2", "#2e7d32"])
    for i, v in enumerate(counts.values):
        ax5.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax5.set_ylabel("Sessions")
    st.pyplot(fig5, width="content")

    st.caption(
        "The target is heavily imbalanced, which is why accuracy alone is "
        "misleading here — a model that predicted 'no purchase' for every "
        "session would already score about 85%."
    )
