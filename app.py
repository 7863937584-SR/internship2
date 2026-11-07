from __future__ import annotations

import streamlit as st

from src.cache import get_sample_data, get_support_model
from src.config import TARGET_COLUMNS
from src.model import model_info
from src.ui import display_target_metrics, page_header

st.set_page_config(page_title="SupportIQ", page_icon="S", layout="wide")


page_header("SupportIQ Dashboard", "Customer support ticket analytics and ML predictions")

df = get_sample_data()
category_counts = df["category"].value_counts()
priority_counts = df["priority"].value_counts()
sentiment_counts = df["sentiment"].value_counts()
priority_order = ["Low", "Medium", "High", "Critical"]
priority_chart = priority_counts.reindex(priority_order).dropna()
high_priority_share = (
    priority_counts.get("High", 0) + priority_counts.get("Critical", 0)
) / max(len(df), 1)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Tickets", f"{len(df):,}")
col2.metric("Categories", df["category"].nunique())
col3.metric("High/Critical Share", f"{high_priority_share:.1%}")
col4.metric("Targets", ", ".join(TARGET_COLUMNS))

left, right = st.columns([3, 2])
with left:
    st.subheader("Tickets by Category")
    st.bar_chart(category_counts)
with right:
    st.subheader("Priority Distribution")
    st.bar_chart(priority_chart)

left, right = st.columns([3, 2])
with left:
    st.subheader("Sentiment Distribution")
    st.bar_chart(sentiment_counts)
with right:
    st.subheader("Recent Tickets")
    recent_columns = [
        column
        for column in ["ticket_id", "subject", "category", "priority", "sentiment", "channel", "created_at"]
        if column in df.columns
    ]
    st.dataframe(
        df.sort_values("created_at", ascending=False).head(10)[recent_columns],
        hide_index=True,
        width="stretch",
    )

st.divider()
st.subheader("Model Status")
try:
    artifact = get_support_model()
    info = model_info(artifact)
    metrics_col, meta_col = st.columns([2, 1])
    with metrics_col:
        st.markdown("**Holdout metrics**")
        display_target_metrics(info.get("metrics", {}))
    with meta_col:
        st.metric("Trained Rows", info.get("trained_rows", "Unknown"))
        st.metric("Artifact Version", info.get("version", "Unknown"))
        st.caption(f"Created: {info.get('created_at', 'Unknown')}")
except Exception as exc:
    st.warning(f"Model is not available yet: {exc}")
    st.page_link("pages/4_Train_Model.py", label="Train a model")
