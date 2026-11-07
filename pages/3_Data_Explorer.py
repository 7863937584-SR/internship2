from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache import get_sample_data
from src.ui import page_header
from src.validation import csv_bytes

st.set_page_config(page_title="Data Explorer", layout="wide")

page_header("Data Explorer", "Explore the bundled support ticket dataset")

df = get_sample_data()

with st.sidebar:
    st.subheader("Filters")
    categories = sorted(df["category"].dropna().unique().tolist())
    priorities = ["Low", "Medium", "High", "Critical"]
    sentiments = sorted(df["sentiment"].dropna().unique().tolist())
    channels = sorted(df["channel"].dropna().unique().tolist())

    selected_categories = st.multiselect("Category", categories, default=categories)
    selected_priorities = st.multiselect("Priority", priorities, default=priorities)
    selected_sentiments = st.multiselect("Sentiment", sentiments, default=sentiments)
    selected_channel = st.selectbox("Channel", ["All"] + channels)
    search = st.text_input("Search text", placeholder="Search subject or message")
    max_tenure = st.slider("Max customer tenure (days)", 0, int(df["customer_tenure_days"].max() or 0), int(df["customer_tenure_days"].max() or 0))

filtered = df.copy()
if selected_categories:
    filtered = filtered[filtered["category"].isin(selected_categories)]
if selected_priorities:
    filtered = filtered[filtered["priority"].isin(selected_priorities)]
if selected_sentiments:
    filtered = filtered[filtered["sentiment"].isin(selected_sentiments)]
if selected_channel != "All":
    filtered = filtered[filtered["channel"] == selected_channel]
if search.strip():
    filtered = filtered[filtered["text"].str.contains(search.strip(), case=False, na=False)]
filtered = filtered[filtered["customer_tenure_days"].fillna(0) <= max_tenure]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Filtered Tickets", f"{len(filtered):,}")
col2.metric("Categories", filtered["category"].nunique() if len(filtered) else 0)
col3.metric("Avg Tenure", f"{filtered['customer_tenure_days'].mean():.0f}" if len(filtered) else "0")
col4.metric("High/Critical", f"{filtered['priority'].isin(['High', 'Critical']).sum():,}" if len(filtered) else "0")

if filtered.empty:
    st.info("No tickets match the current filters.")
    st.stop()

left, middle, right = st.columns(3)
with left:
    st.subheader("Category")
    st.bar_chart(filtered["category"].value_counts())
with middle:
    st.subheader("Priority")
    st.bar_chart(filtered["priority"].value_counts().reindex(priorities).dropna())
with right:
    st.subheader("Sentiment")
    st.bar_chart(filtered["sentiment"].value_counts())

st.subheader("Tickets")
display_columns = [
    column
    for column in ["ticket_id", "subject", "body", "category", "priority", "sentiment", "channel", "product_area", "customer_tenure_days", "created_at"]
    if column in filtered.columns
]
st.dataframe(filtered[display_columns].head(500), hide_index=True, width="stretch")
st.download_button(
    "Download filtered CSV",
    data=csv_bytes(filtered[display_columns]),
    file_name="supportiq_filtered_tickets.csv",
    mime="text/csv",
)
