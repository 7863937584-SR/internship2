from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache import get_sample_data, get_support_model
from src.config import MAX_UPLOAD_ROWS
from src.model import model_info
from src.training import train_and_save
from src.ui import display_target_metrics, page_header
from src.validation import detect_delimiter, read_training_upload, uploaded_file_fingerprint

st.set_page_config(page_title="Train Model", layout="wide")

page_header("Train Model", "Train a multi-target ticket classifier and save it for predictions")

try:
    current_model = get_support_model()
    current_info = model_info(current_model)
    st.success(
        f"Current model: {current_info.get('trained_rows', 'unknown')} rows, "
        f"created {current_info.get('created_at', 'unknown')}"
    )
except Exception as exc:
    st.info(f"No usable model is currently loaded: {exc}")

data_source = st.radio("Training data source", ["Bundled sample dataset", "Upload training CSV"], horizontal=True)
training_df = None
uploaded_file = None
truncated = False

if data_source == "Bundled sample dataset":
    training_df = get_sample_data()
    st.metric("Training rows", f"{len(training_df):,}")
    label_cols = st.columns(3)
    label_cols[0].metric("Categories", training_df["category"].nunique())
    label_cols[1].metric("Priorities", training_df["priority"].nunique())
    label_cols[2].metric("Sentiments", training_df["sentiment"].nunique())
else:
    uploaded_file = st.file_uploader(
        "Upload labeled training CSV",
        type=["csv"],
        help="Required labels: category, priority, sentiment. Required text: subject/message, ticket_text, description, body, or content.",
    )
    if uploaded_file is None:
        st.info("Upload a labeled CSV to train from custom data.")
    else:
        auto_delimiter = detect_delimiter(uploaded_file)
        delimiter = st.selectbox(
            "Delimiter",
            [",", ";", "\t", "|"],
            index=[",", ";", "\t", "|"].index(auto_delimiter) if auto_delimiter in [",", ";", "\t", "|"] else 0,
        )
        try:
            uploaded_file.seek(0)
            preview = pd.read_csv(uploaded_file, nrows=5, delimiter=delimiter, encoding="utf-8-sig")
            st.subheader("Upload preview")
            st.dataframe(preview, hide_index=True, width="stretch")
        except Exception:
            try:
                uploaded_file.seek(0)
                preview = pd.read_csv(uploaded_file, nrows=5, delimiter=delimiter, encoding="latin1")
                st.subheader("Upload preview")
                st.dataframe(preview, hide_index=True, width="stretch")
            except Exception as exc:
                st.warning(f"Could not preview the CSV: {exc}")

st.divider()
confirm_overwrite = st.checkbox("I understand training will overwrite the current model artifact", value=False)
can_train = confirm_overwrite and (training_df is not None or uploaded_file is not None)

if st.button("Train and save model", type="primary", disabled=not can_train):
    try:
        if data_source == "Upload training CSV":
            file_key = f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}:{uploaded_file_fingerprint(uploaded_file)}"
            with st.spinner("Validating uploaded training data..."):
                training_df, truncated = read_training_upload(uploaded_file, delimiter=delimiter)
        else:
            file_key = "sample_dataset"

        with st.spinner("Training category, priority, and sentiment models..."):
            artifact = train_and_save(training_df)
        get_support_model.clear()
        st.session_state["training_result"] = {
            "key": file_key,
            "trained_rows": artifact.get("trained_rows"),
            "created_at": artifact.get("created_at"),
            "metrics": artifact.get("metrics", {}),
            "truncated": truncated,
        }
        st.success("Model trained and saved successfully.")
    except Exception as exc:
        st.error(str(exc))

result = st.session_state.get("training_result")
if result:
    st.divider()
    st.subheader("Latest training result")
    st.metric("Trained rows", result.get("trained_rows", "Unknown"))
    st.caption(f"Created: {result.get('created_at', 'Unknown')}")
    if result.get("truncated"):
        st.warning(f"Training data was truncated to {MAX_UPLOAD_ROWS:,} rows.")
    display_target_metrics(result.get("metrics", {}))
