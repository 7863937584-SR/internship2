from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache import get_support_model
from src.model import predict_normalized_frame
from src.ui import page_header
from src.validation import csv_bytes, detect_delimiter, read_csv_upload, uploaded_file_fingerprint

st.set_page_config(page_title="Batch CSV Prediction", layout="wide")

page_header("Batch CSV Prediction", "Upload a CSV and predict category, priority, and sentiment for every row")

try:
    model = get_support_model()
except Exception as exc:
    st.error(f"Model unavailable: {exc}")
    st.page_link("pages/4_Train_Model.py", label="Train a model")
    st.stop()

uploaded_file = st.file_uploader("Upload ticket CSV", type=["csv"], help="CSV must include subject/message, ticket_text, description, body, or content.")

if uploaded_file is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

auto_delimiter = detect_delimiter(uploaded_file)
delimiter = st.selectbox("Delimiter", [",", ";", "\t", "|"], index=[",", ";", "\t", "|"].index(auto_delimiter) if auto_delimiter in [",", ";", "\t", "|"] else 0)

try:
    uploaded_file.seek(0)
    preview = pd.read_csv(uploaded_file, nrows=5, delimiter=delimiter, encoding="utf-8-sig")
    st.subheader("Preview")
    st.dataframe(preview, hide_index=True, width="stretch")
except Exception:
    try:
        uploaded_file.seek(0)
        preview = pd.read_csv(uploaded_file, nrows=5, delimiter=delimiter, encoding="latin1")
        st.subheader("Preview")
        st.dataframe(preview, hide_index=True, width="stretch")
    except Exception as exc:
        st.warning(f"Could not preview the CSV with the selected delimiter: {exc}")

file_key = f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}:{uploaded_file_fingerprint(uploaded_file)}:{delimiter}"

if st.button("Generate predictions", type="primary"):
    try:
        with st.spinner("Reading and predicting CSV rows..."):
            normalized, truncated = read_csv_upload(uploaded_file, delimiter=delimiter)
            result = predict_normalized_frame(model, normalized)
        st.session_state["batch_prediction"] = {
            "key": file_key,
            "result": result,
            "truncated": truncated,
            "input_rows": len(normalized),
        }
    except Exception as exc:
        st.session_state.pop("batch_prediction", None)
        st.error(str(exc))

stored = st.session_state.get("batch_prediction")
if stored and stored.get("key") == file_key:
    result = stored["result"]
    if stored.get("truncated"):
        st.warning(f"Only the first {stored.get('input_rows')} rows were processed because the upload limit is 10,000 rows.")
    st.subheader("Predictions")
    st.metric("Rows predicted", f"{len(result):,}")
    st.dataframe(result.head(100), hide_index=True, width="stretch")
    st.download_button(
        "Download predictions CSV",
        data=csv_bytes(result),
        file_name="supportiq_predictions.csv",
        mime="text/csv",
    )
elif stored:
    st.caption("The stored predictions belong to a different file or delimiter. Generate predictions again to update them.")
