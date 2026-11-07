from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache import get_support_model
from src.model import predict_frame
from src.ui import confidence_percent, page_header

st.set_page_config(page_title="Predict Single Ticket", layout="wide")

page_header("Single Ticket Prediction", "Predict category, priority, and sentiment for one support ticket")

try:
    model = get_support_model()
except Exception as exc:
    st.error(f"Model unavailable: {exc}")
    st.page_link("pages/4_Train_Model.py", label="Train a model")
    st.stop()


def fingerprint(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


with st.form("single_ticket_form"):
    subject = st.text_input("Subject", placeholder="Short summary of the issue")
    message = st.text_area("Message", placeholder="Paste the customer message here", height=180)
    channel_col, product_col, tenure_col = st.columns(3)
    channel = channel_col.selectbox("Channel", ["email", "chat", "phone", "social", "portal", "unknown"])
    product_area = product_col.text_input("Product area", placeholder="billing, api, console...")
    tenure = tenure_col.number_input("Customer tenure (days)", min_value=0, max_value=100000, value=365)
    submitted = st.form_submit_button("Predict")

payload = {
    "subject": subject.strip(),
    "message": message.strip(),
    "channel": channel,
    "product_area": product_area.strip(),
    "customer_tenure_days": int(tenure),
}
current_fingerprint = fingerprint(payload)

if submitted:
    if not payload["subject"] and not payload["message"]:
        st.error("Provide a subject or message before predicting.")
    else:
        with st.spinner("Predicting ticket..."):
            result = predict_frame(model, pd.DataFrame([payload]))
        st.session_state["single_ticket_prediction"] = {
            "fingerprint": current_fingerprint,
            "result": result,
        }

stored = st.session_state.get("single_ticket_prediction")
if stored and stored.get("fingerprint") == current_fingerprint:
    result = stored["result"].iloc[0]
    st.subheader("Prediction")
    category_col, priority_col, sentiment_col = st.columns(3)
    category_col.metric("Category", result["category"])
    priority_col.metric("Priority", result["priority"])
    sentiment_col.metric("Sentiment", result["sentiment"])

    confidence_col1, confidence_col2, confidence_col3 = st.columns(3)
    confidence_col1.metric("Category Confidence", confidence_percent(result["category_confidence"]))
    confidence_col2.metric("Priority Confidence", confidence_percent(result["priority_confidence"]))
    confidence_col3.metric("Sentiment Confidence", confidence_percent(result["sentiment_confidence"]))
elif stored:
    st.caption("The stored prediction belongs to different inputs. Submit the form again to update it.")
