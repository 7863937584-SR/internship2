from __future__ import annotations

import streamlit as st

from src.data import load_sample_tickets
from src.model import load_support_model


@st.cache_data(show_spinner=False)
def get_sample_data():
    return load_sample_tickets()


@st.cache_resource(show_spinner=False)
def get_support_model():
    return load_support_model()
