from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODEL_PATH
from src.data import normalize_dataframe, synthetic_tickets
from src.model import load_support_model, predict_frame
from src.validation import read_csv_upload


def test_normalize_dataframe_uses_aliases_and_builds_text():
    raw = pd.DataFrame(
        [
            {
                "Title": "Cannot login",
                "Body": "The password reset email never arrives.",
                "Source": "Email",
                "Product": "Identity",
                "Sentiment": "negative",
            }
        ]
    )
    normalized = normalize_dataframe(raw)
    assert "cannot login" in normalized.loc[0, "text"].lower()
    assert "password reset" in normalized.loc[0, "text"].lower()
    assert "channel:email" in normalized.loc[0, "text"].lower()
    assert normalized.loc[0, "sentiment"] == "negative"


def test_synthetic_tickets_has_required_labels():
    df = synthetic_tickets(n_rows=25, seed=7)
    assert len(df) == 25
    assert {"category", "priority", "sentiment"}.issubset(df.columns)
    assert df[["category", "priority", "sentiment"]].notna().all().all()


def test_read_csv_upload_normalizes_text_columns():
    csv_data = "ticket_text,priority\nMy invoice was charged twice,High\n"
    normalized, truncated = read_csv_upload(io.BytesIO(csv_data.encode("utf-8")))
    assert not truncated
    assert "charged twice" in normalized.loc[0, "text"]


def test_trained_model_predicts_labels():
    if not MODEL_PATH.exists():
        pytest.skip("Model artifact has not been trained yet")

    model = load_support_model()
    result = predict_frame(
        model,
        pd.DataFrame(
            [
                {
                    "subject": "Duplicate charge",
                    "message": "I was charged twice for the same invoice.",
                    "channel": "email",
                }
            ]
        ),
    )
    assert result.loc[0, "category"] in model["classes"]["category"]
    assert 0.0 <= result.loc[0, "category_confidence"] <= 1.0
