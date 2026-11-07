from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

from src.config import MODEL_PATH, TARGET_COLUMNS
from src.data import ensure_text_present, normalize_dataframe


def load_support_model(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. Train the model from the Train Model page."
        )

    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "models" not in artifact:
        raise ValueError("The model artifact is invalid or corrupted.")
    for target in TARGET_COLUMNS:
        if target not in artifact["models"]:
            raise ValueError(f"The model artifact is missing the '{target}' model.")
    return artifact


def model_info(artifact: dict) -> dict:
    return {
        "version": artifact.get("version"),
        "created_at": artifact.get("created_at"),
        "trained_rows": artifact.get("trained_rows"),
        "target_columns": artifact.get("target_columns", list(TARGET_COLUMNS)),
        "metrics": artifact.get("metrics", {}),
        "classes": artifact.get("classes", {}),
    }


def _top_prediction(pipeline, texts: list[str]) -> tuple[list[str], list[float]]:
    probabilities = pipeline.predict_proba(texts)
    classes = [str(label) for label in pipeline.classes_]
    best_indexes = probabilities.argmax(axis=1)
    predictions = [classes[index] for index in best_indexes]
    confidences = [float(probabilities[row, index]) for row, index in enumerate(best_indexes)]
    return predictions, confidences


def predict_texts(artifact: dict, texts: Iterable[str]) -> pd.DataFrame:
    text_list = [str(text or "")[:20000] for text in texts]
    if not text_list:
        return pd.DataFrame(columns=[*TARGET_COLUMNS, *(f"{target}_confidence" for target in TARGET_COLUMNS)])

    output = {"text": text_list}
    for target in TARGET_COLUMNS:
        predictions, confidences = _top_prediction(artifact["models"][target], text_list)
        output[target] = predictions
        output[f"{target}_confidence"] = confidences

    return pd.DataFrame(output)


def predict_normalized_frame(artifact: dict, normalized: pd.DataFrame) -> pd.DataFrame:
    normalized = normalized.reset_index(drop=True)
    ensure_text_present(normalized, "prediction data")
    predictions = predict_texts(artifact, normalized["text"].tolist()).drop(columns=["text"])

    context_columns = [
        column
        for column in [
            "ticket_id",
            "subject",
            "body",
            "channel",
            "product_area",
            "customer_tenure_days",
            "created_at",
        ]
        if column in normalized.columns
    ]
    for column in reversed(context_columns):
        predictions.insert(0, column, normalized[column])
    if "ticket_id" not in predictions.columns:
        predictions.insert(0, "ticket_id", "")
    return predictions


def predict_frame(artifact: dict, df: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_dataframe(df)
    return predict_normalized_frame(artifact, normalized)
