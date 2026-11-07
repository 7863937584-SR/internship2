from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import MODEL_PATH, TARGET_COLUMNS
from src.data import ensure_training_labels


def _canonical_labels(df: pd.DataFrame) -> pd.DataFrame:
    canonical = df.copy()
    for target in TARGET_COLUMNS:
        if target in canonical.columns:
            canonical[target] = (
                canonical[target]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"[_\-]+", " ", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.title()
            )
    return canonical



def _classifier_for_labels(y: pd.Series, random_state: int):
    if y.nunique(dropna=False) < 2:
        return DummyClassifier(strategy="most_frequent")
    return LogisticRegression(
        C=4.0,
        class_weight="balanced",
        max_iter=1000,
        random_state=random_state,
    )


def _pipeline_for_labels(y: pd.Series, random_state: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95,
                    max_features=30_000,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", _classifier_for_labels(y, random_state)),
        ]
    )


def train_pipelines(df: pd.DataFrame, random_state: int = 42) -> dict[str, Pipeline]:
    ensure_training_labels(df)
    canonical = _canonical_labels(df)
    features = canonical["text"].fillna("").astype(str)
    pipelines = {}

    for target in TARGET_COLUMNS:
        labels = canonical[target].fillna("").astype(str).str.strip()
        pipeline = _pipeline_for_labels(labels, random_state)
        pipeline.fit(features, labels)
        pipelines[target] = pipeline

    return pipelines


def _safe_stratify(labels: pd.Series) -> pd.Series | None:
    counts = labels.value_counts(dropna=False)
    if len(counts) > 1 and counts.min() >= 2:
        return labels
    return None


def evaluate_holdout(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, dict]:
    canonical = _canonical_labels(df)

    if len(canonical) < 10:
        return {
            "warning": "Holdout evaluation skipped because fewer than 10 labeled rows were provided.",
            "samples": len(canonical),
        }

    train_df, test_df = train_test_split(
        canonical,
        test_size=test_size,
        random_state=random_state,
        stratify=_safe_stratify(canonical["category"]),
    )
    train_features = train_df["text"].fillna("").astype(str)
    test_features = test_df["text"].fillna("").astype(str)
    metrics = {"samples": len(test_df), "train_samples": len(train_df)}

    for target in TARGET_COLUMNS:
        y_train = train_df[target].fillna("").astype(str).str.strip()
        y_test = test_df[target].fillna("").astype(str).str.strip()
        pipeline = _pipeline_for_labels(y_train, random_state)
        pipeline.fit(train_features, y_train)
        predictions = pipeline.predict(test_features)

        metrics[target] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "f1_weighted": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
            "report": classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0,
            ),
        }

    return metrics


def train_and_save(
    df: pd.DataFrame,
    path: Path = MODEL_PATH,
    evaluate: bool = True,
    random_state: int = 42,
) -> dict:
    ensure_training_labels(df)
    canonical = _canonical_labels(df)
    pipelines = train_pipelines(canonical, random_state=random_state)
    metrics = evaluate_holdout(canonical, random_state=random_state) if evaluate else {}

    artifact = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "trained_rows": int(len(canonical)),
        "feature_columns": ["text"],
        "target_columns": list(TARGET_COLUMNS),
        "classes": {
            target: [str(label) for label in pipeline.classes_]
            for target, pipeline in pipelines.items()
        },
        "metrics": metrics,
        "models": pipelines,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(artifact, temporary_path)
    temporary_path.replace(path)
    return artifact
