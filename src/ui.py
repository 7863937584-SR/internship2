from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import TARGET_COLUMNS


def page_header(title: str, subtitle: str | None = None) -> None:
    st.header(title)
    if subtitle:
        st.caption(subtitle)


def show_error(exc: Exception) -> None:
    st.error(str(exc))


def confidence_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def display_target_metrics(metrics: dict) -> None:
    if not metrics:
        st.info("No holdout metrics are available for this model.")
        return

    if "warning" in metrics:
        st.warning(metrics["warning"])
        return

    summary_rows = []
    for target in TARGET_COLUMNS:
        target_metrics = metrics.get(target, {})
        summary_rows.append(
            {
                "Target": target,
                "Accuracy": round(target_metrics.get("accuracy", 0.0), 4),
                "Weighted F1": round(target_metrics.get("f1_weighted", 0.0), 4),
                "Test Samples": metrics.get("samples", 0),
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

    for target in TARGET_COLUMNS:
        report = metrics.get(target, {}).get("report", {})
        if not report:
            continue
        rows = []
        for label, values in report.items():
            if label in {"accuracy", "macro avg", "weighted avg"}:
                continue
            if not isinstance(values, dict):
                continue
            rows.append(
                {
                    "Class": label,
                    "Precision": round(values.get("precision", 0.0), 4),
                    "Recall": round(values.get("recall", 0.0), 4),
                    "F1": round(values.get("f1-score", 0.0), 4),
                    "Support": int(values.get("support", 0)),
                }
            )
        if rows:
            with st.expander(f"{target} class metrics"):
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
