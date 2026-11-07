from __future__ import annotations

import csv
import hashlib
from io import BytesIO
from typing import BinaryIO

import pandas as pd

from src.config import MAX_UPLOAD_ROWS
from src.data import ensure_text_present, ensure_training_labels, normalize_dataframe


def detect_delimiter(uploaded_file: BinaryIO, default: str = ",") -> str:
    try:
        uploaded_file.seek(0)
        sample = uploaded_file.read(8192)
        uploaded_file.seek(0)
        text = sample.decode("utf-8-sig", errors="ignore")
        try:
            dialect = csv.Sniffer().sniff(text, delimiters=",;\t|")
            return dialect.delimiter
        except csv.Error:
            return default
    except Exception:
        return default
    finally:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass


def uploaded_file_fingerprint(uploaded_file: BinaryIO) -> str:
    digest = hashlib.sha256()
    try:
        uploaded_file.seek(0)
        bytes_read = 0
        while bytes_read < 1024 * 1024:
            chunk = uploaded_file.read(min(64 * 1024, 1024 * 1024 - bytes_read))
            if not chunk:
                break
            digest.update(chunk)
            bytes_read += len(chunk)
        uploaded_file.seek(0)
    except Exception:
        pass
    return digest.hexdigest()


def _read_csv_with_encoding(
    uploaded_file: BinaryIO,
    delimiter: str,
    max_rows: int,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "latin1"):
        try:
            uploaded_file.seek(0)
            return pd.read_csv(
                uploaded_file,
                encoding=encoding,
                delimiter=delimiter,
                nrows=max_rows + 1,
            )
        except UnicodeDecodeError as exc:
            last_error = exc
        except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as exc:
            last_error = exc
            break

    if last_error is not None:
        raise ValueError(f"Could not read the CSV file: {last_error}") from last_error
    raise ValueError("Could not read the CSV file.")


def read_csv_upload(
    uploaded_file: BinaryIO,
    delimiter: str = ",",
    max_rows: int = MAX_UPLOAD_ROWS,
) -> tuple[pd.DataFrame, bool]:
    raw = _read_csv_with_encoding(uploaded_file, delimiter=delimiter, max_rows=max_rows)
    truncated = len(raw) > max_rows
    if truncated:
        raw = raw.head(max_rows)
    if raw.empty:
        raise ValueError("The uploaded CSV has no data rows.")

    normalized = normalize_dataframe(raw)
    ensure_text_present(normalized, "uploaded CSV")
    return normalized, truncated


def read_training_upload(
    uploaded_file: BinaryIO,
    delimiter: str = ",",
    max_rows: int = MAX_UPLOAD_ROWS,
) -> tuple[pd.DataFrame, bool]:
    normalized, truncated = read_csv_upload(
        uploaded_file,
        delimiter=delimiter,
        max_rows=max_rows,
    )
    ensure_training_labels(normalized)
    return normalized, truncated


def csv_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()
