"""Safe, cached CSV loading and JSON-safety helpers shared by all services."""
from __future__ import annotations

import hashlib
import math
import threading
from collections import OrderedDict
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.app.config import get_settings


class DatasetError(ValueError):
    """The dataset cannot be read or violates a configured limit."""


_CACHE: "OrderedDict[tuple, pd.DataFrame]" = OrderedDict()
_CACHE_MAX = 6
_lock = threading.Lock()


def load_dataframe(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV as a DataFrame (cached by path + mtime + size).

    The returned frame is shared: callers must treat it as **read-only** and call
    ``.copy()`` before modifying it.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError("Dataset file not found")

    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)

    with _lock:
        cached = _CACHE.get(key)
        if cached is not None:
            _CACHE.move_to_end(key)
            return cached

    settings = get_settings()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise DatasetError("Dataset contains no columns") from None
    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetError("Unable to read dataset file") from exc

    if len(df.columns) == 0:
        raise DatasetError("Dataset contains no columns")
    if len(df) > settings.max_dataset_rows:
        raise DatasetError(
            f"Dataset has {len(df)} rows; the limit is {settings.max_dataset_rows}"
        )
    if len(df.columns) > settings.max_dataset_columns:
        raise DatasetError(
            f"Dataset has {len(df.columns)} columns; "
            f"the limit is {settings.max_dataset_columns}"
        )

    df.columns = [str(c) for c in df.columns]

    with _lock:
        _CACHE[key] = df
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    return df


def clear_cache() -> None:
    with _lock:
        _CACHE.clear()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    numeric = set(numeric_columns(df))
    return [
        c
        for c in df.columns
        if c not in numeric and not pd.api.types.is_datetime64_any_dtype(df[c])
    ]


def json_safe(value):
    """Recursively convert numpy / pandas values to strict-JSON-safe values."""
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def clip_text(value, limit: int = 100) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"
