"""Automated data-cleaning engine: analyse -> recommend -> preview -> apply.

Operations never modify the original file. ``apply_operations`` works on a copy
and the route layer stores the result as a *new dataset version*.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from src.app.services.data_io import (
    categorical_columns,
    clip_text,
    json_safe,
    numeric_columns,
)
from src.app.services.eda import PLACEHOLDER_TOKENS, detect_datetime_columns

NON_NEGATIVE_HINT = re.compile(
    r"(age|price|salary|income|revenue|sales|amount|quantity|qty|count|cost|"
    r"height|weight|population|distance|duration)",
    re.IGNORECASE,
)


def _col(df: pd.DataFrame, name: str) -> str:
    if name not in df.columns:
        raise ValueError(f"Column not found: {name}")
    return name


def _to_numeric_text(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(r"[$€£,\s]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _snake(name: str) -> str:
    text = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip())
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text or "column"


# ---------------------------------------------------------------- recommendations
def analyze_cleaning(df: pd.DataFrame) -> dict:
    recs: list[dict] = []

    def add(category, severity, title, reason, operation, column=None, impact=None):
        recs.append(
            {
                "id": f"rec-{len(recs) + 1}",
                "category": category,
                "severity": severity,
                "column": column,
                "title": title,
                "reason": reason,
                "estimated_impact": impact,
                "operation": operation,
            }
        )

    rows = len(df)
    numeric = set(numeric_columns(df))
    datetime_cols = set(detect_datetime_columns(df))

    # Placeholder tokens in text columns (invalid values)
    for column in categorical_columns(df):
        text = df[column].dropna().astype(str).str.strip().str.lower()
        tokens = sorted(set(text[text.isin(PLACEHOLDER_TOKENS - {""})]))
        if tokens:
            n = int(text.isin(tokens).sum())
            add("invalid_values", "warning", f"Replace placeholder values in '{column}'",
                f"{n} cell(s) contain placeholders ({', '.join(map(repr, tokens[:5]))}) "
                "that represent missing data.",
                {"type": "replace_invalid", "column": column, "values": tokens},
                column, f"{n} cell(s) become missing")

    # Whitespace / case inconsistencies
    for column in categorical_columns(df):
        if column in datetime_cols:
            continue
        values = df[column].dropna().astype(str)
        if values.empty:
            continue
        stripped = values.str.strip().str.replace(r"\s+", " ", regex=True)
        changed = int((stripped != values).sum())
        case_variants = stripped.nunique() - stripped.str.lower().nunique()
        if changed or case_variants > 0:
            lower = case_variants > 0
            add("standardization", "info", f"Standardize text in '{column}'",
                (f"{changed} value(s) have extra whitespace. " if changed else "")
                + (f"{case_variants} category variant(s) differ only by letter case."
                   if lower else ""),
                {"type": "standardize_text", "column": column, "strip": True,
                 "lowercase": lower, "collapse_whitespace": True},
                column, f"{changed} value(s) trimmed" if changed else None)

    # Numbers / dates stored as text
    for column in categorical_columns(df):
        values = df[column].dropna()
        if values.empty:
            continue
        if column in datetime_cols:
            add("data_types", "info", f"Convert '{column}' to datetime",
                "Values parse as dates but are stored as text.",
                {"type": "convert_dtype", "column": column, "dtype": "datetime"}, column)
            continue
        values = values[
            ~values.astype(str).str.strip().str.lower().isin(PLACEHOLDER_TOKENS)
        ]
        if values.empty:
            continue
        parsed = _to_numeric_text(values)
        if parsed.notna().mean() >= 0.9:
            bad = int(parsed.isna().sum())
            add("data_types", "warning", f"Convert '{column}' to numeric",
                "Almost all values are numbers stored as text"
                + (f"; {bad} non-numeric value(s) would become missing." if bad else "."),
                {"type": "convert_dtype", "column": column, "dtype": "numeric"}, column)

    # Invalid numeric values (negatives where they make no sense)
    for column in numeric:
        if NON_NEGATIVE_HINT.search(column):
            negatives = int((pd.to_numeric(df[column], errors="coerce") < 0).sum())
            if negatives:
                add("invalid_values", "warning", f"Negative values in '{column}'",
                    f"{negatives} negative value(s) in a column that is normally "
                    "non-negative.",
                    {"type": "invalid_to_missing", "column": column, "minimum": 0},
                    column, f"{negatives} value(s) become missing")

    # Duplicates
    duplicates = int(df.duplicated().sum())
    if duplicates:
        add("duplicates", "warning", "Remove duplicate rows",
            f"{duplicates} fully duplicated row(s) ({duplicates / rows * 100:.1f}%).",
            {"type": "drop_duplicates", "keep": "first"}, None,
            f"{duplicates} row(s) removed")

    # Empty / mostly-missing columns, then imputation
    for column in df.columns:
        pct = df[column].isna().mean() * 100 if rows else 0
        if rows and df[column].isna().all():
            add("missing_values", "warning", f"Drop empty column '{column}'",
                "The column contains no values.",
                {"type": "drop_columns", "columns": [column]}, column)
        elif pct >= 60:
            add("missing_values", "warning", f"Consider dropping '{column}'",
                f"{pct:.1f}% of values are missing; imputing would mostly be guesswork.",
                {"type": "drop_columns", "columns": [column]}, column)
        elif pct > 0:
            missing = int(df[column].isna().sum())
            if column in numeric:
                skew = pd.to_numeric(df[column], errors="coerce").skew()
                strategy = "median" if pd.isna(skew) or abs(skew) >= 0.5 else "mean"
                add("missing_values", "info", f"Fill missing '{column}' with {strategy}",
                    f"{missing} missing value(s) ({pct:.1f}%).",
                    {"type": "fill_missing", "column": column, "strategy": strategy},
                    column, f"{missing} value(s) filled")
            elif column not in datetime_cols:
                add("missing_values", "info", f"Fill missing '{column}' with mode",
                    f"{missing} missing value(s) ({pct:.1f}%).",
                    {"type": "fill_missing", "column": column, "strategy": "mode"},
                    column, f"{missing} value(s) filled")

    # Outliers
    for column in numeric:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.size < 8 or values.nunique() < 3:
            continue
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        count = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
        if count:
            add("outliers", "info", f"Review outliers in '{column}'",
                f"{count} value(s) fall outside 1.5 x IQR. They may be legitimate; "
                "clipping is the least destructive option.",
                {"type": "handle_outliers", "column": column, "method": "iqr",
                 "action": "clip"}, column, f"{count} value(s) clipped")

    # Column names
    bad_names = [c for c in df.columns if _snake(c) != c]
    if bad_names:
        add("standardization", "info", "Normalize column names to snake_case",
            f"{len(bad_names)} column name(s) contain spaces, capitals or symbols "
            f"(e.g. {', '.join(repr(clip_text(c, 30)) for c in bad_names[:3])}).",
            {"type": "normalize_column_names"}, None)

    order = {"warning": 0, "info": 1}
    recs.sort(key=lambda r: order[r["severity"]])
    for i, rec in enumerate(recs, start=1):
        rec["id"] = f"rec-{i}"
    return json_safe(
        {
            "row_count": int(rows),
            "column_count": int(len(df.columns)),
            "recommendation_count": len(recs),
            "recommendations": recs,
            "notes": [
                "Recommendations are suggestions. Nothing changes until you apply them.",
                "Applying creates a new dataset version; the original stays available.",
            ],
        }
    )


# -------------------------------------------------------------------- operations
def apply_operations(df: pd.DataFrame, operations: list) -> tuple[pd.DataFrame, list[dict]]:
    """Apply validated operations in order to a *copy* of ``df``."""
    work = df.copy()
    reports = []
    for op in operations:
        before_rows, before_cols = len(work), len(work.columns)
        handler = _HANDLERS[op.type]
        work, report = handler(work, op)
        report.setdefault("type", op.type)
        report["rows_before"], report["rows_after"] = before_rows, len(work)
        report["columns_before"], report["columns_after"] = before_cols, len(work.columns)
        reports.append(report)
    return work, reports


def _drop_duplicates(df, op):
    if op.subset:
        for c in op.subset:
            _col(df, c)
    n = int(df.duplicated(subset=op.subset, keep=op.keep).sum())
    out = df.drop_duplicates(subset=op.subset, keep=op.keep).reset_index(drop=True)
    return out, {"rows_affected": n, "message": f"Removed {n} duplicate row(s)."}


def _fill_missing(df, op):
    column = _col(df, op.column)
    series = df[column]
    missing = int(series.isna().sum())
    if op.strategy in {"mean", "median"}:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() == 0:
            raise ValueError(f"Column '{column}' is not numeric; use mode or constant")
        fill = numeric.mean() if op.strategy == "mean" else numeric.median()
        df[column] = numeric.fillna(fill)
        detail = f"{op.strategy} = {fill:.6g}"
    elif op.strategy == "mode":
        modes = series.mode(dropna=True)
        if modes.empty:
            raise ValueError(f"Column '{column}' has no values to compute a mode")
        df[column] = series.fillna(modes.iloc[0])
        detail = f"mode = {clip_text(modes.iloc[0], 40)}"
    else:
        if op.value is None:
            raise ValueError("A value is required for the constant strategy")
        value = op.value
        if pd.api.types.is_numeric_dtype(series) and isinstance(value, str):
            value = pd.to_numeric(value, errors="coerce")
            if pd.isna(value):
                raise ValueError(f"'{op.value}' is not numeric but '{column}' is")
        df[column] = series.fillna(value)
        detail = f"constant = {clip_text(op.value, 40)}"
    return df, {"column": column, "rows_affected": missing,
                "message": f"Filled {missing} missing value(s) in '{column}' ({detail})."}


def _drop_missing_rows(df, op):
    if op.columns:
        for c in op.columns:
            _col(df, c)
    subset = op.columns or None
    n = int(df.isna().any(axis=1).sum() if not subset else df[subset].isna().any(axis=1).sum())
    out = df.dropna(subset=subset).reset_index(drop=True)
    return out, {"rows_affected": n, "message": f"Removed {n} row(s) with missing values."}


def _drop_columns(df, op):
    for c in op.columns:
        _col(df, c)
    if len(op.columns) >= len(df.columns):
        raise ValueError("Cannot drop every column")
    return df.drop(columns=list(op.columns)), {
        "rows_affected": 0, "message": f"Dropped column(s): {', '.join(op.columns)}."}


def _convert_dtype(df, op):
    column = _col(df, op.column)
    series = df[column]
    before_missing = int(series.isna().sum())
    if op.dtype == "numeric":
        converted = _to_numeric_text(series) if not pd.api.types.is_numeric_dtype(series) \
            else series
    elif op.dtype == "datetime":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            converted = pd.to_datetime(series, errors="coerce")
    elif op.dtype == "boolean":
        mapping = {"true": True, "false": False, "yes": True, "no": False,
                   "y": True, "n": False, "1": True, "0": False}
        converted = series.astype(str).str.strip().str.lower().map(mapping)
        converted = converted.where(series.notna())
    elif op.dtype == "category":
        converted = series.astype("category")
    else:
        converted = series.astype("string").where(series.notna())
    lost = int(converted.isna().sum()) - before_missing
    df[column] = converted
    return df, {"column": column, "rows_affected": int(series.notna().sum()),
                "message": f"Converted '{column}' to {op.dtype}"
                + (f"; {lost} value(s) could not be converted and became missing."
                   if lost > 0 else "."),
                "values_lost": max(lost, 0)}


def _handle_outliers(df, op):
    column = _col(df, op.column)
    values = pd.to_numeric(df[column], errors="coerce")
    if values.notna().sum() < 4:
        raise ValueError(f"Column '{column}' has too few numeric values")
    if op.method == "iqr":
        k = op.threshold or 1.5
        q1, q3 = values.quantile([0.25, 0.75])
        low, high = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    else:
        k = op.threshold or 3.0
        mean, std = values.mean(), values.std()
        if not std or np.isnan(std):
            raise ValueError(f"Column '{column}' has no variation")
        low, high = mean - k * std, mean + k * std
    mask = (values < low) | (values > high)
    n = int(mask.sum())
    if op.action == "clip":
        df[column] = values.clip(lower=low, upper=high)
    elif op.action == "set_missing":
        df[column] = values.mask(mask)
    else:
        df = df.loc[~mask].reset_index(drop=True)
    return df, {"column": column, "rows_affected": n,
                "message": f"{op.action}: {n} outlier(s) in '{column}' outside "
                           f"[{low:.6g}, {high:.6g}] ({op.method}, threshold {k})."}


def _normalize_column_names(df, op):
    seen: dict[str, int] = {}
    new_names = []
    for name in df.columns:
        base = _snake(name)
        seen[base] = seen.get(base, 0) + 1
        new_names.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    renamed = sum(1 for a, b in zip(df.columns, new_names) if a != b)
    df.columns = new_names
    return df, {"rows_affected": 0, "message": f"Renamed {renamed} column(s) to snake_case."}


def _standardize_text(df, op):
    column = _col(df, op.column)
    series = df[column]
    if pd.api.types.is_numeric_dtype(series):
        raise ValueError(f"Column '{column}' is numeric, not text")
    text = series.astype("string")
    new = text
    if op.strip:
        new = new.str.strip()
    if op.collapse_whitespace:
        new = new.str.replace(r"\s+", " ", regex=True)
    if op.lowercase:
        new = new.str.lower()
    changed = int(((new != text) & text.notna()).sum())
    df[column] = new.astype(object).where(series.notna())
    return df, {"column": column, "rows_affected": changed,
                "message": f"Standardized {changed} value(s) in '{column}'."}


def _replace_invalid(df, op):
    column = _col(df, op.column)
    tokens = {t.strip().lower() for t in (op.values or PLACEHOLDER_TOKENS)}
    series = df[column]
    mask = series.astype("string").str.strip().str.lower().isin(tokens) & series.notna()
    n = int(mask.sum())
    df[column] = series.mask(mask)
    return df, {"column": column, "rows_affected": n,
                "message": f"Replaced {n} placeholder value(s) in '{column}' with missing."}


def _invalid_to_missing(df, op):
    column = _col(df, op.column)
    values = pd.to_numeric(df[column], errors="coerce")
    mask = pd.Series(False, index=df.index)
    if op.minimum is not None:
        mask |= values < op.minimum
    if op.maximum is not None:
        mask |= values > op.maximum
    n = int(mask.sum())
    df[column] = values.mask(mask)
    return df, {"column": column, "rows_affected": n,
                "message": f"Set {n} out-of-range value(s) in '{column}' to missing."}


def _scale_column(df, op):
    column = _col(df, op.column)
    values = pd.to_numeric(df[column], errors="coerce")
    if values.notna().sum() < 2:
        raise ValueError(f"Column '{column}' has too few numeric values")
    if op.method == "minmax":
        span = values.max() - values.min()
        if span == 0:
            raise ValueError(f"Column '{column}' is constant; cannot scale")
        df[column] = (values - values.min()) / span
    else:
        std = values.std()
        if not std or np.isnan(std):
            raise ValueError(f"Column '{column}' is constant; cannot scale")
        df[column] = (values - values.mean()) / std
    return df, {"column": column, "rows_affected": int(values.notna().sum()),
                "message": f"Scaled '{column}' with {op.method} scaling "
                           "(original units are lost)."}


_HANDLERS = {
    "drop_duplicates": _drop_duplicates,
    "fill_missing": _fill_missing,
    "drop_missing_rows": _drop_missing_rows,
    "drop_columns": _drop_columns,
    "convert_dtype": _convert_dtype,
    "handle_outliers": _handle_outliers,
    "normalize_column_names": _normalize_column_names,
    "standardize_text": _standardize_text,
    "replace_invalid": _replace_invalid,
    "invalid_to_missing": _invalid_to_missing,
    "scale_column": _scale_column,
}


# ---------------------------------------------------------------------- preview
def _snapshot(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
        "missing_by_column": {str(c): int(df[c].isna().sum()) for c in df.columns},
    }


def preview_cleaning(df: pd.DataFrame, operations: list) -> tuple[pd.DataFrame, dict]:
    cleaned, reports = apply_operations(df, operations)
    before, after = _snapshot(df), _snapshot(cleaned)

    dtype_changes = [
        {"column": c, "from": before["dtypes"][c], "to": after["dtypes"][c]}
        for c in after["dtypes"]
        if c in before["dtypes"] and before["dtypes"][c] != after["dtypes"][c]
    ]
    missing_changes = [
        {"column": c, "from": before["missing_by_column"][c],
         "to": after["missing_by_column"][c]}
        for c in after["missing_by_column"]
        if c in before["missing_by_column"]
        and before["missing_by_column"][c] != after["missing_by_column"][c]
    ]
    stat_changes = []
    for column in numeric_columns(cleaned):
        if column in df.columns and column in numeric_columns(df):
            a = pd.to_numeric(df[column], errors="coerce")
            b = pd.to_numeric(cleaned[column], errors="coerce")
            if a.notna().any() and b.notna().any() and (
                abs(a.mean() - b.mean()) > 1e-9 or abs(a.std() - b.std()) > 1e-9
                or a.min() != b.min() or a.max() != b.max()
            ):
                stat_changes.append(
                    {"column": column,
                     "before": {"mean": a.mean(), "std": a.std(),
                                "min": a.min(), "max": a.max()},
                     "after": {"mean": b.mean(), "std": b.std(),
                               "min": b.min(), "max": b.max()}})

    warnings_out = []
    if after["rows"] == 0:
        warnings_out.append("These operations remove every row.")
    if after["rows"] < before["rows"] * 0.5:
        warnings_out.append(
            f"More than half of the rows ({before['rows'] - after['rows']}) are removed.")
    if any(op.type == "scale_column" for op in operations):
        warnings_out.append("Scaling changes units; interpret values with care.")

    return cleaned, json_safe(
        {
            "before": {k: v for k, v in before.items() if k != "missing_by_column"},
            "after": {k: v for k, v in after.items() if k != "missing_by_column"},
            "delta": {
                "rows": after["rows"] - before["rows"],
                "columns": after["columns"] - before["columns"],
                "missing_cells": after["missing_cells"] - before["missing_cells"],
                "duplicate_rows": after["duplicate_rows"] - before["duplicate_rows"],
            },
            "dtype_changes": dtype_changes,
            "missing_value_changes": missing_changes,
            "numeric_changes": stat_changes,
            "operations": reports,
            "warnings": warnings_out,
        }
    )
