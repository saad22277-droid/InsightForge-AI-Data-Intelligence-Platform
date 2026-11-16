"""Automated exploratory data analysis. Every number comes from the data."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from src.app.services.data_io import (
    categorical_columns,
    clip_text,
    json_safe,
    numeric_columns,
)
from src.app.services.quality import quality_report_from_dataframe

MAX_DISTRIBUTION_COLUMNS = 12
MAX_TOP_VALUES = 10
PLACEHOLDER_TOKENS = {
    "", "n/a", "na", "nan", "null", "none", "nil", "-", "--", "?", "unknown",
    "missing", "#n/a", "undefined",
}


def detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    """Object columns whose non-null values (nearly) all parse as dates."""
    found = []
    for column in categorical_columns(df):
        series = df[column].dropna()
        if series.empty or series.nunique() < 3:
            continue
        sample = series.astype(str).head(200)
        if not sample.str.contains(r"[-/:]", regex=True).mean() > 0.9:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= 0.9:
            found.append(column)
    return found


def _round(value, digits: int = 4):
    return json_safe(None if value is None else round(float(value), digits))


def numeric_statistics(df: pd.DataFrame) -> dict:
    stats = {}
    for column in numeric_columns(df):
        series = pd.to_numeric(df[column], errors="coerce")
        clean = series.dropna()
        stats[column] = {
            "count": int(clean.size),
            "missing": int(series.isna().sum()),
            "mean": _round(clean.mean()) if clean.size else None,
            "std": _round(clean.std()) if clean.size > 1 else None,
            "min": _round(clean.min()) if clean.size else None,
            "q1": _round(clean.quantile(0.25)) if clean.size else None,
            "median": _round(clean.median()) if clean.size else None,
            "q3": _round(clean.quantile(0.75)) if clean.size else None,
            "max": _round(clean.max()) if clean.size else None,
            "sum": _round(clean.sum()) if clean.size else None,
            "skewness": _round(clean.skew()) if clean.size > 2 else None,
            "kurtosis": _round(clean.kurt()) if clean.size > 3 else None,
            "zeros": int((clean == 0).sum()),
            "unique": int(clean.nunique()),
        }
    return stats


def categorical_statistics(df: pd.DataFrame) -> dict:
    stats = {}
    total = len(df)
    for column in categorical_columns(df):
        series = df[column]
        counts = series.dropna().astype(str).value_counts()
        stats[column] = {
            "count": int(series.notna().sum()),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique()),
            "mode": clip_text(counts.index[0]) if len(counts) else None,
            "top_values": [
                {
                    "value": clip_text(value),
                    "count": int(count),
                    "percentage": round(count / total * 100, 2) if total else 0.0,
                }
                for value, count in counts.head(MAX_TOP_VALUES).items()
            ],
        }
    return stats


def missing_analysis(df: pd.DataFrame) -> dict:
    total_rows = len(df)
    per_column = [
        {
            "column": str(column),
            "missing": int(df[column].isna().sum()),
            "percentage": (
                round(float(df[column].isna().mean() * 100), 2) if total_rows else 0.0
            ),
        }
        for column in df.columns
    ]
    per_column.sort(key=lambda item: (-item["missing"], item["column"]))
    rows_with_missing = int(df.isna().any(axis=1).sum())
    return {
        "total_missing_cells": int(df.isna().sum().sum()),
        "rows_with_missing": rows_with_missing,
        "complete_rows": int(total_rows - rows_with_missing),
        "columns": per_column,
        "column_with_most_missing": (
            per_column[0]["column"] if per_column and per_column[0]["missing"] else None
        ),
    }


def duplicate_analysis(df: pd.DataFrame) -> dict:
    mask = df.duplicated(keep="first")
    count = int(mask.sum())
    return {
        "duplicate_rows": count,
        "percentage": round(count / len(df) * 100, 2) if len(df) else 0.0,
        "example_row_indices": [int(i) for i in df.index[mask][:10]],
    }


def _shape_label(skew: float | None) -> str:
    if skew is None:
        return "insufficient data"
    if abs(skew) < 0.5:
        return "approximately symmetric"
    if abs(skew) < 1:
        return "moderately right-skewed" if skew > 0 else "moderately left-skewed"
    return "highly right-skewed" if skew > 0 else "highly left-skewed"


def distribution_analysis(df: pd.DataFrame, bins: int = 10) -> dict:
    result = {}
    for column in numeric_columns(df)[:MAX_DISTRIBUTION_COLUMNS]:
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            continue
        if values.nunique() == 1:
            edges = np.array([values.iloc[0] - 0.5, values.iloc[0] + 0.5])
            counts = np.array([len(values)])
        else:
            counts, edges = np.histogram(values.to_numpy(), bins=bins)
        skew = float(values.skew()) if len(values) > 2 else None
        result[column] = {
            "bins": [
                {
                    "start": float(edges[i]),
                    "end": float(edges[i + 1]),
                    "count": int(counts[i]),
                }
                for i in range(len(counts))
            ],
            "skewness": _round(skew),
            "shape": _shape_label(None if skew is None or np.isnan(skew) else skew),
        }
    return result


def _strength(r: float) -> str:
    a = abs(r)
    if a >= 0.8:
        return "very strong"
    if a >= 0.6:
        return "strong"
    if a >= 0.4:
        return "moderate"
    if a >= 0.2:
        return "weak"
    return "negligible"


def correlation_analysis(df: pd.DataFrame, top_n: int = 10) -> dict:
    columns = [
        c for c in numeric_columns(df) if pd.to_numeric(df[c], errors="coerce").nunique() > 1
    ]
    if len(columns) < 2:
        return {"columns": columns, "matrix": [], "top_pairs": []}

    numeric = df[columns].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr(method="pearson")
    matrix = [
        [json_safe(None if np.isnan(v) else round(float(v), 4)) for v in row]
        for row in corr.to_numpy()
    ]
    pairs = []
    for i, a in enumerate(columns):
        for j in range(i + 1, len(columns)):
            value = corr.iloc[i, j]
            if np.isnan(value):
                continue
            pairs.append(
                {
                    "column_a": a,
                    "column_b": columns[j],
                    "correlation": round(float(value), 4),
                    "strength": _strength(float(value)),
                    "direction": "positive" if value > 0 else "negative",
                    "observations": int(numeric[[a, columns[j]]].dropna().shape[0]),
                }
            )
    pairs.sort(key=lambda p: -abs(p["correlation"]))
    return {"columns": columns, "matrix": matrix, "top_pairs": pairs[:top_n]}


def outlier_analysis(df: pd.DataFrame, iqr_multiplier: float = 1.5) -> dict:
    columns = {}
    total = 0
    for column in numeric_columns(df):
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.size < 4 or values.nunique() < 2:
            continue
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        outliers = values[(values < lower) | (values > upper)]
        total += int(outliers.size)
        columns[column] = {
            "method": f"IQR x {iqr_multiplier}",
            "lower_bound": _round(lower),
            "upper_bound": _round(upper),
            "outlier_count": int(outliers.size),
            "outlier_percentage": round(outliers.size / values.size * 100, 2),
            "min_outlier": _round(outliers.min()) if outliers.size else None,
            "max_outlier": _round(outliers.max()) if outliers.size else None,
            "example_values": [
                _round(v) for v in outliers.sort_values(key=abs, ascending=False).head(5)
            ],
        }
    return {"total_outliers": total, "columns": columns}


def column_insights(df: pd.DataFrame) -> list[dict]:
    insights: list[dict] = []
    rows = len(df)
    datetime_cols = set(detect_datetime_columns(df))

    def add(column, severity, kind, message):
        insights.append(
            {"column": column, "severity": severity, "type": kind, "message": message}
        )

    for column in df.columns:
        series = df[column]
        missing_pct = series.isna().mean() * 100 if rows else 0
        unique = series.nunique()
        if rows and series.isna().all():
            add(column, "warning", "empty", "Column contains no values.")
            continue
        if missing_pct >= 40:
            add(column, "warning", "high_missing",
                f"{missing_pct:.1f}% of values are missing.")
        elif missing_pct > 0:
            add(column, "info", "some_missing",
                f"{missing_pct:.1f}% of values are missing.")
        if unique <= 1 and rows > 1:
            add(column, "warning", "constant",
                "Column has a single constant value and carries no information.")
        if (
            rows > 1
            and unique == series.notna().sum()
            and not pd.api.types.is_float_dtype(series)
            and column not in datetime_cols
        ):
            add(column, "info", "identifier",
                "Every value is unique, so this looks like an identifier. "
                "It is excluded from ML features automatically.")
        if column in datetime_cols:
            add(column, "info", "datetime_like",
                "Values look like dates but are stored as text.")

    for column in numeric_columns(df):
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.size > 2 and values.nunique() > 1:
            skew = float(values.skew())
            if abs(skew) >= 1:
                add(column, "info", "skewed",
                    f"Distribution is {_shape_label(skew)} (skewness {skew:.2f}); "
                    "the mean may be a poor summary.")

    for column in categorical_columns(df):
        series = df[column].dropna()
        if not len(series):
            continue
        unique = series.nunique()
        if unique > 50 and unique / len(series) > 0.5 and column not in datetime_cols:
            add(column, "info", "high_cardinality",
                f"{unique} distinct values; high-cardinality text is hard to model.")
        counts = series.astype(str).value_counts(normalize=True)
        if 1 < unique <= 50 and counts.iloc[0] >= 0.9:
            add(column, "info", "dominant_category",
                f"'{clip_text(counts.index[0], 40)}' makes up "
                f"{counts.iloc[0] * 100:.1f}% of values.")
        stripped = series.astype(str)
        if (stripped != stripped.str.strip()).any():
            add(column, "info", "whitespace",
                "Some values have leading/trailing whitespace.")
        if series.astype(str).str.strip().str.lower().isin(PLACEHOLDER_TOKENS - {""}).any():
            add(column, "warning", "placeholder_values",
                "Contains placeholder tokens such as 'N/A', '-' or 'unknown'.")
    return insights


def generate_eda(df: pd.DataFrame) -> dict:
    numeric = numeric_columns(df)
    categorical = categorical_columns(df)
    overview = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "datetime_like_columns": detect_datetime_columns(df),
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
    }
    return json_safe(
        {
            "overview": overview,
            "numeric_statistics": numeric_statistics(df),
            "categorical_statistics": categorical_statistics(df),
            "missing": missing_analysis(df),
            "duplicates": duplicate_analysis(df),
            "distributions": distribution_analysis(df),
            "correlations": correlation_analysis(df),
            "outliers": outlier_analysis(df),
            "column_insights": column_insights(df),
            "quality": quality_report_from_dataframe(df),
        }
    )
