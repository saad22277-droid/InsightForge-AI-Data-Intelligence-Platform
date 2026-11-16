"""Chart data generators for the interactive visualization workspace.

These complement the original ``visualization.generate_chart_data`` (which is
kept unchanged for API compatibility) with line, box, heatmap, missing-value and
aggregated bar charts. All values come straight from the dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.app.services.data_io import (
    clip_text,
    json_safe,
    numeric_columns,
)

CHART_TYPES = {
    "histogram", "distribution", "bar", "line", "scatter", "box", "heatmap", "missing",
}
AGGREGATIONS = {"count", "mean", "sum", "median", "min", "max"}
MAX_POINTS = 500
MAX_GROUPS = 12


def _require(df: pd.DataFrame, column: str | None, label: str) -> str:
    if not column:
        raise ValueError(f"{label} is required for this chart")
    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")
    return column


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce")
    if values.notna().sum() == 0:
        raise ValueError(f"Column contains no valid numeric values: {column}")
    return values


def _aggregate(frame: pd.DataFrame, by: str, value: str | None, agg: str) -> pd.Series:
    if agg == "count" or value is None:
        return frame.groupby(by, dropna=False).size()
    series = pd.to_numeric(frame[value], errors="coerce")
    return series.groupby(frame[by]).agg(agg)


def generate_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_column: str | None = None,
    y_column: str | None = None,
    group_by: str | None = None,
    agg: str = "count",
    bins: int = 10,
    columns: list[str] | None = None,
) -> dict:
    if chart_type not in CHART_TYPES:
        raise ValueError(
            "Unsupported chart type. Choose from: " + ", ".join(sorted(CHART_TYPES))
        )
    if agg not in AGGREGATIONS:
        raise ValueError("agg must be one of: " + ", ".join(sorted(AGGREGATIONS)))
    if not 1 <= bins <= 100:
        raise ValueError("bins must be between 1 and 100")

    if chart_type in {"histogram", "distribution"}:
        return _histogram(df, _require(df, x_column, "x_column"), bins, chart_type)
    if chart_type == "bar":
        return _bar(df, _require(df, x_column, "x_column"), y_column, agg)
    if chart_type == "line":
        return _line(df, _require(df, x_column, "x_column"),
                     _require(df, y_column, "y_column"), agg)
    if chart_type == "scatter":
        return _scatter(df, _require(df, x_column, "x_column"),
                        _require(df, y_column, "y_column"), group_by)
    if chart_type == "box":
        return _box(df, _require(df, x_column, "x_column"), group_by)
    if chart_type == "heatmap":
        return _heatmap(df, columns)
    return _missing(df)


def _histogram(df, column, bins, chart_type) -> dict:
    values = _numeric(df, column).dropna()
    counts, edges = np.histogram(values.to_numpy(), bins=bins)
    result = {
        "chart_type": chart_type,
        "x_column": column,
        "bins": [
            {"start": float(edges[i]), "end": float(edges[i + 1]), "count": int(counts[i])}
            for i in range(len(counts))
        ],
        "n": int(values.size),
    }
    if chart_type == "distribution":
        result["summary"] = {
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()) if values.size > 1 else None,
            "skewness": float(values.skew()) if values.size > 2 else None,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return json_safe(result)


def _bar(df, x, y, agg) -> dict:
    if y:
        _require(df, y, "y_column")
        if agg == "count":
            agg = "mean"
        _numeric(df, y)
    series = _aggregate(df.assign(**{x: df[x].fillna("(Missing)").astype(str)}), x, y, agg)
    series = series.sort_values(ascending=False).head(20)
    return json_safe(
        {
            "chart_type": "bar",
            "x_column": x,
            "y_column": y,
            "aggregation": agg if y else "count",
            "data": [
                {"category": clip_text(k, 60), "value": float(v), "count": float(v)}
                for k, v in series.items()
            ],
        }
    )


def _line(df, x, y, agg) -> dict:
    yv = pd.to_numeric(df[y], errors="coerce")
    xs = df[x]
    x_kind = "numeric"
    if pd.api.types.is_numeric_dtype(xs):
        xv = pd.to_numeric(xs, errors="coerce")
    else:
        parsed = pd.to_datetime(xs, errors="coerce")
        if parsed.notna().mean() >= 0.8:
            xv, x_kind = parsed, "datetime"
        else:
            xv, x_kind = xs.astype(str), "category"
    frame = pd.DataFrame({"x": xv, "y": yv}).dropna()
    if frame.empty:
        raise ValueError("No valid (x, y) pairs for a line chart")
    effective = "mean" if agg == "count" else agg
    grouped = frame.groupby("x")["y"].agg(effective).reset_index()
    if x_kind != "category":
        grouped = grouped.sort_values("x")
    if len(grouped) > MAX_POINTS:
        grouped = grouped.iloc[np.linspace(0, len(grouped) - 1, MAX_POINTS).astype(int)]
    return json_safe(
        {
            "chart_type": "line",
            "x_column": x,
            "y_column": y,
            "x_kind": x_kind,
            "aggregation": effective,
            "data": [
                {"x": (v.isoformat() if hasattr(v, "isoformat") else v), "y": float(w)}
                for v, w in zip(grouped["x"], grouped["y"])
            ],
        }
    )


def _scatter(df, x, y, group_by) -> dict:
    frame = pd.DataFrame({"x": pd.to_numeric(df[x], errors="coerce"),
                          "y": pd.to_numeric(df[y], errors="coerce")})
    if group_by:
        _require(df, group_by, "group_by")
        frame["g"] = df[group_by].fillna("(Missing)").astype(str)
    frame = frame.dropna(subset=["x", "y"])
    if frame.empty:
        raise ValueError("No valid numeric pairs found for scatter plot")
    total = len(frame)
    if total > MAX_POINTS:
        frame = frame.sample(MAX_POINTS, random_state=42)
    correlation = None
    if total > 2 and frame["x"].nunique() > 1 and frame["y"].nunique() > 1:
        correlation = float(frame["x"].corr(frame["y"]))
    result = {
        "chart_type": "scatter",
        "x_column": x,
        "y_column": y,
        "group_by": group_by,
        "total_points": int(total),
        "displayed_points": int(len(frame)),
        "correlation": correlation,
    }
    if group_by:
        top = frame["g"].value_counts().head(8).index
        result["series"] = [
            {"name": clip_text(g, 40),
             "data": [{"x": float(a), "y": float(b)}
                      for a, b in zip(sub["x"], sub["y"])]}
            for g in top for sub in [frame[frame["g"] == g]]
        ]
    else:
        result["data"] = [{"x": float(a), "y": float(b)}
                          for a, b in zip(frame["x"], frame["y"])]
    return json_safe(result)


def _box_stats(values: pd.Series) -> dict:
    q1, med, q3 = values.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    low_fence, high_fence = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = values[(values >= low_fence) & (values <= high_fence)]
    outliers = values[(values < low_fence) | (values > high_fence)]
    return {
        "n": int(values.size),
        "min": float(inside.min()) if inside.size else float(values.min()),
        "q1": float(q1),
        "median": float(med),
        "q3": float(q3),
        "max": float(inside.max()) if inside.size else float(values.max()),
        "mean": float(values.mean()),
        "outlier_count": int(outliers.size),
        "outliers": [float(v) for v in outliers.head(20)],
    }


def _box(df, x, group_by) -> dict:
    values = _numeric(df, x)
    if group_by:
        _require(df, group_by, "group_by")
        frame = pd.DataFrame({"v": values, "g": df[group_by].fillna("(Missing)").astype(str)})
        frame = frame.dropna(subset=["v"])
        top = frame["g"].value_counts().head(MAX_GROUPS).index
        boxes = [{"label": clip_text(g, 40), **_box_stats(frame.loc[frame["g"] == g, "v"])}
                 for g in top]
    else:
        boxes = [{"label": x, **_box_stats(values.dropna())}]
    return json_safe({"chart_type": "box", "x_column": x, "group_by": group_by, "boxes": boxes})


def _heatmap(df, columns) -> dict:
    if columns:
        for column in columns:
            _require(df, column, "columns")
        selected = [c for c in columns if c in numeric_columns(df)]
    else:
        selected = numeric_columns(df)
    selected = [c for c in selected if pd.to_numeric(df[c], errors="coerce").nunique() > 1][:25]
    if len(selected) < 2:
        raise ValueError("At least two numeric columns with variation are required")
    corr = df[selected].apply(pd.to_numeric, errors="coerce").corr()
    return json_safe(
        {
            "chart_type": "heatmap",
            "columns": selected,
            "matrix": [[None if np.isnan(v) else round(float(v), 4) for v in row]
                       for row in corr.to_numpy()],
        }
    )


def _missing(df) -> dict:
    rows = len(df)
    data = [
        {
            "column": str(c),
            "missing": int(df[c].isna().sum()),
            "percentage": round(float(df[c].isna().mean() * 100), 2) if rows else 0.0,
        }
        for c in df.columns
    ]
    data.sort(key=lambda d: -d["missing"])
    return json_safe({"chart_type": "missing", "total_rows": int(rows), "data": data})
