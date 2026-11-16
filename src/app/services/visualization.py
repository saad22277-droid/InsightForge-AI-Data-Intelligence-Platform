from pathlib import Path

import pandas as pd


SUPPORTED_CHART_TYPES = {"histogram", "bar", "scatter"}


def generate_chart_data(
    file_path: str,
    chart_type: str,
    x_column: str,
    y_column: str | None = None,
    bins: int = 10,
) -> dict:
    """Generate JSON-ready chart data from a CSV dataset."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError("Dataset file not found")

    if chart_type not in SUPPORTED_CHART_TYPES:
        raise ValueError(
            f"Unsupported chart type. Choose from: "
            f"{', '.join(sorted(SUPPORTED_CHART_TYPES))}"
        )

    if chart_type == "histogram" and not 1 <= bins <= 100:
        raise ValueError("bins must be between 1 and 100")

    df = pd.read_csv(path)

    if x_column not in df.columns:
        raise ValueError(f"Column not found: {x_column}")

    if chart_type == "histogram":
        return _generate_histogram(df, x_column, bins)

    if chart_type == "bar":
        return _generate_bar_chart(df, x_column)

    if chart_type == "scatter":
        if not y_column:
            raise ValueError("y_column is required for scatter charts")

        if y_column not in df.columns:
            raise ValueError(f"Column not found: {y_column}")

        return _generate_scatter_plot(df, x_column, y_column)

    raise ValueError("Unsupported chart type")


def _generate_histogram(
    df: pd.DataFrame,
    column: str,
    bins: int,
) -> dict:
    numeric_values = pd.to_numeric(df[column], errors="coerce").dropna()

    if numeric_values.empty:
        raise ValueError(
            f"Column contains no valid numeric values: {column}"
        )

    histogram_counts, histogram_edges = _histogram(numeric_values, bins)

    return {
        "chart_type": "histogram",
        "x_column": column,
        "bins": [
            {
                "start": float(histogram_edges[i]),
                "end": float(histogram_edges[i + 1]),
                "count": int(histogram_counts[i]),
            }
            for i in range(len(histogram_counts))
        ],
    }


def _histogram(values: pd.Series, bins: int):
    """Return histogram counts and bin edges."""
    import numpy as np

    counts, edges = np.histogram(values.to_numpy(), bins=bins)
    return counts, edges


def _generate_bar_chart(df: pd.DataFrame, column: str) -> dict:
    value_counts = (
        df[column]
        .fillna("(Missing)")
        .astype(str)
        .value_counts()
        .head(20)
    )

    return {
        "chart_type": "bar",
        "x_column": column,
        "data": [
            {"category": category, "count": int(count)}
            for category, count in value_counts.items()
        ],
    }


def _generate_scatter_plot(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> dict:
    x_values = pd.to_numeric(df[x_column], errors="coerce")
    y_values = pd.to_numeric(df[y_column], errors="coerce")

    valid_rows = x_values.notna() & y_values.notna()

    if not valid_rows.any():
        raise ValueError("No valid numeric pairs found for scatter plot")

    points = [
        {"x": float(x), "y": float(y)}
        for x, y in zip(
            x_values[valid_rows].head(500),
            y_values[valid_rows].head(500),
        )
    ]

    return {
        "chart_type": "scatter",
        "x_column": x_column,
        "y_column": y_column,
        "data": points,
    }

