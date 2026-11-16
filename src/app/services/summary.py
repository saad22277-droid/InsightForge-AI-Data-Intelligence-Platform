from pathlib import Path

import pandas as pd


def generate_summary(file_path: str) -> dict:
    """Generate summary statistics for a CSV dataset."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError("Dataset file not found")

    df = pd.read_csv(path)

    numeric_columns = df.select_dtypes(include="number")

    numeric_statistics = {}

    for column in numeric_columns.columns:
        series = numeric_columns[column]

        numeric_statistics[column] = {
            "mean": _to_json_number(series.mean()),
            "median": _to_json_number(series.median()),
            "min": _to_json_number(series.min()),
            "max": _to_json_number(series.max()),
            "standard_deviation": _to_json_number(series.std()),
        }

    column_details = {}

    for column in df.columns:
        series = df[column]

        column_details[column] = {
            "data_type": str(series.dtype),
            "missing_values": int(series.isna().sum()),
            "missing_percentage": (
                round(float(series.isna().mean() * 100), 2)
                if len(df) > 0
                else 0.0
            ),
            "unique_values": int(series.nunique()),
        }

    return {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": list(df.columns),
        "column_details": column_details,
        "numeric_statistics": numeric_statistics,
    }


def _to_json_number(value):
    """Convert pandas numeric results to JSON-safe values."""
    if pd.isna(value):
        return None

    return float(value)