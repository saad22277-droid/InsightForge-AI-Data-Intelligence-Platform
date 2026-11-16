
from pathlib import Path
import math

import pandas as pd

from src.app.schemas.dataset import DatasetFilter


SUPPORTED_OPERATORS = {
    "eq", "ne", "gt", "gte", "lt", "lte",
    "contains", "in", "is_null", "not_null",
}


def query_dataset(
    file_path: str,
    filters: list[DatasetFilter],
    limit: int = 100,
    offset: int = 0,
) -> dict:
    """Filter a CSV dataset and return a paginated JSON-ready result."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError("Dataset file not found")

    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")

    if offset < 0:
        raise ValueError("offset must be non-negative")

    if len(filters) > 20:
        raise ValueError("A maximum of 20 filters is allowed")

    df = pd.read_csv(path)

    for condition in filters:
        if condition.column not in df.columns:
            raise ValueError(f"Column not found: {condition.column}")

        if condition.operator not in SUPPORTED_OPERATORS:
            raise ValueError(
                f"Unsupported operator: {condition.operator}"
            )

        mask = _build_mask(df[condition.column], condition)
        df = df.loc[mask]

    total_matches = len(df)
    page = df.iloc[offset:offset + limit].copy()

    # Convert missing values and NumPy scalar values to JSON-compatible values.
    page = page.astype(object).where(pd.notna(page), None)
    records = page.to_dict(orient="records")

    return {
        "total_matches": total_matches,
        "limit": limit,
        "offset": offset,
        "returned_rows": len(records),
        "rows": records,
    }


def _build_mask(
    series: pd.Series,
    condition: DatasetFilter,
) -> pd.Series:
    operator = condition.operator
    value = condition.value

    if operator == "is_null":
        return series.isna()

    if operator == "not_null":
        return series.notna()

    if operator == "contains":
        if not isinstance(value, str):
            raise ValueError("contains requires a string value")

        return series.astype("string").str.contains(
            value,
            case=False,
            regex=False,
            na=False,
        )

    if operator == "in":
        if not isinstance(value, list):
            raise ValueError("in requires a list value")

        return series.isin(value)

    if value is None or isinstance(value, list):
        raise ValueError(f"{operator} requires a single non-null value")

    if operator in {"gt", "gte", "lt", "lte"}:
        numeric_value = _as_number(value)
        numeric_series = pd.to_numeric(series, errors="coerce")

        if operator == "gt":
            return numeric_series > numeric_value
        if operator == "gte":
            return numeric_series >= numeric_value
        if operator == "lt":
            return numeric_series < numeric_value
        return numeric_series <= numeric_value

    if operator in {"eq", "ne"}:
        # Compare numeric columns numerically when a numeric value is supplied.
        if pd.api.types.is_numeric_dtype(series) and isinstance(
            value, (int, float)
        ) and not isinstance(value, bool):
            comparable = pd.to_numeric(series, errors="coerce")
            matches = comparable == value
        else:
            matches = series.eq(value)

        matches = matches.fillna(False)
        return matches if operator == "eq" else ~matches & series.notna()

    raise ValueError(f"Unsupported operator: {operator}")


def _as_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Comparison operators require a numeric value")

    number = float(value)

    if not math.isfinite(number):
        raise ValueError("Numeric filter values must be finite")

    return number
