from pathlib import Path

import pandas as pd

def generate_quality_report(file_path: str) -> dict:
    """Generate a deterministic, JSON-serializable data quality report."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError("Dataset file not found")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        raise ValueError("Dataset contains no columns") from None

    return quality_report_from_dataframe(df)


def quality_report_from_dataframe(df: pd.DataFrame) -> dict:
    """Quality report for an already-loaded DataFrame."""
    row_count, column_count = df.shape

    if column_count == 0:
        raise ValueError("Dataset contains no columns")

    missing_counts = df.isna().sum()
    total_cells = row_count * column_count
    missing_cells = int(missing_counts.sum())
    duplicate_rows = int(df.duplicated().sum())

    empty_columns = [
        str(column)
        for column in df.columns
        if df[column].isna().all()
    ]

    column_quality = {}

    for column in df.columns:
        missing = int(missing_counts[column])
        missing_percentage = (
            round(missing / row_count * 100, 2)
            if row_count
            else 0.0
        )

        column_quality[str(column)] = {
            "missing_values": missing,
            "missing_percentage": missing_percentage,
            "empty_column": bool(df[column].isna().all()),
        }

    warnings = []

    if row_count == 0:
        completeness_score = 0.0
        uniqueness_score = 0.0
        warnings.append("Dataset contains no data rows.")
    else:
        completeness_score = round(
            (total_cells - missing_cells) / total_cells * 100,
            2,
        )
        uniqueness_score = round(
            (row_count - duplicate_rows) / row_count * 100,
            2,
        )

    nonempty_column_count = column_count - len(empty_columns)
    nonempty_column_score = round(
        nonempty_column_count / column_count * 100,
        2,
    )

    quality_score = round(
        completeness_score * 0.5
        + uniqueness_score * 0.3
        + nonempty_column_score * 0.2,
        2,
    )

    if missing_cells:
        warnings.append(
            f"{missing_cells} missing value(s) found across the dataset."
        )

    if duplicate_rows:
        warnings.append(f"{duplicate_rows} duplicate row(s) found.")

    if empty_columns:
        warnings.append(
            f"{len(empty_columns)} column(s) contain no usable values."
        )

    recommendations = []

    if missing_cells:
        recommendations.append(
            "Investigate missing values and decide whether to fill, "
            "retain, or remove them."
        )

    if duplicate_rows:
        recommendations.append(
            "Review duplicate rows before deciding whether to remove them."
        )

    if empty_columns:
        recommendations.append(
            "Review empty columns and remove them if they are unnecessary."
        )

    if not warnings:
        recommendations.append(
            "No issues were detected by these basic quality checks."
        )

    return {
        "row_count": row_count,
        "column_count": column_count,
        "quality_score": quality_score,
        "score_components": {
            "completeness": completeness_score,
            "uniqueness": uniqueness_score,
            "nonempty_columns": nonempty_column_score,
        },
        "missing_cells": missing_cells,
        "duplicate_rows": duplicate_rows,
        "empty_columns": empty_columns,
        "column_quality": column_quality,
        "warnings": warnings,
        "recommendations": recommendations,
    }
