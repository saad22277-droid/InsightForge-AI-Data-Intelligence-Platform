import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.app.schemas.cleaning import CleaningRequest
from src.app.services.cleaning import analyze_cleaning, apply_operations, preview_cleaning


def ops(*items):
    return CleaningRequest(operations=list(items)).operations


@pytest.fixture
def messy():
    return pd.DataFrame({
        "Customer Age": [25, 30, -5, 41, None, 30, 30],
        "Price ($)": ["$1,200", "300", "N/A", "450", "500", "300", "300"],
        "Region": [" north", "North", "south ", "SOUTH", "north", "North", "North"],
        "score": [1.0, 2.0, 3.0, 4.0, 100.0, 2.0, 2.0],
    })


def test_original_is_never_mutated(messy):
    before = messy.copy(deep=True)
    apply_operations(messy, ops({"type": "drop_duplicates"},
                                {"type": "fill_missing", "column": "Customer Age", "strategy": "median"}))
    pd.testing.assert_frame_equal(messy, before)


def test_drop_duplicates_counts(messy):
    out, reports = apply_operations(messy, ops({"type": "drop_duplicates"}))
    assert reports[0]["rows_affected"] == 2 and len(out) == 5


def test_fill_missing_strategies(messy):
    out, _ = apply_operations(messy, ops(
        {"type": "fill_missing", "column": "Customer Age", "strategy": "mean"}))
    assert out["Customer Age"].isna().sum() == 0
    out, _ = apply_operations(messy, ops(
        {"type": "fill_missing", "column": "Customer Age", "strategy": "constant", "value": "0"}))
    assert (out["Customer Age"] == 0).sum() == 1


def test_fill_missing_rejects_bad_input(messy):
    with pytest.raises(ValueError, match="not numeric"):
        apply_operations(messy, ops(
            {"type": "fill_missing", "column": "Region", "strategy": "mean"}))
    with pytest.raises(ValueError, match="Column not found"):
        apply_operations(messy, ops(
            {"type": "fill_missing", "column": "nope", "strategy": "mean"}))


def test_replace_invalid_then_convert_numeric(messy):
    out, _ = apply_operations(messy, ops(
        {"type": "replace_invalid", "column": "Price ($)"},
        {"type": "convert_dtype", "column": "Price ($)", "dtype": "numeric"}))
    assert pd.api.types.is_numeric_dtype(out["Price ($)"])
    assert out["Price ($)"].iloc[0] == 1200 and out["Price ($)"].isna().sum() == 1


def test_invalid_to_missing_negative(messy):
    out, rep = apply_operations(messy, ops(
        {"type": "invalid_to_missing", "column": "Customer Age", "minimum": 0}))
    assert rep[0]["rows_affected"] == 1 and out["Customer Age"].min() == 25


def test_outlier_actions(messy):
    clip, _ = apply_operations(messy, ops({"type": "handle_outliers", "column": "score", "action": "clip"}))
    assert clip["score"].max() < 100
    rem, _ = apply_operations(messy, ops({"type": "handle_outliers", "column": "score", "action": "remove"}))
    assert len(rem) == len(messy) - 1
    miss, _ = apply_operations(messy, ops({"type": "handle_outliers", "column": "score", "action": "set_missing"}))
    assert miss["score"].isna().sum() == 1


def test_standardize_text_and_column_names(messy):
    out, _ = apply_operations(messy, ops(
        {"type": "standardize_text", "column": "Region", "lowercase": True},
        {"type": "normalize_column_names"}))
    assert sorted(out["region"].unique()) == ["north", "south"]
    assert "customer_age" in out.columns and "price" in out.columns


def test_scale_column(messy):
    out, _ = apply_operations(messy, ops({"type": "scale_column", "column": "score", "method": "minmax"}))
    assert out["score"].min() == 0 and out["score"].max() == 1


def test_cannot_drop_every_column(messy):
    with pytest.raises(ValueError, match="every column"):
        apply_operations(messy, ops({"type": "drop_columns", "columns": list(messy.columns)}))


def test_schema_rejects_unknown_operation_and_bad_bounds():
    with pytest.raises(ValidationError):
        CleaningRequest(operations=[{"type": "exec", "code": "import os"}])
    with pytest.raises(ValidationError):
        CleaningRequest(operations=[])
    with pytest.raises(ValidationError):
        CleaningRequest(operations=[{"type": "invalid_to_missing", "column": "a"}])


def test_recommendations_are_applicable(messy):
    recs = analyze_cleaning(messy)["recommendations"]
    assert recs and all("operation" in r and r["reason"] for r in recs)
    categories = {r["category"] for r in recs}
    assert {"duplicates", "invalid_values", "standardization"} <= categories
    operations = CleaningRequest(operations=[r["operation"] for r in recs]).operations
    cleaned, comparison = preview_cleaning(messy, operations)
    assert comparison["delta"]["duplicate_rows"] < 0
    assert comparison["after"]["missing_cells"] >= 0
    assert len(cleaned.columns) == 4


def test_preview_warns_when_rows_removed():
    df = pd.DataFrame({"a": [None] * 5 + [1]})
    _, cmp = preview_cleaning(df, ops({"type": "drop_missing_rows"}))
    assert any("half" in w for w in cmp["warnings"])
