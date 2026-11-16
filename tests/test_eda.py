import json

import numpy as np
import pandas as pd

from src.app.services.eda import detect_datetime_columns, generate_eda


def test_eda_is_strict_json_and_complete(sales_df):
    eda = generate_eda(sales_df)
    json.dumps(eda, allow_nan=False)
    assert set(eda) == {"overview", "numeric_statistics", "categorical_statistics",
                        "missing", "duplicates", "distributions", "correlations",
                        "outliers", "column_insights", "quality"}
    assert eda["overview"]["row_count"] == 200


def test_missing_analysis_matches_pandas(sales_df):
    eda = generate_eda(sales_df)
    assert eda["missing"]["column_with_most_missing"] == "age"
    assert eda["missing"]["columns"][0]["missing"] == int(sales_df["age"].isna().sum())
    assert eda["missing"]["total_missing_cells"] == int(sales_df.isna().sum().sum())


def test_numeric_statistics_exact(sales_df):
    stats = generate_eda(sales_df)["numeric_statistics"]["units"]
    assert stats["mean"] == round(sales_df["units"].mean(), 4)
    assert stats["max"] == sales_df["units"].max()


def test_outlier_detection_finds_planted_outlier(sales_df):
    out = generate_eda(sales_df)["outliers"]["columns"]["sales"]
    assert out["outlier_count"] >= 1
    assert out["max_outlier"] == 90000.0


def test_correlation_top_pair_is_plausible(sales_df):
    pair = generate_eda(sales_df)["correlations"]["top_pairs"][0]
    assert {pair["column_a"], pair["column_b"]} <= set(sales_df.columns)
    assert -1 <= pair["correlation"] <= 1


def test_duplicates_detected():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    assert generate_eda(df)["duplicates"]["duplicate_rows"] == 1


def test_insights_flag_constant_identifier_and_placeholder():
    df = pd.DataFrame({
        "id": range(30), "const": ["k"] * 30,
        "txt": ["N/A", "ok"] * 15, "v": np.arange(30) % 7,
    })
    types = {(i["column"], i["type"]) for i in generate_eda(df)["column_insights"]}
    assert ("id", "identifier") in types
    assert ("const", "constant") in types
    assert ("txt", "placeholder_values") in types


def test_datetime_like_detection():
    df = pd.DataFrame({"d": pd.date_range("2024-01-01", periods=20).astype(str),
                       "t": list("abcde") * 4})
    assert detect_datetime_columns(df) == ["d"]


def test_single_numeric_column_has_no_correlation():
    eda = generate_eda(pd.DataFrame({"a": [1, 2, 3, 4]}))
    assert eda["correlations"]["top_pairs"] == []
