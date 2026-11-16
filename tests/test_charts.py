import pytest

from src.app.services.charts import generate_chart


def test_histogram_counts_sum_to_n(sales_df):
    chart = generate_chart(sales_df, "histogram", "price", bins=8)
    assert sum(b["count"] for b in chart["bins"]) == chart["n"] == 200
    assert len(chart["bins"]) == 8


def test_bar_aggregated_values_match_pandas(sales_df):
    chart = generate_chart(sales_df, "bar", "region", "sales", agg="sum")
    expected = sales_df.groupby("region")["sales"].sum().max()
    assert chart["data"][0]["value"] == pytest.approx(expected)


def test_bar_counts_without_y(sales_df):
    chart = generate_chart(sales_df, "bar", "category")
    assert sum(d["value"] for d in chart["data"]) == 200


def test_scatter_and_groups(sales_df):
    plain = generate_chart(sales_df, "scatter", "units", "sales")
    assert plain["total_points"] == 200 and plain["correlation"] is not None
    grouped = generate_chart(sales_df, "scatter", "units", "sales", group_by="region")
    assert len(grouped["series"]) == 4


def test_box_stats(sales_df):
    box = generate_chart(sales_df, "box", "price", group_by="region")
    assert len(box["boxes"]) == 4
    b = box["boxes"][0]
    assert b["min"] <= b["q1"] <= b["median"] <= b["q3"] <= b["max"]


def test_line_numeric_x(sales_df):
    line = generate_chart(sales_df, "line", "units", "sales", agg="mean")
    xs = [p["x"] for p in line["data"]]
    assert xs == sorted(xs)


def test_heatmap_is_symmetric_with_unit_diagonal(sales_df):
    hm = generate_chart(sales_df, "heatmap")
    m = hm["matrix"]
    assert all(m[i][i] == pytest.approx(1.0) for i in range(len(m)))
    assert m[0][1] == m[1][0]


def test_missing_chart(sales_df):
    data = generate_chart(sales_df, "missing")["data"]
    assert data[0]["column"] == "age" and data[0]["missing"] == 20


@pytest.mark.parametrize("kwargs", [
    {"chart_type": "pie"},
    {"chart_type": "histogram"},
    {"chart_type": "histogram", "x_column": "region"},
    {"chart_type": "scatter", "x_column": "units"},
    {"chart_type": "bar", "x_column": "ghost"},
    {"chart_type": "bar", "x_column": "region", "agg": "median_of_medians"},
])
def test_invalid_requests_raise(sales_df, kwargs):
    with pytest.raises(ValueError):
        generate_chart(sales_df, **kwargs)
