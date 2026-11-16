import hashlib
from pathlib import Path

import pytest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ uploads/limits
def test_upload_rejects_oversized_and_non_csv(client):
    big = b"a,b\n" + b"1,2\n" * 400_000  # > 1 MB limit set by the env fixture
    r = client.post("/datasets/upload", files={"file": ("big.csv", big, "text/csv")})
    assert r.status_code == 413 and "too large" in r.json()["detail"]
    r = client.post("/datasets/upload", files={"file": ("x.exe", b"MZ", "application/octet-stream")})
    assert r.status_code == 400
    r = client.post("/datasets/upload", files={"file": ("e.csv", b"   ", "text/csv")})
    assert r.status_code == 400


def test_upload_filename_cannot_escape_storage(client, env, sales_csv_bytes):
    r = client.post("/datasets/upload",
                    files={"file": ("../../evil.csv", sales_csv_bytes, "text/csv")})
    assert r.status_code == 200
    stored = Path(r.json()["file_path"]).resolve()
    assert env["settings"].raw_dir.resolve() in stored.parents


def test_metadata_and_exports(client, dataset_id):
    meta = client.get(f"/datasets/{dataset_id}/metadata").json()
    assert meta["row_count"] == 200 and meta["version_count"] == 1 and len(meta["sha256"]) == 64
    assert meta["active_version"] == 1
    assert {c["name"] for c in meta["columns"]} >= {"region", "sales"}
    csv = client.get(f"/datasets/{dataset_id}/exports/columns.csv")
    assert csv.status_code == 200 and csv.text.startswith("column,dtype,missing")
    assert "sales" in csv.text
    assert client.get(f"/datasets/{dataset_id}/versions/1/download").status_code == 200
    assert client.get(f"/datasets/{dataset_id}/versions/9/download").status_code == 404
    assert client.get("/datasets/999/metadata").status_code == 404


# --------------------------------------------------------------------------- EDA
def test_eda_endpoint_returns_real_values(client, dataset_id, sales_df):
    eda = client.get(f"/datasets/{dataset_id}/eda").json()
    assert eda["overview"]["row_count"] == 200
    assert eda["missing"]["column_with_most_missing"] == "age"
    assert eda["numeric_statistics"]["units"]["max"] == int(sales_df["units"].max())
    assert client.get("/datasets/999/eda").status_code == 404


def test_charts_endpoint(client, dataset_id):
    r = client.get(f"/datasets/{dataset_id}/charts",
                   params={"chart_type": "bar", "x_column": "region", "y_column": "sales", "agg": "sum"})
    assert r.status_code == 200 and len(r.json()["data"]) == 4
    r = client.get(f"/datasets/{dataset_id}/charts", params={"chart_type": "heatmap"})
    assert r.status_code == 200 and r.json()["columns"]
    r = client.get(f"/datasets/{dataset_id}/charts", params={"chart_type": "histogram", "x_column": "ghost"})
    assert r.status_code == 400
    r = client.get(f"/datasets/{dataset_id}/charts", params={"chart_type": "histogram", "x_column": "price", "bins": 500})
    assert r.status_code == 422


# ----------------------------------------------------------------------- cleaning
def test_cleaning_workflow_preserves_original_and_creates_version(client, dataset_id, env):
    from src.app.models.dataset import Dataset
    with env["Session"]() as db:
        original_path = db.get(Dataset, dataset_id).file_path
    original_hash = sha(original_path)

    recs = client.get(f"/datasets/{dataset_id}/cleaning/recommendations").json()
    assert recs["recommendation_count"] > 0
    fill = next(r for r in recs["recommendations"] if r["operation"]["type"] == "fill_missing")
    ops = [fill["operation"]]

    preview = client.post(f"/datasets/{dataset_id}/cleaning/preview", json={"operations": ops}).json()
    assert preview["delta"]["missing_cells"] == -20
    assert len(preview["sample_after"]) == 10
    # preview must not persist anything
    assert client.get(f"/datasets/{dataset_id}/versions").json()["versions"].__len__() == 1

    applied = client.post(f"/datasets/{dataset_id}/cleaning/apply",
                          json={"operations": ops, "note": "fill ages"})
    assert applied.status_code == 200 and applied.json()["new_version"] == 2

    assert sha(original_path) == original_hash                      # original untouched
    assert client.get(f"/datasets/{dataset_id}/versions").json()["versions"].__len__() == 2
    assert client.get(f"/datasets/{dataset_id}/metadata").json()["active_version"] == 2
    eda = client.get(f"/datasets/{dataset_id}/eda").json()           # analyses use the new version
    assert eda["missing"]["total_missing_cells"] == 0
    history = client.get(f"/datasets/{dataset_id}/cleaning/history").json()
    assert history[0]["result_version"] == 2 and history[0]["summary"]["note"] == "fill ages"

    # roll back to the original without deleting anything
    assert client.post(f"/datasets/{dataset_id}/versions/1/activate").status_code == 200
    assert client.get(f"/datasets/{dataset_id}/eda").json()["missing"]["total_missing_cells"] == 20
    assert client.post(f"/datasets/{dataset_id}/versions/7/activate").status_code == 404


def test_cleaning_validation_errors(client, dataset_id):
    r = client.post(f"/datasets/{dataset_id}/cleaning/preview",
                    json={"operations": [{"type": "fill_missing", "column": "ghost", "strategy": "mean"}]})
    assert r.status_code == 400 and "Column not found" in r.json()["detail"]
    r = client.post(f"/datasets/{dataset_id}/cleaning/apply",
                    json={"operations": [{"type": "run_shell", "cmd": "rm -rf /"}]})
    assert r.status_code == 422
    r = client.post(f"/datasets/{dataset_id}/cleaning/apply", json={"operations": []})
    assert r.status_code == 422
    r = client.post(f"/datasets/{dataset_id}/cleaning/apply",
                    json={"operations": [{"type": "drop_missing_rows", "columns": ["region"]},
                                         {"type": "handle_outliers", "column": "units", "method": "iqr",
                                          "action": "remove", "threshold": 0.0001}]})
    assert r.status_code in (200, 400)


def test_cleaning_that_removes_everything_is_refused(client, dataset_id):
    r = client.post(f"/datasets/{dataset_id}/cleaning/apply",
                    json={"operations": [{"type": "invalid_to_missing", "column": "units", "minimum": 10_000},
                                         {"type": "drop_missing_rows"}]})
    assert r.status_code == 400 and "every row" in r.json()["detail"]
    assert len(client.get(f"/datasets/{dataset_id}/versions").json()["versions"]) == 1


# ------------------------------------------------------------------ deletion
def test_deleting_dataset_removes_models_reports_and_files(client, dataset_id, use_llm):
    run = client.post(f"/datasets/{dataset_id}/ml/train",
                      json={"task": "regression", "algorithm": "ridge", "target": "sales"}).json()
    report = client.post(f"/datasets/{dataset_id}/reports", json={}).json()
    from src.app.config import get_settings
    reports_dir = get_settings().reports_dir
    models = list(get_settings().models_dir.rglob("*.joblib"))
    assert models and list(reports_dir.glob("*.pdf"))
    assert client.delete(f"/datasets/{dataset_id}").status_code == 200
    assert not list(get_settings().models_dir.rglob("*.joblib"))
    assert not list(reports_dir.glob("*.pdf"))
    assert client.get(f"/ml/runs/{run['id']}").status_code == 404
    assert client.get(f"/reports/{report['id']}/download").status_code == 404
