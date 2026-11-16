from pathlib import Path

import pytest

from src.app.config import get_settings

TRAIN = {"task": "regression", "algorithm": "random_forest", "target": "units",
         "hyperparameters": {"n_estimators": 30}}


def train(client, did, **overrides):
    r = client.post(f"/datasets/{did}/ml/train", json={**TRAIN, **overrides})
    assert r.status_code == 200, r.text
    return r.json()


def test_train_persists_run_with_real_metrics_and_artifact(client, dataset_id):
    run = train(client, dataset_id)
    assert run["status"] == "completed" and run["version"] == 1
    assert run["metrics"]["r2"] > run["metrics"]["baseline_r2"]
    assert run["primary_metric"] == "r2" and run["primary_metric_value"] == run["metrics"]["r2"]
    assert run["environment"]["scikit_learn"] and run["dataset_sha256"] and run["config"]["random_state"] == 42
    assert Path(get_settings().models_dir / str(dataset_id)).is_dir()
    assert client.get(f"/ml/runs/{run['id']}").json()["details"]["split"]["train_rows"] > 0


def test_model_versions_increment_per_model_name(client, dataset_id):
    a = train(client, dataset_id, model_name="sales-model")
    b = train(client, dataset_id, model_name="sales-model")
    c = train(client, dataset_id, model_name="other-model")
    assert (a["version"], b["version"], c["version"]) == (1, 2, 1)
    runs = client.get(f"/datasets/{dataset_id}/ml/runs").json()
    assert len(runs) == 3 and runs[0]["id"] == c["id"]
    assert len(client.get(f"/datasets/{dataset_id}/ml/runs", params={"task": "clustering"}).json()) == 0


def test_predict_regression_and_input_warnings(client, dataset_id):
    run = train(client, dataset_id)
    r = client.post(f"/ml/runs/{run['id']}/predict",
                    json={"records": [{"sales": 1000, "price": 100, "region": "North", "bogus": 1}]})
    body = r.json()
    assert r.status_code == 200 and isinstance(body["predictions"][0]["prediction"], float)
    assert 0 < body["predictions"][0]["prediction"] < 5000
    assert any("unknown" in w.lower() for w in body["warnings"])
    assert any("missing" in w.lower() for w in body["warnings"])
    schema = client.get(f"/ml/runs/{run['id']}/schema").json()
    assert {f["name"] for f in schema["features"]} >= {"sales", "price"}
    assert client.post(f"/ml/runs/{run['id']}/predict", json={"records": []}).status_code == 422


def test_classification_predict_returns_probabilities(client, dataset_id):
    run = train(client, dataset_id, task="classification", algorithm="logistic_regression",
                target="category", hyperparameters={})
    r = client.post(f"/ml/runs/{run['id']}/predict",
                    json={"records": [{"units": 5, "price": 90, "region": "East"}]}).json()
    probs = r["predictions"][0]["probabilities"]
    assert abs(sum(probs.values()) - 1) < 1e-3 and r["predictions"][0]["prediction"] in probs


def test_clustering_assigns_clusters(client, dataset_id):
    run = train(client, dataset_id, task="clustering", algorithm="kmeans", target=None,
                n_clusters=3, hyperparameters={})
    assert run["metrics"]["n_clusters"] == 3 and run["primary_metric"] == "silhouette"
    r = client.post(f"/ml/runs/{run['id']}/predict", json={"records": [{"units": 5, "price": 90}]}).json()
    assert r["predictions"][0]["cluster"] in (0, 1, 2)


def test_invalid_training_requests_return_400_and_record_nothing(client, dataset_id):
    for body in [
        {**TRAIN, "target": "ghost"},
        {**TRAIN, "target": "region"},                      # text target for regression
        {**TRAIN, "features": ["price", "units"]},          # target as feature (leakage)
        {**TRAIN, "algorithm": "logistic_regression"},      # wrong task
        {**TRAIN, "hyperparameters": {"n_jobs": 9}},
    ]:
        assert client.post(f"/datasets/{dataset_id}/ml/train", json=body).status_code == 400, body
    assert client.post(f"/datasets/{dataset_id}/ml/train", json={**TRAIN, "test_size": 0.99}).status_code == 422
    assert client.get(f"/datasets/{dataset_id}/ml/runs").json() == []
    assert client.post("/datasets/999/ml/train", json=TRAIN).status_code == 404


def test_compare_ranks_models_and_experiments_endpoints(client, dataset_id):
    r = client.post(f"/datasets/{dataset_id}/ml/compare",
                    json={"task": "regression", "target": "sales",
                          "algorithms": ["linear_regression", "ridge", "random_forest"]}).json()
    scores = [x["primary_metric_value"] for x in r["runs"]]
    assert scores == sorted(scores, reverse=True) and r["best_run_id"] == r["runs"][0]["id"]
    ids = ",".join(str(x["id"]) for x in r["runs"][:2])
    cmp = client.get("/experiments/compare", params={"ids": ids}).json()
    assert cmp["task"] == "regression" and "r2" in cmp["best_by_metric"] and len(cmp["runs"]) == 2
    assert client.get("/experiments/compare", params={"ids": "1"}).status_code == 400
    assert client.get("/experiments/compare", params={"ids": "1,999"}).status_code == 404
    exps = client.get("/experiments").json()
    assert len(exps) == 3 and exps[0]["dataset_filename"] == "sales.csv"
    summary = client.get("/experiments/summary").json()
    assert summary["total_runs"] == 3 and summary["best_models"][0]["task"] == "regression"


def test_comparing_different_tasks_is_rejected(client, dataset_id):
    a = train(client, dataset_id)
    b = train(client, dataset_id, task="classification", algorithm="logistic_regression",
              target="category", hyperparameters={})
    assert client.get("/experiments/compare", params={"ids": f"{a['id']},{b['id']}"}).status_code == 400


def test_reproduce_gives_identical_metrics_even_after_cleaning(client, dataset_id):
    run = train(client, dataset_id)
    client.post(f"/datasets/{dataset_id}/cleaning/apply",
                json={"operations": [{"type": "fill_missing", "column": "age", "strategy": "median"}]})
    r = client.post(f"/ml/runs/{run['id']}/reproduce").json()
    assert r["metrics_identical"] is True and r["new_run"]["version"] == 2
    assert r["new_run"]["dataset_sha256"] == run["dataset_sha256"]


def test_tampered_artifact_is_not_loaded(client, dataset_id):
    run = train(client, dataset_id)
    artifact = next(get_settings().models_dir.rglob("*.joblib"))
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    r = client.post(f"/ml/runs/{run['id']}/predict", json={"records": [{"units": 1}]})
    assert r.status_code == 409 and "checksum" in r.json()["detail"]


def test_artifact_outside_storage_is_refused(client, dataset_id, env):
    from src.app.models.ml_run import MLRun
    run = train(client, dataset_id)
    outside = env["tmp"] / "evil.joblib"
    outside.write_bytes(b"x")
    with env["Session"]() as db:
        row = db.get(MLRun, run["id"])
        row.artifact_path = str(outside)
        db.commit()
    r = client.post(f"/ml/runs/{run['id']}/predict", json={"records": [{"units": 1}]})
    assert r.status_code == 409 and "outside" in r.json()["detail"]


def test_delete_run_removes_artifact(client, dataset_id):
    run = train(client, dataset_id)
    assert client.delete(f"/ml/runs/{run['id']}").status_code == 200
    assert not list(get_settings().models_dir.rglob("*.joblib"))
    assert client.get(f"/ml/runs/{run['id']}").status_code == 404


def test_catalog(client):
    cat = client.get("/ml/catalog").json()
    assert "kmeans" in cat["algorithms"]["clustering"]
