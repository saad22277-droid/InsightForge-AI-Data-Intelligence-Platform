import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.app.schemas.ml import TrainRequest
from src.app.services.ml.training import train_model


@pytest.fixture
def df():
    rng = np.random.default_rng(1)
    n = 300
    d = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                      "c": rng.choice(["u", "v", "w"], n)})
    d["y"] = 3 * d.a - 2 * d.b + (d.c == "u") * 2 + rng.normal(size=n) * 0.3
    d["label"] = np.where(d.y > d.y.median(), "hi", "lo")
    return d


def req(**kw):
    base = dict(task="regression", algorithm="random_forest", target="y")
    base.update(kw)
    return TrainRequest(**base)


def test_regression_metrics_are_real_and_beat_baseline(df):
    r = train_model(df, req())
    assert r.metrics["r2"] > 0.8
    assert r.metrics["baseline_r2"] < 0.1
    assert r.metrics["rmse"] >= r.metrics["mae"] > 0
    json.dumps(r.metrics, allow_nan=False)
    json.dumps(r.details, allow_nan=False)


@pytest.mark.parametrize("algo", ["linear_regression", "ridge", "random_forest", "gradient_boosting"])
def test_all_regressors_train(df, algo):
    assert train_model(df, req(algorithm=algo)).metrics["r2"] > 0.7


@pytest.mark.parametrize("algo", ["logistic_regression", "random_forest", "gradient_boosting"])
def test_classifiers_report_full_metrics(df, algo):
    r = train_model(df, req(task="classification", algorithm=algo, target="label"))
    for key in ("accuracy", "balanced_accuracy", "f1_weighted", "f1_macro",
                "baseline_accuracy", "roc_auc"):
        assert key in r.metrics
    assert r.metrics["accuracy"] > r.metrics["baseline_accuracy"]
    assert np.array(r.details["confusion_matrix"]).sum() == r.details["split"]["test_rows"]


def test_clustering_finds_structure():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"x": np.r_[rng.normal(0, .3, 100), rng.normal(5, .3, 100)],
                      "y": np.r_[rng.normal(0, .3, 100), rng.normal(5, .3, 100)]})
    r = train_model(d, TrainRequest(task="clustering", algorithm="kmeans"))
    assert r.metrics["n_clusters"] == 2 and r.metrics["silhouette"] > 0.8
    assert sum(p["size"] for p in r.details["cluster_profiles"]) == 200


def test_split_happens_before_fitting_preprocessors(df):
    r = train_model(df, req(test_size=0.25))
    scaler = r.pipeline.named_steps["prep"].named_transformers_["num"].named_steps["scaler"]
    assert scaler.n_samples_seen_ == r.details["split"]["train_rows"] < len(df)
    # scaler statistics come from training rows only, not the full column
    assert scaler.mean_[0] != pytest.approx(df["a"].mean(), abs=1e-12)


def test_target_cannot_be_a_feature(df):
    with pytest.raises(ValueError, match="leakage"):
        train_model(df, req(features=["a", "y"]))


def test_derived_columns_and_identifiers_are_excluded(df):
    d = df.copy()
    d["id"] = range(len(d))
    d["y_copy"] = d["y"] * 1.0001
    r = train_model(d, req())
    excluded = {e["column"]: e["reason"] for e in r.details["excluded_features"]}
    assert "identifier" in excluded["id"] and "leakage" in excluded["y_copy"]
    assert "id" not in r.features and "y_copy" not in r.features


def test_classification_leak_from_numeric_column_detected(df):
    r = train_model(df, req(task="classification", algorithm="random_forest", target="label"))
    assert "y" not in r.features  # y fully determines label (y > median)


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.assign(y="text"), "not numeric"),
    (lambda d: d.head(10), "usable rows"),
])
def test_regression_validation(df, mutate, match):
    with pytest.raises(ValueError, match=match):
        train_model(mutate(df), req())


def test_classification_validation(df):
    d = df.copy()
    d.loc[0, "label"] = "unique_class"
    with pytest.raises(ValueError, match="at least 2 rows"):
        train_model(d, req(task="classification", target="label"))
    d2 = df.assign(label="same")
    with pytest.raises(ValueError, match="two classes"):
        train_model(d2, req(task="classification", target="label"))


def test_missing_target_and_unknown_column(df):
    with pytest.raises(ValueError, match="required"):
        train_model(df, req(target=None))
    with pytest.raises(ValueError, match="not found"):
        train_model(df, req(target="ghost"))
    with pytest.raises(ValueError, match="not found"):
        train_model(df, req(features=["ghost"]))


def test_algorithm_must_match_task(df):
    with pytest.raises(ValueError, match="not available"):
        train_model(df, req(algorithm="logistic_regression"))


def test_hyperparameters_are_whitelisted_and_range_checked(df):
    with pytest.raises(ValueError, match="not supported"):
        train_model(df, req(hyperparameters={"n_jobs": 8}))
    with pytest.raises(ValueError, match="between"):
        train_model(df, req(hyperparameters={"n_estimators": 100000}))
    r = train_model(df, req(hyperparameters={"n_estimators": 20, "max_depth": 3}))
    assert r.effective_config["hyperparameters"] == {"n_estimators": 20, "max_depth": 3}


def test_same_seed_is_reproducible(df):
    a = train_model(df, req(random_state=5))
    b = train_model(df, req(random_state=5))
    assert a.metrics == b.metrics


def test_feature_selection_limits_features(df):
    r = train_model(df, req(algorithm="ridge", features=["a", "b", "c"],
                            feature_selection={"method": "k_best", "k": 2}))
    assert len(r.details["selected_encoded_features"]) == 2
    with pytest.raises(ValueError, match="clustering"):
        train_model(df, TrainRequest(task="clustering", algorithm="kmeans",
                                     feature_selection={"method": "k_best", "k": 2}))


def test_missing_values_are_imputed_inside_pipeline(df):
    d = df.copy()
    d.loc[d.sample(frac=0.1, random_state=0).index, "a"] = np.nan
    d.loc[d.sample(frac=0.1, random_state=1).index, "c"] = None
    assert train_model(d, req()).metrics["r2"] > 0.5


def test_request_schema_bounds():
    with pytest.raises(ValidationError):
        TrainRequest(task="regression", algorithm="ridge", target="y", test_size=0.9)
    with pytest.raises(ValidationError):
        TrainRequest(task="magic", algorithm="x")
    with pytest.raises(ValidationError):
        TrainRequest(task="regression", algorithm="ridge", target="y", model_name="../evil")
