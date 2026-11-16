"""ML run registry: persistence, versioning, prediction, reproducibility."""
from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path

import joblib
import pandas as pd
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.app import metrics as app_metrics
from src.app.config import get_settings
from src.app.models.dataset import Dataset
from src.app.models.dataset_version import DatasetVersion
from src.app.models.ml_run import MLRun
from src.app.services.data_io import json_safe, load_dataframe, sha256_file
from src.app.services.ml.training import (
    ALGORITHMS,
    PRIMARY_METRIC,
    TrainingResult,
    coerce_features,
    environment_info,
    train_model,
)

logger = logging.getLogger(__name__)


class ModelArtifactError(RuntimeError):
    """The stored model file is missing, outside storage, or has been modified."""


def _default_model_name(task: str, target: str | None, algorithm: str) -> str:
    base = {"regression": "reg", "classification": "clf", "clustering": "cluster"}[task]
    suffix = re.sub(r"[^\w\-.]+", "_", target or "all")[:40]
    return f"{base}-{suffix}-{algorithm}"


def _latest_version_number(db: Session, dataset_id: int) -> int | None:
    return (
        db.query(func.max(DatasetVersion.version_number))
        .filter(DatasetVersion.dataset_id == dataset_id)
        .scalar()
    )


def _next_model_version(db: Session, dataset_id: int, model_name: str) -> int:
    current = (
        db.query(func.max(MLRun.version))
        .filter(MLRun.dataset_id == dataset_id, MLRun.model_name == model_name)
        .scalar()
    )
    return (current or 0) + 1


def run_to_dict(run: MLRun, include_details: bool = True) -> dict:
    data = {
        "id": run.id,
        "dataset_id": run.dataset_id,
        "dataset_version_number": run.dataset_version_number,
        "dataset_sha256": run.dataset_sha256,
        "model_name": run.model_name,
        "version": run.version,
        "task": run.task,
        "algorithm": run.algorithm,
        "target": run.target,
        "features": run.features,
        "config": run.config,
        "metrics": run.metrics,
        "primary_metric": run.primary_metric,
        "primary_metric_value": run.primary_metric_value,
        "status": run.status,
        "error": run.error,
        "duration_ms": run.duration_ms,
        "has_artifact": bool(run.artifact_path),
        "environment": run.environment,
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }
    if include_details:
        data["details"] = run.details
    return data


def train_and_register(
    db: Session, dataset: Dataset, request, df: pd.DataFrame | None = None,
    source_path: str | None = None, dataset_version_number: int | None = None,
) -> tuple[MLRun, TrainingResult]:
    """Train, persist the artifact, and record the run. Raises ValueError for
    invalid requests (nothing is recorded in that case)."""
    settings = get_settings()
    settings.ensure_dirs()
    path = source_path or dataset.file_path
    frame = df if df is not None else load_dataframe(path)

    result = train_model(frame, request)
    app_metrics.incr("ml_trainings")

    model_name = request.model_name or _default_model_name(
        result.task, result.target, result.algorithm)
    version = _next_model_version(db, dataset.id, model_name)

    folder = settings.models_dir / str(dataset.id)
    folder.mkdir(parents=True, exist_ok=True)
    artifact = folder / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', model_name)}_v{version}_{uuid.uuid4().hex[:8]}.joblib"
    joblib.dump(
        {
            "pipeline": result.pipeline,
            "task": result.task,
            "algorithm": result.algorithm,
            "target": result.target,
            "features": result.features,
            "numeric_features": result.numeric_features,
            "categorical_features": result.categorical_features,
            "classes": result.classes,
        },
        artifact,
        compress=3,
    )

    primary = PRIMARY_METRIC[result.task]
    config = {
        "task": request.task,
        "algorithm": request.algorithm,
        "target": request.target,
        "features_requested": request.features,
        "test_size": request.test_size,
        "cv_folds": request.cv_folds,
        "feature_selection": request.feature_selection.model_dump(),
        "n_clusters": request.n_clusters,
        "hyperparameters": result.effective_config["hyperparameters"],
        "random_state": result.effective_config["random_state"],
        "model_name": request.model_name,
    }
    details = dict(result.details)
    details["warnings"] = result.warnings
    details["numeric_features"] = result.numeric_features
    details["categorical_features"] = result.categorical_features
    details["classes"] = result.classes

    if dataset_version_number is None:
        dataset_version_number = _latest_version_number(db, dataset.id)

    run = MLRun(
        dataset_id=dataset.id,
        dataset_version_number=dataset_version_number,
        dataset_sha256=sha256_file(path),
        model_name=model_name,
        version=version,
        task=result.task,
        algorithm=result.algorithm,
        target=result.target,
        features=result.features,
        config=json_safe(config),
        metrics=result.metrics,
        primary_metric=primary,
        primary_metric_value=result.metrics.get(primary),
        details=json_safe(details),
        environment=environment_info(),
        status="completed",
        artifact_path=str(artifact),
        artifact_sha256=sha256_file(artifact),
        duration_ms=result.duration_ms,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    logger.info("model trained", extra={"run_id": run.id, "task": run.task,
                                        "algorithm": run.algorithm,
                                        "duration_ms": run.duration_ms})
    return run, result


def compare_train(db: Session, dataset: Dataset, request) -> list[MLRun]:
    """Train several algorithms with identical data/split settings."""
    runs = []
    errors = []
    df = load_dataframe(dataset.file_path)
    for algorithm in request.algorithms:
        single = _single_request(request, algorithm)
        try:
            run, _ = train_and_register(db, dataset, single, df=df)
            runs.append(run)
        except ValueError as exc:
            errors.append(f"{algorithm}: {exc}")
    if not runs:
        raise ValueError("; ".join(errors) or "No models could be trained")
    runs.sort(key=lambda r: (r.primary_metric_value is None,
                             -(r.primary_metric_value or 0)))
    return runs, errors


def _single_request(request, algorithm):
    from src.app.schemas.ml import TrainRequest

    data = request.model_dump(exclude={"algorithms"})
    data["algorithm"] = algorithm
    data["model_name"] = None
    return TrainRequest(**data)


# ------------------------------------------------------------------ model loading
def load_artifact(run: MLRun) -> dict:
    settings = get_settings()
    if not run.artifact_path:
        raise ModelArtifactError("This run has no stored model")
    path = Path(run.artifact_path).resolve()
    root = settings.models_dir.resolve()
    if root not in path.parents:
        raise ModelArtifactError("Model file is outside the model storage directory")
    if not path.is_file():
        raise ModelArtifactError("Model file not found on disk")
    if run.artifact_sha256 and sha256_file(path) != run.artifact_sha256:
        raise ModelArtifactError(
            "Model file checksum mismatch; the artifact was modified and will not be loaded")
    return joblib.load(path)


def predict(run: MLRun, records: list[dict]) -> dict:
    artifact = load_artifact(run)
    numeric = artifact["numeric_features"]
    categorical = artifact["categorical_features"]
    expected = set(numeric + categorical)

    frame = pd.DataFrame(records)
    provided = set(frame.columns)
    unknown = sorted(provided - expected)
    missing = sorted(expected - provided)
    X = coerce_features(frame, numeric, categorical)

    warnings = []
    if unknown:
        warnings.append(f"Ignored unknown field(s): {', '.join(unknown[:10])}")
    if missing:
        warnings.append(
            f"Missing field(s) filled with training-set defaults: {', '.join(missing[:10])}")
    for column in numeric:
        if column in frame.columns:
            raw = frame[column]
            bad = raw.notna() & pd.to_numeric(raw, errors="coerce").isna()
            if bad.any():
                warnings.append(f"Non-numeric value(s) in '{column}' treated as missing")

    pipeline = artifact["pipeline"]
    predictions = pipeline.predict(X)
    out = []
    probabilities = None
    if artifact["task"] == "classification" and hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X)
        classes = [str(c) for c in pipeline.classes_]

    for i, value in enumerate(predictions):
        item = {"prediction": json_safe(value.item() if hasattr(value, "item") else value)}
        if artifact["task"] == "clustering":
            item = {"cluster": int(value)}
        if probabilities is not None:
            item["probabilities"] = {c: round(float(p), 4)
                                     for c, p in zip(classes, probabilities[i])}
        out.append(item)
    app_metrics.incr("predictions", len(out))
    return {"run_id": run.id, "task": artifact["task"], "target": artifact["target"],
            "predictions": out, "warnings": warnings}


# ----------------------------------------------------------------- reproducibility
def reproduce(db: Session, run: MLRun) -> tuple[MLRun, dict]:
    """Retrain with the stored configuration on the exact recorded data."""
    from src.app.schemas.ml import TrainRequest

    dataset = db.query(Dataset).filter(Dataset.id == run.dataset_id).first()
    if dataset is None:
        raise ValueError("Dataset no longer exists")
    version = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.dataset_id == run.dataset_id,
                DatasetVersion.version_number == run.dataset_version_number)
        .first()
    )
    path = version.file_path if version else dataset.file_path
    if not Path(path).is_file():
        raise ValueError("The dataset file used by this run no longer exists")
    if run.dataset_sha256 and sha256_file(path) != run.dataset_sha256:
        raise ValueError(
            "The dataset file has changed since this run; exact reproduction is impossible")

    cfg = run.config
    request = TrainRequest(
        task=cfg["task"], algorithm=cfg["algorithm"], target=cfg.get("target"),
        features=cfg.get("features_requested"), test_size=cfg["test_size"],
        cv_folds=cfg["cv_folds"], feature_selection=cfg["feature_selection"],
        n_clusters=cfg.get("n_clusters"), hyperparameters=cfg.get("hyperparameters", {}),
        model_name=run.model_name, random_state=cfg["random_state"],
    )
    new_run, _ = train_and_register(
        db, dataset, request, source_path=path,
        dataset_version_number=run.dataset_version_number)
    differences = {}
    for key, old in (run.metrics or {}).items():
        new = (new_run.metrics or {}).get(key)
        if isinstance(old, (int, float)) and isinstance(new, (int, float)):
            differences[key] = {"original": old, "reproduced": new,
                                "abs_difference": abs(old - new)}
    identical = all(d["abs_difference"] < 1e-9 for d in differences.values())
    return new_run, {"metrics_identical": identical, "metric_comparison": differences}


def delete_run(db: Session, run: MLRun) -> None:
    artifact = run.artifact_path
    db.delete(run)
    db.commit()
    if artifact:
        Path(artifact).unlink(missing_ok=True)


def algorithms_catalog() -> dict:
    return {
        "algorithms": ALGORITHMS,
        "primary_metrics": PRIMARY_METRIC,
    }
