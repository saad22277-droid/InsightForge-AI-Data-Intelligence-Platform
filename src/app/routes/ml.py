import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.app.database import get_db
from src.app.models.ml_run import MLRun
from src.app.routes.common import get_dataset_or_404, load_df_or_http
from src.app.schemas.ml import CompareRequest, PredictRequest, TrainRequest
from src.app.services.ml import service as ml_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["machine-learning"])


def _get_run(db: Session, run_id: int) -> MLRun:
    run = db.query(MLRun).filter(MLRun.id == run_id).first()
    if run is None:
        raise HTTPException(status_code=404, detail="Model run not found")
    return run


@router.get("/ml/catalog")
def catalog():
    return ml_service.algorithms_catalog()


@router.post("/datasets/{dataset_id}/ml/train")
def train(dataset_id: int, body: TrainRequest, db: Session = Depends(get_db)):
    dataset = get_dataset_or_404(db, dataset_id)
    load_df_or_http(dataset)
    try:
        run, result = ml_service.train_and_register(db, dataset, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except Exception:
        logger.exception("training failed")
        raise HTTPException(status_code=500, detail="Model training failed") from None
    return {**ml_service.run_to_dict(run), "warnings": result.warnings}


@router.post("/datasets/{dataset_id}/ml/compare")
def compare(dataset_id: int, body: CompareRequest, db: Session = Depends(get_db)):
    """Train several algorithms with identical settings; ranked by primary metric."""
    dataset = get_dataset_or_404(db, dataset_id)
    load_df_or_http(dataset)
    try:
        runs, errors = ml_service.compare_train(db, dataset, body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return {"runs": [ml_service.run_to_dict(r, include_details=False) for r in runs],
            "errors": errors,
            "best_run_id": runs[0].id if runs else None}


@router.get("/datasets/{dataset_id}/ml/runs")
def list_runs(dataset_id: int, task: str | None = Query(None), db: Session = Depends(get_db)):
    get_dataset_or_404(db, dataset_id)
    query = db.query(MLRun).filter(MLRun.dataset_id == dataset_id)
    if task:
        query = query.filter(MLRun.task == task)
    return [ml_service.run_to_dict(r, include_details=False)
            for r in query.order_by(MLRun.id.desc()).all()]


@router.get("/ml/runs/{run_id}")
def get_run(run_id: int, db: Session = Depends(get_db)):
    return ml_service.run_to_dict(_get_run(db, run_id))


@router.delete("/ml/runs/{run_id}")
def delete_run(run_id: int, db: Session = Depends(get_db)):
    run = _get_run(db, run_id)
    ml_service.delete_run(db, run)
    return {"message": "Model run deleted", "id": run_id}


@router.get("/ml/runs/{run_id}/schema")
def run_schema(run_id: int, db: Session = Depends(get_db)):
    run = _get_run(db, run_id)
    details = run.details or {}
    return {"run_id": run.id, "task": run.task, "target": run.target,
            "classes": details.get("classes"), "features": details.get("feature_schema", [])}


@router.post("/ml/runs/{run_id}/predict")
def predict(run_id: int, body: PredictRequest, db: Session = Depends(get_db)):
    run = _get_run(db, run_id)
    try:
        return ml_service.predict(run, body.records)
    except ml_service.ModelArtifactError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@router.post("/ml/runs/{run_id}/reproduce")
def reproduce(run_id: int, db: Session = Depends(get_db)):
    run = _get_run(db, run_id)
    try:
        new_run, comparison = ml_service.reproduce(db, run)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    return {"new_run": ml_service.run_to_dict(new_run, include_details=False), **comparison}
