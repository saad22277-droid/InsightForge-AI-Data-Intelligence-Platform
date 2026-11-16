from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.app.database import get_db
from src.app.models.dataset import Dataset
from src.app.models.ml_run import MLRun
from src.app.services.ml import service as ml_service

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("")
def list_experiments(
    task: str | None = Query(None),
    dataset_id: int | None = Query(None, ge=1),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    """Experiment tracker: every training run across datasets (newest first)."""
    query = db.query(MLRun, Dataset.filename).join(Dataset, Dataset.id == MLRun.dataset_id)
    if task:
        query = query.filter(MLRun.task == task)
    if dataset_id:
        query = query.filter(MLRun.dataset_id == dataset_id)
    rows = query.order_by(MLRun.id.desc()).limit(limit).all()
    return [{**ml_service.run_to_dict(run, include_details=False), "dataset_filename": name}
            for run, name in rows]


@router.get("/summary")
def summary(db: Session = Depends(get_db)):
    runs = db.query(MLRun).all()
    best = {}
    for run in runs:
        if run.primary_metric_value is None:
            continue
        key = (run.dataset_id, run.task, run.target)
        if key not in best or run.primary_metric_value > best[key].primary_metric_value:
            best[key] = run
    return {
        "total_runs": len(runs),
        "by_task": {t: sum(1 for r in runs if r.task == t)
                    for t in ("regression", "classification", "clustering")},
        "best_models": [
            {"run_id": r.id, "dataset_id": r.dataset_id, "task": r.task, "target": r.target,
             "model_name": r.model_name, "version": r.version, "algorithm": r.algorithm,
             "primary_metric": r.primary_metric, "value": r.primary_metric_value}
            for r in best.values()],
    }


@router.get("/compare")
def compare(ids: str = Query(..., description="Comma-separated run ids (2-6)"),
            db: Session = Depends(get_db)):
    try:
        run_ids = [int(i) for i in ids.split(",") if i.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="ids must be integers") from None
    if not 2 <= len(set(run_ids)) <= 6:
        raise HTTPException(status_code=400, detail="Provide between 2 and 6 distinct run ids")
    runs = db.query(MLRun).filter(MLRun.id.in_(run_ids)).all()
    if len(runs) != len(set(run_ids)):
        raise HTTPException(status_code=404, detail="One or more runs were not found")
    tasks = {r.task for r in runs}
    if len(tasks) > 1:
        raise HTTPException(status_code=400, detail="Only runs of the same task can be compared")
    metric_names = sorted({k for r in runs for k, v in (r.metrics or {}).items()
                           if isinstance(v, (int, float))})
    lower_is_better = {"mae", "rmse", "davies_bouldin", "inertia", "baseline_mae"}
    winners = {}
    for name in metric_names:
        scored = [(r.metrics[name], r.id) for r in runs
                  if isinstance((r.metrics or {}).get(name), (int, float))]
        if len(scored) > 1:
            pick = min(scored) if name in lower_is_better else max(scored)
            winners[name] = pick[1]
    return {
        "task": tasks.pop(),
        "runs": [ml_service.run_to_dict(r, include_details=False) for r in
                 sorted(runs, key=lambda r: r.id)],
        "metrics": metric_names,
        "best_by_metric": winners,
    }
