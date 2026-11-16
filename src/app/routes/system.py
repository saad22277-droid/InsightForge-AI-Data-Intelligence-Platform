import os

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.app import metrics
from src.app.config import get_settings
from src.app.database import get_db
from src.app.models.chat import ChatMessage
from src.app.models.dataset import Dataset
from src.app.models.ml_run import MLRun
from src.app.models.report import Report

router = APIRouter(tags=["system"])


def _dir_writable(path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return os.access(path, os.W_OK)
    except OSError:
        return False


@router.get("/health/ready")
def readiness(db: Session = Depends(get_db)):
    """Readiness: database reachable and storage directories writable."""
    settings = get_settings()
    checks = {"database": False,
              "storage_raw": _dir_writable(settings.raw_dir),
              "storage_models": _dir_writable(settings.models_dir),
              "storage_reports": _dir_writable(settings.reports_dir)}
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception:  # noqa: BLE001
        pass
    ready = all(checks.values())
    return JSONResponse(status_code=200 if ready else 503,
                        content={"status": "ready" if ready else "degraded", "checks": checks,
                                 "ai_configured": settings.ai_configured})


@router.get("/metrics")
def operational_metrics():
    return metrics.snapshot()


@router.get("/system/status")
def system_status(db: Session = Depends(get_db)):
    settings = get_settings()
    return {
        "settings": settings.public_dict(),
        "counts": {"datasets": db.query(Dataset).count(),
                   "model_runs": db.query(MLRun).count(),
                   "reports": db.query(Report).count(),
                   "chat_messages": db.query(ChatMessage).count()},
        "metrics": metrics.snapshot(),
    }
