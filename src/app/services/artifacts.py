"""Locate on-disk artifacts (models, reports) belonging to datasets so they can be
removed together with the dataset. Never raises: cleanup must not block deletion."""
from __future__ import annotations

import logging
from pathlib import Path

from sqlalchemy.orm import Session

from src.app.models.ml_run import MLRun
from src.app.models.report import Report

logger = logging.getLogger(__name__)


def collect_artifact_paths(db: Session, dataset_ids: list[int]) -> set[Path]:
    paths: set[Path] = set()
    if not dataset_ids:
        return paths
    try:
        for run in db.query(MLRun).filter(MLRun.dataset_id.in_(dataset_ids)).all():
            if run.artifact_path:
                paths.add(Path(run.artifact_path))
        for report in db.query(Report).filter(Report.dataset_id.in_(dataset_ids)).all():
            if report.file_path:
                paths.add(Path(report.file_path))
    except Exception:  # noqa: BLE001
        logger.warning("could not collect artifact paths for cleanup")
    return {p for p in paths if isinstance(p, Path)}
