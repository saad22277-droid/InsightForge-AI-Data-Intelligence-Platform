import logging
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.app import metrics
from src.app.config import get_settings
from src.app.database import get_db
from src.app.models.cleaning_run import CleaningRun
from src.app.models.dataset_version import DatasetVersion
from src.app.routes.common import get_dataset_or_404, load_df_or_http
from src.app.schemas.cleaning import CleaningApplyRequest, CleaningRequest
from src.app.services import cleaning as cleaning_service
from src.app.services.data_io import clear_cache, json_safe

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/datasets", tags=["cleaning"])


@router.get("/{dataset_id}/cleaning/recommendations")
def recommendations(dataset_id: int, db: Session = Depends(get_db)):
    df = load_df_or_http(get_dataset_or_404(db, dataset_id))
    return cleaning_service.analyze_cleaning(df)


@router.post("/{dataset_id}/cleaning/preview")
def preview(dataset_id: int, body: CleaningRequest, db: Session = Depends(get_db)):
    """Dry run: shows before/after comparison. Nothing is saved."""
    df = load_df_or_http(get_dataset_or_404(db, dataset_id))
    try:
        cleaned, comparison = cleaning_service.preview_cleaning(df, body.operations)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    comparison["sample_after"] = json_safe(
        cleaned.head(10).astype(object).where(cleaned.head(10).notna(), None).to_dict("records"))
    comparison["sample_before"] = json_safe(
        df.head(10).astype(object).where(df.head(10).notna(), None).to_dict("records"))
    return comparison


@router.post("/{dataset_id}/cleaning/apply")
def apply(dataset_id: int, body: CleaningApplyRequest, db: Session = Depends(get_db)):
    """Apply operations and store the result as a NEW dataset version.

    The previous versions (including the original upload) are never modified.
    """
    dataset = get_dataset_or_404(db, dataset_id)
    df = load_df_or_http(dataset)
    try:
        cleaned, comparison = cleaning_service.preview_cleaning(df, body.operations)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    if len(cleaned) == 0:
        raise HTTPException(status_code=400, detail="Cleaning would remove every row")

    current = (db.query(func.max(DatasetVersion.version_number))
               .filter(DatasetVersion.dataset_id == dataset_id).scalar()) or 0
    next_version = current + 1
    settings = get_settings()
    settings.ensure_dirs()
    stem = Path(dataset.filename).stem[:60]
    path = settings.raw_dir / f"{uuid4()}_{stem}_cleaned_v{next_version}.csv"
    try:
        cleaned.to_csv(path, index=False)
        version = DatasetVersion(
            dataset_id=dataset_id, version_number=next_version, file_path=str(path),
            row_count=len(cleaned), column_count=len(cleaned.columns))
        db.add(version)
        dataset.file_path = str(path)
        dataset.row_count = len(cleaned)
        dataset.column_count = len(cleaned.columns)
        db.add(CleaningRun(
            dataset_id=dataset_id, source_version=current or None,
            result_version=next_version,
            operations=[op.model_dump() for op in body.operations],
            summary=json_safe({"delta": comparison["delta"],
                               "operations": comparison["operations"],
                               "note": body.note})))
        db.commit()
    except Exception:
        db.rollback()
        path.unlink(missing_ok=True)
        logger.exception("cleaning apply failed")
        raise HTTPException(status_code=500, detail="Failed to save cleaned dataset") from None

    clear_cache()
    metrics.incr("cleaning_applied")
    return {"message": "Cleaning applied", "new_version": next_version,
            "previous_version": current or None, "comparison": comparison}


@router.get("/{dataset_id}/cleaning/history")
def history(dataset_id: int, db: Session = Depends(get_db)):
    get_dataset_or_404(db, dataset_id)
    runs = (db.query(CleaningRun).filter(CleaningRun.dataset_id == dataset_id)
            .order_by(CleaningRun.id.desc()).all())
    return [{"id": r.id, "source_version": r.source_version,
             "result_version": r.result_version, "operations": r.operations,
             "summary": r.summary,
             "created_at": r.created_at.isoformat() if r.created_at else None}
            for r in runs]


@router.post("/{dataset_id}/versions/{version_number}/activate")
def activate_version(dataset_id: int, version_number: int, db: Session = Depends(get_db)):
    """Make an earlier version the active one (rollback). No data is deleted."""
    dataset = get_dataset_or_404(db, dataset_id)
    version = (db.query(DatasetVersion)
               .filter(DatasetVersion.dataset_id == dataset_id,
                       DatasetVersion.version_number == version_number).first())
    if version is None:
        raise HTTPException(status_code=404, detail="Dataset version not found")
    if not Path(version.file_path).is_file():
        raise HTTPException(status_code=404, detail="Dataset version file not found")
    dataset.file_path = version.file_path
    dataset.row_count = version.row_count
    dataset.column_count = version.column_count
    db.commit()
    clear_cache()
    return {"message": f"Version {version_number} is now active",
            "active_version": version_number}
