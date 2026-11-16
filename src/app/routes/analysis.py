import io
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import func
from sqlalchemy.orm import Session

from src.app.database import get_db
from src.app.models.dataset_version import DatasetVersion
from src.app.routes.common import get_dataset_or_404, load_df_or_http
from src.app.services import charts as chart_service
from src.app.services import eda as eda_service
from src.app.services.data_io import json_safe, sha256_file

router = APIRouter(prefix="/datasets", tags=["analysis"])


@router.get("/{dataset_id}/eda")
def get_eda(dataset_id: int, db: Session = Depends(get_db)):
    """Complete automated exploratory analysis (computed from the data)."""
    df = load_df_or_http(get_dataset_or_404(db, dataset_id))
    return eda_service.generate_eda(df)


@router.get("/{dataset_id}/charts")
def get_chart(
    dataset_id: int,
    chart_type: str = Query(...),
    x_column: str | None = Query(None, max_length=255),
    y_column: str | None = Query(None, max_length=255),
    group_by: str | None = Query(None, max_length=255),
    agg: str = Query("count", max_length=20),
    bins: int = Query(10, ge=1, le=100),
    columns: str | None = Query(None, max_length=2000, description="Comma-separated"),
    db: Session = Depends(get_db),
):
    df = load_df_or_http(get_dataset_or_404(db, dataset_id))
    try:
        return chart_service.generate_chart(
            df, chart_type, x_column, y_column, group_by, agg, bins,
            [c for c in columns.split(",") if c] if columns else None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@router.get("/{dataset_id}/metadata")
def get_metadata(dataset_id: int, db: Session = Depends(get_db)):
    dataset = get_dataset_or_404(db, dataset_id)
    path = Path(dataset.file_path)
    exists = path.is_file()
    version_count = (db.query(func.count(DatasetVersion.id))
                     .filter(DatasetVersion.dataset_id == dataset_id).scalar())
    info = {
        "id": dataset.id,
        "filename": dataset.filename,
        "status": dataset.status,
        "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
        "row_count": dataset.row_count,
        "column_count": dataset.column_count,
        "version_count": int(version_count or 0),
        "active_version": next(
            (v.version_number for v in db.query(DatasetVersion)
             .filter(DatasetVersion.dataset_id == dataset_id).all()
             if v.file_path == dataset.file_path), None),
        "file_exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": sha256_file(path) if exists else None,
    }
    if exists:
        df = load_df_or_http(dataset)
        info["columns"] = [
            {"name": c, "dtype": str(df[c].dtype), "missing": int(df[c].isna().sum()),
             "unique": int(df[c].nunique())} for c in df.columns]
    return json_safe(info)


@router.get("/{dataset_id}/exports/columns.csv")
def export_column_profile(dataset_id: int, db: Session = Depends(get_db)):
    """Per-column profile as CSV (counts, missing, statistics, outliers)."""
    dataset = get_dataset_or_404(db, dataset_id)
    df = load_df_or_http(dataset)
    eda = eda_service.generate_eda(df)
    rows = []
    for column in df.columns:
        row = {"column": column, "dtype": eda["overview"]["dtypes"][column],
               "missing": int(df[column].isna().sum()),
               "unique": int(df[column].nunique())}
        row.update({k: v for k, v in eda["numeric_statistics"].get(column, {}).items()
                    if k in ("mean", "std", "min", "median", "max", "skewness")})
        outlier = eda["outliers"]["columns"].get(column)
        row["outliers_iqr"] = outlier["outlier_count"] if outlier else None
        rows.append(row)
    buffer = io.StringIO()
    pd.DataFrame(rows).to_csv(buffer, index=False)
    buffer.seek(0)
    name = Path(dataset.filename).stem
    return StreamingResponse(
        iter([buffer.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}_column_profile.csv"'})


@router.get("/{dataset_id}/versions/{version_number}/download")
def download_version(dataset_id: int, version_number: int, db: Session = Depends(get_db)):
    dataset = get_dataset_or_404(db, dataset_id)
    version = (db.query(DatasetVersion)
               .filter(DatasetVersion.dataset_id == dataset_id,
                       DatasetVersion.version_number == version_number).first())
    if version is None:
        raise HTTPException(status_code=404, detail="Dataset version not found")
    path = Path(version.file_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Dataset version file not found")
    stem = Path(dataset.filename).stem
    return FileResponse(path, media_type="text/csv",
                        filename=f"{stem}_v{version_number}.csv")
