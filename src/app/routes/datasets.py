from src.app.schemas.dataset import DatasetValidationRequest
from src.app.services.validation import validate_dataset



import pandas as pd

from uuid import uuid4
from pathlib import Path

from src.app.database import SessionLocal
from fastapi.responses import FileResponse
from src.app.models.dataset import Dataset
from src.app.models.dataset_version import DatasetVersion

from src.app.schemas.dataset import DatasetResponse
from src.app.schemas.dataset import DatasetQueryRequest

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from src.app.services.query import query_dataset
from src.app.services.summary import generate_summary
from src.app.config import get_settings
from src.app.services.artifacts import collect_artifact_paths
from src.app.services.profiling import generate_profile
from src.app.services.quality import generate_quality_report
from src.app.services.visualization import generate_chart_data


router = APIRouter(
    prefix="/datasets",
    tags=["datasets"],
)

UPLOAD_DIR = Path("data/raw")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PROFILE_DIR = Path("data/profiles")
PROFILE_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.lower() != ".csv":
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported",
        )

    unique_name = f"{uuid4()}_{Path(file.filename).name}"
    file_path = UPLOAD_DIR / unique_name
    profile_path = PROFILE_DIR / f"{Path(unique_name).stem}_profile.html"

    content = await file.read()

    if len(content) > get_settings().max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Maximum upload size is {get_settings().max_upload_mb} MB",
        )

    if not content.strip():
        raise HTTPException(
            status_code=400,
            detail="Uploaded CSV file is empty",
        )

    try:
        file_path.write_bytes(content)
        df = pd.read_csv(file_path)

        if len(df.columns) == 0:
            raise HTTPException(
                status_code=400,
                detail="CSV file contains no columns",
            )

        generate_profile(str(file_path), str(profile_path))

        db = SessionLocal()

        try:
            dataset = Dataset(
                filename=Path(file.filename).name,
                file_path=str(file_path),
                profile_path=str(profile_path),
                row_count=len(df),
                column_count=len(df.columns),
                status="uploaded",
            )

            db.add(dataset)
            db.flush()

            dataset_version = DatasetVersion(
                dataset_id=dataset.id,
                version_number=1,
                file_path=str(file_path),
                row_count=len(df),
                column_count=len(df.columns),
            )

            db.add(dataset_version)
            db.commit()
            db.refresh(dataset)


            return {
                "id": dataset.id,
                "filename": dataset.filename,
                "content_type": file.content_type,
                "file_path": str(file_path),
                "profile_path": str(profile_path),
                "row_count": len(df),
                "column_count": len(df.columns),
            }
        finally:
            db.close()

    except HTTPException:
        raise
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Invalid CSV file",
        )
    except Exception:
        file_path.unlink(missing_ok=True)
        profile_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to process uploaded dataset",
        )

@router.post("/{dataset_id}/versions/upload")
async def upload_dataset_version(
    dataset_id: int,
    file: UploadFile = File(...),
):
    if not file.filename or Path(file.filename).suffix.lower() != ".csv":
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported",
        )

    unique_name = f"{uuid4()}_{Path(file.filename).name}"
    file_path = UPLOAD_DIR / unique_name
    profile_path = PROFILE_DIR / f"{Path(unique_name).stem}_profile.html"

    content = await file.read()

    if len(content) > get_settings().max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Maximum upload size is {get_settings().max_upload_mb} MB",
        )

    if not content.strip():
        raise HTTPException(
            status_code=400,
            detail="Uploaded CSV file is empty",
        )

    try:
        file_path.write_bytes(content)
        df = pd.read_csv(file_path)

        if len(df.columns) == 0:
            raise HTTPException(
                status_code=400,
                detail="CSV file contains no columns",
            )

        generate_profile(str(file_path), str(profile_path))

        db = SessionLocal()

        try:
            dataset = (
                db.query(Dataset)
                .filter(Dataset.id == dataset_id)
                .first()
            )

            if dataset is None:
                raise HTTPException(
                    status_code=404,
                    detail="Dataset not found",
                )

            latest_version = (
                db.query(DatasetVersion)
                .filter(DatasetVersion.dataset_id == dataset_id)
                .order_by(DatasetVersion.version_number.desc())
                .first()
            )

            next_version = (
                latest_version.version_number + 1
                if latest_version
                else 1
            )

            dataset_version = DatasetVersion(
                dataset_id=dataset_id,
                version_number=next_version,
                file_path=str(file_path),
                row_count=len(df),
                column_count=len(df.columns),
            )

            dataset.filename = Path(file.filename).name
            dataset.file_path = str(file_path)
            dataset.profile_path = str(profile_path)
            dataset.row_count = len(df)
            dataset.column_count = len(df.columns)

            db.add(dataset_version)
            db.commit()
            db.refresh(dataset_version)

            return {
                "dataset_id": dataset_id,
                "version_id": dataset_version.id,
                "version_number": next_version,
                "filename": dataset.filename,
                "file_path": str(file_path),
                "profile_path": str(profile_path),
                "row_count": len(df),
                "column_count": len(df.columns),
            }

        finally:
            db.close()

    except HTTPException:
        file_path.unlink(missing_ok=True)
        profile_path.unlink(missing_ok=True)
        raise
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
        file_path.unlink(missing_ok=True)
        profile_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Invalid CSV file",
        )
    except Exception:
        file_path.unlink(missing_ok=True)
        profile_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to process dataset version",
        )

@router.get("/", response_model=list[DatasetResponse])
def list_datasets():
    db = SessionLocal()

    try:
        datasets = db.query(Dataset).order_by(Dataset.id).all()

        return [
            {
                "id": dataset.id,
                "filename": dataset.filename,
                "profile_path": dataset.profile_path,
                "row_count": dataset.row_count,
                "column_count": dataset.column_count,
                "status": dataset.status,
            }
            for dataset in datasets
        ]

    finally:
        db.close()

@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: int):
    db = SessionLocal()

    try:
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()

        if dataset is None:
            raise HTTPException(
                status_code=404,
                detail="Dataset not found",
            )

        return dataset

    finally:
        db.close()

@router.get("/{dataset_id}/versions")
def list_dataset_versions(dataset_id: int):
    db = SessionLocal()

    try:
        dataset = (
            db.query(Dataset)
            .filter(Dataset.id == dataset_id)
            .first()
        )

        if dataset is None:
            raise HTTPException(
                status_code=404,
                detail="Dataset not found",
            )

        versions = (
            db.query(DatasetVersion)
            .filter(DatasetVersion.dataset_id == dataset_id)
            .order_by(DatasetVersion.version_number)
            .all()
        )

        return {
            "dataset_id": dataset_id,
            "versions": [
                {
                    "id": version.id,
                    "version_number": version.version_number,
                    "file_path": version.file_path,
                    "row_count": version.row_count,
                    "column_count": version.column_count,
                    "created_at": version.created_at,
                }
                for version in versions
            ],
        }

    finally:
        db.close()


@router.get("/{dataset_id}/versions/compare")
def compare_dataset_versions(
    dataset_id: int,
    from_version: int = Query(..., alias="from", ge=1),
    to_version: int = Query(..., alias="to", ge=1),
):
    if from_version == to_version:
        raise HTTPException(
            status_code=400,
            detail="Cannot compare a version with itself",
        )

    db = SessionLocal()

    try:
        dataset = (
            db.query(Dataset)
            .filter(Dataset.id == dataset_id)
            .first()
        )

        if dataset is None:
            raise HTTPException(
                status_code=404,
                detail="Dataset not found",
            )

        versions = (
            db.query(DatasetVersion)
            .filter(
                DatasetVersion.dataset_id == dataset_id,
                DatasetVersion.version_number.in_(
                    [from_version, to_version]
                ),
            )
            .all()
        )

        versions_by_number = {
            version.version_number: version
            for version in versions
        }

        old_version = versions_by_number.get(from_version)
        new_version = versions_by_number.get(to_version)

        if old_version is None or new_version is None:
            raise HTTPException(
                status_code=404,
                detail="One or both dataset versions were not found",
            )

        try:
            old_df = pd.read_csv(old_version.file_path)
            new_df = pd.read_csv(new_version.file_path)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="Version file not found",
            )
        except (
            pd.errors.ParserError,
            pd.errors.EmptyDataError,
            UnicodeDecodeError,
            ValueError,
        ):
            raise HTTPException(
                status_code=500,
                detail="Failed to read dataset version files",
            )

        old_columns = list(old_df.columns)
        new_columns = list(new_df.columns)

        old_column_set = set(old_columns)
        new_column_set = set(new_columns)

        added_columns = [
            column
            for column in new_columns
            if column not in old_column_set
        ]

        removed_columns = [
            column
            for column in old_columns
            if column not in new_column_set
        ]

        common_columns = [
            column
            for column in old_columns
            if column in new_column_set
        ]

        dtype_changes = []

        for column in common_columns:
            old_dtype = str(old_df[column].dtype)
            new_dtype = str(new_df[column].dtype)

            if old_dtype != new_dtype:
                dtype_changes.append(
                    {
                        "column": column,
                        "from": old_dtype,
                        "to": new_dtype,
                    }
                )

        all_columns = list(
            dict.fromkeys(old_columns + new_columns)
        )

        missing_value_changes = []

        for column in all_columns:
            old_missing = (
                int(old_df[column].isna().sum())
                if column in old_column_set
                else None
            )

            new_missing = (
                int(new_df[column].isna().sum())
                if column in new_column_set
                else None
            )

            missing_value_changes.append(
                {
                    "column": column,
                    "from": old_missing,
                    "to": new_missing,
                    "delta": (
                        new_missing - old_missing
                        if old_missing is not None
                        and new_missing is not None
                        else None
                    ),
                }
            )

        old_row_count = len(old_df)
        new_row_count = len(new_df)

        old_column_count = len(old_columns)
        new_column_count = len(new_columns)

        return {
            "dataset_id": dataset_id,
            "from_version": from_version,
            "to_version": to_version,
            "row_count": {
                "from": old_row_count,
                "to": new_row_count,
                "delta": new_row_count - old_row_count,
            },
            "column_count": {
                "from": old_column_count,
                "to": new_column_count,
                "delta": new_column_count - old_column_count,
            },
            "added_columns": added_columns,
            "removed_columns": removed_columns,
            "dtype_changes": dtype_changes,
            "missing_value_changes": missing_value_changes,
        }

    finally:
        db.close()

@router.delete("/clear")
def clear_workspace():
    db = SessionLocal()

    try:
        datasets = db.query(Dataset).all()
        versions = db.query(DatasetVersion).all()

        file_paths = {
            Path(file_path)
            for file_path in [
                *(dataset.file_path for dataset in datasets),
                *(dataset.profile_path for dataset in datasets),
                *(version.file_path for version in versions),
            ]
            if file_path
        }

        file_paths |= collect_artifact_paths(db, [d.id for d in datasets])

        for dataset in datasets:
            db.delete(dataset)

        db.commit()

        for file_path in file_paths:
            file_path.unlink(missing_ok=True)

        return {
            "message": "Workspace cleared successfully",
            "deleted_datasets": len(datasets),
        }

    finally:
        db.close()


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int):
    db = SessionLocal()

    try:
        dataset = (
            db.query(Dataset)
            .filter(Dataset.id == dataset_id)
            .first()
        )

        if dataset is None:
            raise HTTPException(
                status_code=404,
                detail="Dataset not found",
            )

        versions = (
            db.query(DatasetVersion)
            .filter(DatasetVersion.dataset_id == dataset_id)
            .all()
        )

        file_paths = {
            Path(path)
            for path in [
                dataset.file_path,
                dataset.profile_path,
                *(version.file_path for version in versions),
            ]
            if path
        }

        file_paths |= collect_artifact_paths(db, [dataset_id])

        db.delete(dataset)
        db.commit()

        for file_path in file_paths:
            file_path.unlink(missing_ok=True)

        return {
            "message": "Dataset deleted successfully",
            "id": dataset_id,
        }

    finally:
        db.close()

@router.get("/{dataset_id}/profile")
def get_dataset_profile(dataset_id: int):
    db = SessionLocal()

    try:
        dataset = (
            db.query(Dataset)
            .filter(Dataset.id == dataset_id)
            .first()
        )

        if dataset is None:
            raise HTTPException(
                status_code=404,
                detail="Dataset not found",
            )

        if not dataset.profile_path:
            raise HTTPException(
                status_code=404,
                detail="Profile report not found",
            )

        profile_path = Path(dataset.profile_path)

        if not profile_path.is_file():
            raise HTTPException(
                status_code=404,
                detail="Profile report not found",
            )

        return FileResponse(
            path=profile_path,
            media_type="text/html",
            filename=profile_path.name,
        )

    finally:
        db.close()

@router.get("/{dataset_id}/summary")
def get_dataset_summary(dataset_id: int):
    db = SessionLocal()

    try:
        dataset = (
            db.query(Dataset)
            .filter(Dataset.id == dataset_id)
            .first()
        )

        if dataset is None:
            raise HTTPException(
                status_code=404,
                detail="Dataset not found",
            )

        try:
            summary = generate_summary(dataset.file_path)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="Dataset file not found",
            )
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="Unable to analyze dataset file",
            )

        return {
            "dataset_id": dataset.id,
            "filename": dataset.filename,
            "summary": summary,
        }

    finally:
        db.close()

@router.get("/{dataset_id}/visualization")
def get_dataset_visualization(
    dataset_id: int,
    chart_type: str,
    x_column: str,
    y_column: str | None = None,
    bins: int = 10,
):
    db = SessionLocal()

    try:
        dataset = (
            db.query(Dataset)
            .filter(Dataset.id == dataset_id)
            .first()
        )

        if dataset is None:
            raise HTTPException(
                status_code=404,
                detail="Dataset not found",
            )

        try:
            chart_data = generate_chart_data(
                file_path=dataset.file_path,
                chart_type=chart_type,
                x_column=x_column,
                y_column=y_column,
                bins=bins,
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="Dataset file not found",
            )
        except (pd.errors.ParserError, UnicodeDecodeError):
            raise HTTPException(
                status_code=400,
                detail="Unable to read dataset file",
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            )

        return {
            "dataset_id": dataset.id,
            "filename": dataset.filename,
            "visualization": chart_data,
        }

    finally:
        db.close()


@router.get("/{dataset_id}/export")
def export_dataset(dataset_id: int):
    db = SessionLocal()

    try:
        dataset = (
            db.query(Dataset)
            .filter(Dataset.id == dataset_id)
            .first()
        )

        if dataset is None:
            raise HTTPException(
                status_code=404,
                detail="Dataset not found",
            )

        file_path = Path(dataset.file_path)

        if not file_path.is_file():
            raise HTTPException(
                status_code=404,
                detail="Dataset file not found",
            )

        return FileResponse(
            path=file_path,
            media_type="text/csv",
            filename=dataset.filename,
        )

    finally:
        db.close()


@router.post("/{dataset_id}/query")
def query_dataset_endpoint(
    dataset_id: int,
    request: DatasetQueryRequest,
):
    db = SessionLocal()

    try:
        dataset = (
            db.query(Dataset)
            .filter(Dataset.id == dataset_id)
            .first()
        )

        if dataset is None:
            raise HTTPException(
                status_code=404,
                detail="Dataset not found",
            )

        try:
            result = query_dataset(
                file_path=dataset.file_path,
                filters=request.filters,
                limit=request.limit,
                offset=request.offset,
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="Dataset file not found",
            )
        except (pd.errors.ParserError, UnicodeDecodeError):
            raise HTTPException(
                status_code=400,
                detail="Unable to read dataset file",
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            )

        return {
            "dataset_id": dataset.id,
            "filename": dataset.filename,
            **result,
        }

    finally:
        db.close()


@router.get("/{dataset_id}/quality")
def get_dataset_quality(dataset_id: int):
    db = SessionLocal()

    try:
        dataset = (
            db.query(Dataset)
            .filter(Dataset.id == dataset_id)
            .first()
        )

        if dataset is None:
            raise HTTPException(
                status_code=404,
                detail="Dataset not found",
            )

        try:
            report = generate_quality_report(dataset.file_path)
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="Dataset file not found",
            )
        except (pd.errors.ParserError, UnicodeDecodeError):
            raise HTTPException(
                status_code=400,
                detail="Unable to read dataset file",
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            )

        return {
            "dataset_id": dataset.id,
            "filename": dataset.filename,
            "quality": report,
        }

    finally:
        db.close()


@router.post("/{dataset_id}/validate")
def validate_dataset_endpoint(
    dataset_id: int,
    request: DatasetValidationRequest,
):
    db = SessionLocal()

    try:
        dataset = (
            db.query(Dataset)
            .filter(Dataset.id == dataset_id)
            .first()
        )

        if dataset is None:
            raise HTTPException(
                status_code=404,
                detail="Dataset not found",
            )

        try:
            result = validate_dataset(
                file_path=dataset.file_path,
                rules=request.rules,
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail="Dataset file not found",
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            )

        return {
            "dataset_id": dataset.id,
            "filename": dataset.filename,
            "validation": result,
        }

    finally:
        db.close()
