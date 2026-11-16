"""Helpers shared by the new routers."""
from __future__ import annotations

import pandas as pd
from fastapi import HTTPException
from sqlalchemy.orm import Session

from src.app.models.dataset import Dataset
from src.app.services.data_io import DatasetError, load_dataframe


def get_dataset_or_404(db: Session, dataset_id: int) -> Dataset:
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset


def load_df_or_http(dataset: Dataset) -> pd.DataFrame:
    try:
        return load_dataframe(dataset.file_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset file not found") from None
    except DatasetError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
