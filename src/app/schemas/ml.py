import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Scalar = str | int | float | bool | None


class FeatureSelection(BaseModel):
    method: Literal["none", "k_best"] = "none"
    k: int | None = Field(default=None, ge=1, le=200)


class TrainBase(BaseModel):
    task: Literal["regression", "classification", "clustering"]
    target: str | None = Field(default=None, max_length=255)
    features: list[str] | None = Field(default=None, max_length=500)
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    cv_folds: int = Field(default=5, ge=0, le=10)
    feature_selection: FeatureSelection = Field(default_factory=FeatureSelection)
    n_clusters: int | None = Field(default=None, ge=2, le=20)
    hyperparameters: dict[str, Scalar] = Field(default_factory=dict)
    model_name: str | None = Field(default=None, max_length=100)
    random_state: int | None = Field(default=None, ge=0, le=2**31 - 1)

    @field_validator("model_name")
    @classmethod
    def _name(cls, value):
        if value is not None and not re.fullmatch(r"[\w\-. ]{1,100}", value):
            raise ValueError("model_name may contain letters, digits, space, _ - .")
        return value

    @field_validator("hyperparameters")
    @classmethod
    def _hp_size(cls, value):
        if len(value) > 10:
            raise ValueError("At most 10 hyperparameters")
        return value


class TrainRequest(TrainBase):
    algorithm: str = Field(max_length=50)


class CompareRequest(TrainBase):
    algorithms: list[str] = Field(min_length=2, max_length=6)


class PredictRequest(BaseModel):
    records: list[dict[str, Scalar]] = Field(min_length=1, max_length=200)
