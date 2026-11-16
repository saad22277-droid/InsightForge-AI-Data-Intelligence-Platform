from pydantic import BaseModel
import math

class DatasetResponse(BaseModel):
    id: int
    filename: str
    profile_path: str | None
    row_count: int | None
    column_count: int | None
    status: str


from typing import Literal

from pydantic import BaseModel, Field


class DatasetFilter(BaseModel):
    column: str = Field(min_length=1, max_length=255)
    operator: Literal[
        "eq",
        "ne",
        "gt",
        "gte",
        "lt",
        "lte",
        "contains",
        "in",
        "is_null",
        "not_null",
    ]
    value: str | int | float | bool | list[str | int | float | bool] | None = None


class DatasetQueryRequest(BaseModel):
    filters: list[DatasetFilter] = Field(default_factory=list, max_length=20)
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class RequiredColumnRule(BaseModel):
    type: Literal["required_column"]
    column: str = Field(min_length=1, max_length=255)


class NumericRangeRule(BaseModel):
    type: Literal["numeric_range"]
    column: str = Field(min_length=1, max_length=255)
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def validate_range(self):
        if self.minimum is None and self.maximum is None:
            raise ValueError(
                "At least one of minimum or maximum must be provided"
            )

        if self.minimum is not None and not math.isfinite(self.minimum):
            raise ValueError("minimum must be a finite number")

        if self.maximum is not None and not math.isfinite(self.maximum):
            raise ValueError("maximum must be a finite number")

        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError(
                "minimum must be less than or equal to maximum"
            )

        return self


class AllowedValuesRule(BaseModel):
    type: Literal["allowed_values"]
    column: str = Field(min_length=1, max_length=255)
    values: list[str | int | float | bool] = Field(
        min_length=1,
        max_length=100,
    )


class DatasetValidationRequest(BaseModel):
    rules: list[
        RequiredColumnRule | NumericRangeRule | AllowedValuesRule
    ] = Field(min_length=1, max_length=50)

