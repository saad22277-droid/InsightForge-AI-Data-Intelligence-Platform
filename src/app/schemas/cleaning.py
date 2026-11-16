from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, model_validator

ColumnName = Annotated[str, Field(min_length=1, max_length=255)]


class DropDuplicates(BaseModel):
    type: Literal["drop_duplicates"]
    subset: list[ColumnName] | None = Field(default=None, max_length=100)
    keep: Literal["first", "last"] = "first"


class FillMissing(BaseModel):
    type: Literal["fill_missing"]
    column: ColumnName
    strategy: Literal["mean", "median", "mode", "constant"]
    value: str | int | float | bool | None = None


class DropMissingRows(BaseModel):
    type: Literal["drop_missing_rows"]
    columns: list[ColumnName] | None = Field(default=None, max_length=100)


class DropColumns(BaseModel):
    type: Literal["drop_columns"]
    columns: list[ColumnName] = Field(min_length=1, max_length=100)


class ConvertDtype(BaseModel):
    type: Literal["convert_dtype"]
    column: ColumnName
    dtype: Literal["numeric", "datetime", "string", "category", "boolean"]


class HandleOutliers(BaseModel):
    type: Literal["handle_outliers"]
    column: ColumnName
    method: Literal["iqr", "zscore"] = "iqr"
    action: Literal["clip", "remove", "set_missing"] = "clip"
    threshold: float | None = Field(default=None, gt=0, le=100)


class NormalizeColumnNames(BaseModel):
    type: Literal["normalize_column_names"]


class StandardizeText(BaseModel):
    type: Literal["standardize_text"]
    column: ColumnName
    strip: bool = True
    lowercase: bool = False
    collapse_whitespace: bool = True


class ReplaceInvalid(BaseModel):
    type: Literal["replace_invalid"]
    column: ColumnName
    values: list[str] | None = Field(default=None, max_length=100)


class InvalidToMissing(BaseModel):
    """Set numeric values outside [minimum, maximum] to missing."""

    type: Literal["invalid_to_missing"]
    column: ColumnName
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def _bounds(self):
        if self.minimum is None and self.maximum is None:
            raise ValueError("At least one of minimum or maximum is required")
        if (self.minimum is not None and self.maximum is not None
                and self.minimum > self.maximum):
            raise ValueError("minimum must be less than or equal to maximum")
        return self


class ScaleColumn(BaseModel):
    type: Literal["scale_column"]
    column: ColumnName
    method: Literal["minmax", "standard"]


CleaningOperation = Annotated[
    Union[
        DropDuplicates, FillMissing, DropMissingRows, DropColumns, ConvertDtype,
        HandleOutliers, NormalizeColumnNames, StandardizeText, ReplaceInvalid,
        InvalidToMissing, ScaleColumn,
    ],
    Field(discriminator="type"),
]


class CleaningRequest(BaseModel):
    operations: list[CleaningOperation] = Field(min_length=1, max_length=50)


class CleaningApplyRequest(CleaningRequest):
    note: str | None = Field(default=None, max_length=300)
