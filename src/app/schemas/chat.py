from typing import Literal

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: int | None = Field(default=None, ge=1)


class ChartSpec(BaseModel):
    chart_type: str = Field(max_length=30)
    x_column: str | None = Field(default=None, max_length=255)
    y_column: str | None = Field(default=None, max_length=255)
    group_by: str | None = Field(default=None, max_length=255)
    agg: str = Field(default="count", max_length=20)
    bins: int = Field(default=10, ge=1, le=100)


class ExplainRequest(BaseModel):
    kind: Literal["dataset", "quality", "eda", "correlations", "ml_run", "chart", "insights"]
    run_id: int | None = Field(default=None, ge=1)
    chart: ChartSpec | None = None
    focus: str | None = Field(default=None, max_length=200)
