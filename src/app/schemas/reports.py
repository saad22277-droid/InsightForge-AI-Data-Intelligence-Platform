from pydantic import BaseModel, Field


class ReportRequest(BaseModel):
    title: str | None = Field(default=None, max_length=150)
    include_ai: bool = False
    run_ids: list[int] | None = Field(default=None, max_length=5)
