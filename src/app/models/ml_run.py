from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from src.app.database import Base
from src.app.models.types import JSONType


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class MLRun(Base):
    """One tracked training run == one registered model version."""

    __tablename__ = "ml_runs"
    __table_args__ = (
        UniqueConstraint(
            "dataset_id", "model_name", "version", name="uq_ml_run_model_version"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dataset_version_number: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    dataset_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_name: Mapped[str] = mapped_column(String(150), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    task: Mapped[str] = mapped_column(String(30), nullable=False)
    algorithm: Mapped[str] = mapped_column(String(50), nullable=False)
    target: Mapped[str | None] = mapped_column(String(255), nullable=True)
    features: Mapped[list] = mapped_column(JSONType, nullable=False)
    config: Mapped[dict] = mapped_column(JSONType, nullable=False)
    metrics: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    primary_metric: Mapped[str | None] = mapped_column(String(50), nullable=True)
    primary_metric_value: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    details: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    environment: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    artifact_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_now, nullable=False
    )
