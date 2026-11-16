from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column

from src.app.database import Base
from src.app.models.types import JSONType


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class CleaningRun(Base):
    """Audit trail: which operations produced which dataset version."""

    __tablename__ = "cleaning_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    result_version: Mapped[int] = mapped_column(Integer, nullable=False)
    operations: Mapped[list] = mapped_column(JSONType, nullable=False)
    summary: Mapped[dict] = mapped_column(JSONType, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_now, nullable=False
    )
