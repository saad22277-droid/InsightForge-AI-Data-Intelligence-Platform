from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from src.app.database import Base
from src.app.models.types import JSONType


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Report(Base):
    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    include_ai: Mapped[bool] = mapped_column(Boolean, nullable=False)
    ai_included: Mapped[bool] = mapped_column(Boolean, nullable=False)
    sections: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_now, nullable=False
    )
