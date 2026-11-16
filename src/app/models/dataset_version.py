from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.app.database import Base


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    __table_args__ = (
        UniqueConstraint(
            "dataset_id",
            "version_number",
            name="uq_dataset_version_number",
        ),
    )

    dataset: Mapped["Dataset"] = relationship(
        back_populates="versions",
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    dataset_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    version_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    file_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    row_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    column_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )