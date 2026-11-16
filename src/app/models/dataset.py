from datetime import datetime

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.app.database import Base


class Dataset(Base):
    __tablename__ = "datasets"

    versions: Mapped[list["DatasetVersion"]] = relationship(
    back_populates="dataset",
    cascade="all, delete-orphan",
    passive_deletes=True,
)

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    file_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    profile_path: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    row_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    column_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="uploaded",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )