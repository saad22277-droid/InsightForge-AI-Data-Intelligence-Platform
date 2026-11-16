"""initial database schema

Revision ID: be1adc90fcf0
Revises: 
Create Date: 2026-11-04 14:31:29.748507

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'be1adc90fcf0'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    This revision was originally empty because the ``datasets`` table was created
    outside Alembic during early development. Without it, a fresh database fails
    at the next revision (which has a foreign key to ``datasets``).

    The table is now created here *only if it does not already exist*, so
    existing databases (which are already past this revision) are unaffected and
    fresh databases can be initialised with ``alembic upgrade head``.
    """
    bind = op.get_bind()
    if "datasets" in sa.inspect(bind).get_table_names():
        return

    op.create_table(
        "datasets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("file_path", sa.String(length=500), nullable=False),
        sa.Column("profile_path", sa.String(length=500), nullable=True),
        sa.Column("row_count", sa.Integer(), nullable=True),
        sa.Column("column_count", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=50), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_datasets_id"), "datasets", ["id"], unique=False)


def downgrade() -> None:
    """Downgrade schema (the datasets table is intentionally left in place)."""
    pass
