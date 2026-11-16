"""Add Stage 5 tables: chat, cleaning runs, ML runs, reports

Revision ID: a1b2c3d4e5f6
Revises: 73db4e02e19b
Create Date: 2026-11-13 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "73db4e02e19b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

JSON_TYPE = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")


def upgrade() -> None:
    op.create_table(
        "chat_conversations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_chat_conversations_id"), "chat_conversations", ["id"])
    op.create_index(
        op.f("ix_chat_conversations_dataset_id"), "chat_conversations", ["dataset_id"]
    )

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("conversation_id", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(length=20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("evidence", JSON_TYPE, nullable=True),
        sa.Column("grounding", JSON_TYPE, nullable=True),
        sa.Column("model", sa.String(length=100), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["conversation_id"], ["chat_conversations.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_chat_messages_id"), "chat_messages", ["id"])
    op.create_index(
        op.f("ix_chat_messages_conversation_id"), "chat_messages", ["conversation_id"]
    )

    op.create_table(
        "cleaning_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("source_version", sa.Integer(), nullable=True),
        sa.Column("result_version", sa.Integer(), nullable=False),
        sa.Column("operations", JSON_TYPE, nullable=False),
        sa.Column("summary", JSON_TYPE, nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_cleaning_runs_id"), "cleaning_runs", ["id"])
    op.create_index(op.f("ix_cleaning_runs_dataset_id"), "cleaning_runs", ["dataset_id"])

    op.create_table(
        "ml_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("dataset_version_number", sa.Integer(), nullable=True),
        sa.Column("dataset_sha256", sa.String(length=64), nullable=True),
        sa.Column("model_name", sa.String(length=150), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("task", sa.String(length=30), nullable=False),
        sa.Column("algorithm", sa.String(length=50), nullable=False),
        sa.Column("target", sa.String(length=255), nullable=True),
        sa.Column("features", JSON_TYPE, nullable=False),
        sa.Column("config", JSON_TYPE, nullable=False),
        sa.Column("metrics", JSON_TYPE, nullable=True),
        sa.Column("primary_metric", sa.String(length=50), nullable=True),
        sa.Column("primary_metric_value", sa.Float(), nullable=True),
        sa.Column("details", JSON_TYPE, nullable=True),
        sa.Column("environment", JSON_TYPE, nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("artifact_path", sa.String(length=500), nullable=True),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "dataset_id", "model_name", "version", name="uq_ml_run_model_version"
        ),
    )
    op.create_index(op.f("ix_ml_runs_id"), "ml_runs", ["id"])
    op.create_index(op.f("ix_ml_runs_dataset_id"), "ml_runs", ["dataset_id"])

    op.create_table(
        "reports",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=200), nullable=False),
        sa.Column("file_path", sa.String(length=500), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("include_ai", sa.Boolean(), nullable=False),
        sa.Column("ai_included", sa.Boolean(), nullable=False),
        sa.Column("sections", JSON_TYPE, nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_reports_id"), "reports", ["id"])
    op.create_index(op.f("ix_reports_dataset_id"), "reports", ["dataset_id"])


def downgrade() -> None:
    op.drop_table("reports")
    op.drop_table("ml_runs")
    op.drop_table("cleaning_runs")
    op.drop_table("chat_messages")
    op.drop_table("chat_conversations")
