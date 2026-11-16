"""Shared column types.

``JSONType`` uses JSONB on PostgreSQL and generic JSON elsewhere (the test-suite
runs the same models on SQLite).
"""
from sqlalchemy import JSON
from sqlalchemy.dialects.postgresql import JSONB

JSONType = JSON().with_variant(JSONB(), "postgresql")
