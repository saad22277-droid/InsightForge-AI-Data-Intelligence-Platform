"""Alembic checks. The fresh-database test needs PostgreSQL:

    TEST_POSTGRES_URL=postgresql://user:pass@localhost:5432/scratch_db pytest tests/test_migrations.py

(The database is dropped/recreated by the test, so use a scratch database.)
"""
import os
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

ROOT = Path(__file__).resolve().parents[1]


def _config():
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    return cfg


def test_single_linear_migration_head():
    script = ScriptDirectory.from_config(_config())
    assert len(script.get_heads()) == 1
    chain = [r.revision for r in script.walk_revisions()]
    assert chain[0] == "a1b2c3d4e5f6" and chain[-1] == "be1adc90fcf0"


def test_models_and_migration_define_the_same_tables():
    import src.app.models  # noqa: F401
    from src.app.database import Base
    migration = (ROOT / "migrations/versions/a1b2c3d4e5f6_add_stage5_ai_ml_tables.py").read_text()
    for table in ("chat_conversations", "chat_messages", "cleaning_runs", "ml_runs", "reports"):
        assert table in Base.metadata.tables and f'"{table}"' in migration


@pytest.mark.skipif(not os.getenv("TEST_POSTGRES_URL"), reason="TEST_POSTGRES_URL not set")
def test_fresh_postgres_database_migrates_and_matches_models(monkeypatch):
    import subprocess
    import sys
    from sqlalchemy import create_engine, inspect

    url = os.environ["TEST_POSTGRES_URL"]
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    env = {**os.environ, "DATABASE_URL": url}
    for command in (["upgrade", "head"], ["check"]):
        result = subprocess.run([sys.executable, "-m", "alembic", *command], cwd=ROOT, env=env,
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    tables = set(inspect(engine).get_table_names())
    assert {"datasets", "dataset_versions", "chat_messages", "ml_runs", "reports"} <= tables
