"""Shared fixtures.

* Unit tests run on plain pandas data.
* API integration tests use an in-memory SQLite database (same SQLAlchemy models as
  PostgreSQL) and a scripted fake LLM - they never touch PostgreSQL or Gemini.
"""
import os

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
os.environ.pop("GEMINI_API_KEY", None)  # tests must never depend on a real key

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine, event  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

import src.app.models  # noqa: E402,F401  (register all tables)
from src.app import metrics  # noqa: E402
from src.app.config import get_settings  # noqa: E402
from src.app.database import Base, get_db  # noqa: E402
from src.app.main import app  # noqa: E402
from src.app.routes import datasets as datasets_routes  # noqa: E402
from src.app.services import data_io  # noqa: E402
from src.app.services.ai import guard  # noqa: E402
from src.app.services.ai.provider import get_llm, get_llm_optional  # noqa: E402
from tests.helpers import FakeLLM  # noqa: E402


@pytest.fixture
def sales_df():
    rng = np.random.default_rng(7)
    n = 200
    df = pd.DataFrame({
        "order_id": range(1, n + 1),
        "region": rng.choice(["North", "South", "East", "West"], n),
        "category": rng.choice(["Tech", "Home", "Toys"], n, p=[0.5, 0.3, 0.2]),
        "units": rng.integers(1, 20, n),
        "price": rng.normal(100, 15, n).round(2),
        "age": rng.integers(18, 70, n).astype(float),
    })
    df["sales"] = (df["units"] * df["price"] * rng.normal(1, 0.02, n)).round(2)
    df["is_big"] = np.where(df["sales"] > df["sales"].median(), "yes", "no")
    df.loc[rng.choice(n, 20, replace=False), "age"] = np.nan
    df.loc[3, "sales"] = 90000.0  # clear outlier
    return df


@pytest.fixture
def sales_csv_bytes(sales_df):
    return sales_df.to_csv(index=False).encode()


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated app environment: temp data dir, SQLite DB, no profile generation."""
    monkeypatch.setenv("AIDP_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    monkeypatch.setenv("AI_REQUESTS_PER_MINUTE", "1000")
    get_settings.cache_clear()
    settings = get_settings()
    settings.ensure_dirs()

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)

    @event.listens_for(engine, "connect")
    def _fk(dbapi_connection, _):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def override_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    monkeypatch.setattr(datasets_routes, "SessionLocal", Session)
    monkeypatch.setattr(datasets_routes, "UPLOAD_DIR", settings.raw_dir)
    monkeypatch.setattr(datasets_routes, "PROFILE_DIR", settings.profiles_dir)
    monkeypatch.setattr(
        datasets_routes, "generate_profile",
        lambda src, dst: Path(dst).write_text("<html>profile</html>"))
    app.dependency_overrides[get_db] = override_db
    data_io.clear_cache()
    guard.reset()
    metrics.reset()
    yield {"Session": Session, "settings": settings, "tmp": tmp_path}
    app.dependency_overrides.clear()
    get_settings.cache_clear()
    data_io.clear_cache()


@pytest.fixture
def client(env):
    return TestClient(app)


@pytest.fixture
def dataset_id(client, sales_csv_bytes):
    response = client.post("/datasets/upload",
                           files={"file": ("sales.csv", sales_csv_bytes, "text/csv")})
    assert response.status_code == 200, response.text
    return response.json()["id"]


@pytest.fixture
def use_llm(fake_llm):
    """Install the scripted fake LLM for AI endpoints."""
    app.dependency_overrides[get_llm] = lambda: fake_llm
    app.dependency_overrides[get_llm_optional] = lambda: (fake_llm, None)
    yield fake_llm
