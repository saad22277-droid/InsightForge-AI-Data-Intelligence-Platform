"""Central application configuration.

All settings come from environment variables (optionally loaded from ``.env``).
Secrets such as ``GEMINI_API_KEY`` are never hard-coded, logged, or returned
by any API endpoint.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # --- Gemini -----------------------------------------------------------
    gemini_api_key: str | None
    gemini_model: str
    gemini_timeout_seconds: float
    gemini_max_retries: int
    gemini_temperature: float
    gemini_max_output_tokens: int
    # --- AI orchestration limits -----------------------------------------
    ai_max_tool_calls: int
    ai_max_history_messages: int
    ai_max_message_chars: int
    ai_max_tool_rows: int
    ai_max_cell_chars: int
    ai_requests_per_minute: int
    ai_max_concurrent: int
    # --- Uploads / data ---------------------------------------------------
    max_upload_mb: int
    max_dataset_rows: int
    max_dataset_columns: int
    data_dir: Path
    # --- ML ---------------------------------------------------------------
    ml_min_rows: int
    ml_max_train_rows: int
    ml_random_state: int
    # --- Ops --------------------------------------------------------------
    log_level: str
    log_json: bool
    cors_origins: tuple[str, ...]

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def profiles_dir(self) -> Path:
        return self.data_dir / "profiles"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    @property
    def ai_configured(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        for directory in (
            self.raw_dir,
            self.profiles_dir,
            self.models_dir,
            self.reports_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def public_dict(self) -> dict:
        """Settings safe to expose to the UI (never includes the API key)."""
        return {
            "gemini_model": self.gemini_model,
            "gemini_configured": self.ai_configured,
            "gemini_timeout_seconds": self.gemini_timeout_seconds,
            "gemini_max_retries": self.gemini_max_retries,
            "ai_max_tool_calls": self.ai_max_tool_calls,
            "ai_max_history_messages": self.ai_max_history_messages,
            "ai_requests_per_minute": self.ai_requests_per_minute,
            "max_upload_mb": self.max_upload_mb,
            "max_dataset_rows": self.max_dataset_rows,
            "max_dataset_columns": self.max_dataset_columns,
            "ml_min_rows": self.ml_min_rows,
            "ml_max_train_rows": self.ml_max_train_rows,
            "log_level": self.log_level,
        }


@lru_cache
def get_settings() -> Settings:
    api_key = (os.getenv("GEMINI_API_KEY") or "").strip() or None
    origins = os.getenv(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    )
    return Settings(
        gemini_api_key=api_key,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip(),
        gemini_timeout_seconds=_float("GEMINI_TIMEOUT_SECONDS", 60.0),
        gemini_max_retries=_int("GEMINI_MAX_RETRIES", 2),
        gemini_temperature=_float("GEMINI_TEMPERATURE", 0.0),
        gemini_max_output_tokens=_int("GEMINI_MAX_OUTPUT_TOKENS", 2048),
        ai_max_tool_calls=_int("AI_MAX_TOOL_CALLS", 6),
        ai_max_history_messages=_int("AI_MAX_HISTORY_MESSAGES", 12),
        ai_max_message_chars=_int("AI_MAX_MESSAGE_CHARS", 1000),
        ai_max_tool_rows=_int("AI_MAX_TOOL_ROWS", 20),
        ai_max_cell_chars=_int("AI_MAX_CELL_CHARS", 80),
        ai_requests_per_minute=_int("AI_REQUESTS_PER_MINUTE", 20),
        ai_max_concurrent=_int("AI_MAX_CONCURRENT", 2),
        max_upload_mb=_int("MAX_UPLOAD_MB", 25),
        max_dataset_rows=_int("MAX_DATASET_ROWS", 500_000),
        max_dataset_columns=_int("MAX_DATASET_COLUMNS", 500),
        data_dir=Path(os.getenv("AIDP_DATA_DIR", "data")),
        ml_min_rows=_int("ML_MIN_ROWS", 20),
        ml_max_train_rows=_int("ML_MAX_TRAIN_ROWS", 100_000),
        ml_random_state=_int("ML_RANDOM_STATE", 42),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        log_json=os.getenv("LOG_JSON", "true").lower() in {"1", "true", "yes"},
        cors_origins=tuple(o.strip() for o in origins.split(",") if o.strip()),
    )
