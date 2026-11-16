"""Creates (and caches) the production LLM client. Tests override these functions
through FastAPI dependency overrides, so no test ever calls the live Gemini API."""
from __future__ import annotations

import threading

from src.app.config import get_settings
from src.app.services.ai.errors import AIServiceError
from src.app.services.ai.llm import GeminiClient, LLMClient

_lock = threading.Lock()
_cached: tuple[tuple, LLMClient] | None = None


def get_llm() -> LLMClient:
    """FastAPI dependency. Raises AIServiceError(not_configured) without a key."""
    global _cached
    settings = get_settings()
    if not settings.ai_configured:
        raise AIServiceError(
            "not_configured",
            "Gemini is not configured. Set GEMINI_API_KEY in your .env file and restart "
            "the backend. All non-AI features keep working without it.")
    key = (settings.gemini_api_key, settings.gemini_model, settings.gemini_timeout_seconds)
    with _lock:
        if _cached is None or _cached[0] != key:
            _cached = (key, GeminiClient(settings))
        return _cached[1]


def get_llm_optional() -> tuple[LLMClient | None, str | None]:
    """Dependency for features where AI is a bonus (e.g. reports)."""
    try:
        return get_llm(), None
    except AIServiceError as exc:
        return None, exc.message
