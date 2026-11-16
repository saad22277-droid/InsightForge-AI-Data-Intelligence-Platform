"""Protects the (free-tier) Gemini quota and the server from overload."""
from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager

from src.app.config import get_settings
from src.app.services.ai.errors import AIServiceError

_lock = threading.Lock()
_calls: deque[float] = deque()
_semaphore: threading.BoundedSemaphore | None = None
_semaphore_size = 0


def reset() -> None:
    global _semaphore, _semaphore_size
    with _lock:
        _calls.clear()
        _semaphore, _semaphore_size = None, 0


def _get_semaphore(size: int) -> threading.BoundedSemaphore:
    global _semaphore, _semaphore_size
    with _lock:
        if _semaphore is None or _semaphore_size != size:
            _semaphore, _semaphore_size = threading.BoundedSemaphore(size), size
        return _semaphore


@contextmanager
def ai_request_slot():
    """Allow at most N concurrent AI requests and M requests per minute."""
    settings = get_settings()
    now = time.monotonic()
    with _lock:
        while _calls and now - _calls[0] > 60:
            _calls.popleft()
        if len(_calls) >= settings.ai_requests_per_minute:
            wait = max(1, int(60 - (now - _calls[0])))
            raise AIServiceError(
                "rate_limited",
                f"Too many AI requests. Limit is {settings.ai_requests_per_minute} per "
                f"minute (protects your Gemini quota). Try again in about {wait}s.",
                retry_after=wait,
            )
        _calls.append(now)
    semaphore = _get_semaphore(max(1, settings.ai_max_concurrent))
    if not semaphore.acquire(blocking=False):
        raise AIServiceError("busy", "The AI service is busy with other requests. Please retry.",
                             retry_after=5)
    try:
        yield
    finally:
        semaphore.release()
