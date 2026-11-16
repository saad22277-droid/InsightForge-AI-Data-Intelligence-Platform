"""Tiny in-process operational metrics (resets when the server restarts)."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

_lock = threading.Lock()
_started = time.time()
_counters: dict[str, int] = defaultdict(int)
_latencies: dict[str, deque] = defaultdict(lambda: deque(maxlen=500))


def incr(name: str, amount: int = 1) -> None:
    with _lock:
        _counters[name] += amount


def observe(name: str, seconds: float) -> None:
    with _lock:
        _latencies[name].append(seconds)


def reset() -> None:
    with _lock:
        _counters.clear()
        _latencies.clear()


def snapshot() -> dict:
    with _lock:
        latencies = {}
        for name, values in _latencies.items():
            ordered = sorted(values)
            if not ordered:
                continue
            p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
            latencies[name] = {
                "count": len(ordered),
                "avg_ms": round(sum(ordered) / len(ordered) * 1000, 2),
                "p95_ms": round(p95 * 1000, 2),
            }
        return {
            "uptime_seconds": round(time.time() - _started, 1),
            "counters": dict(_counters),
            "latency": latencies,
        }
