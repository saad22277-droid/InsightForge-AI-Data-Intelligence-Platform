"""Post-hoc verification that every number in an AI answer is backed by computed data."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

NUMBER = re.compile(
    r"(?<![\w.])(?P<num>-?\$?\d[\d,]*(?:\.\d+)?)(?P<suffix>%|[kKmMbB](?![A-Za-z]))?"
)
LIST_MARKER = re.compile(r"^\s*(?:[-*]\s*)?\d+[.)]\s+", re.MULTILINE)
MULTIPLIER = {"k": 1e3, "m": 1e6, "b": 1e9}


@dataclass
class Token:
    raw: str
    value: float
    decimals: int
    percent: bool
    multiplier: float


def extract_numbers(text: str) -> list[Token]:
    text = LIST_MARKER.sub("", text or "")
    tokens = []
    for match in NUMBER.finditer(text):
        raw_num = match.group("num")
        cleaned = raw_num.replace(",", "").replace("$", "")
        # "1,2" style fragments are not thousands separators
        if "," in raw_num and not re.fullmatch(r"-?\$?\d{1,3}(,\d{3})+(\.\d+)?", raw_num):
            continue
        try:
            value = float(cleaned)
        except ValueError:
            continue
        suffix = match.group("suffix") or ""
        decimals = len(cleaned.split(".")[1]) if "." in cleaned else 0
        tokens.append(
            Token(
                raw=match.group(0),
                value=value,
                decimals=decimals,
                percent=suffix == "%",
                multiplier=MULTIPLIER.get(suffix.lower(), 1.0),
            )
        )
    return tokens


def collect_numbers(obj, into: set[float] | None = None) -> set[float]:
    """Gather every number in a JSON-like structure (including numbers in strings
    and list lengths, since "3 columns" is legitimately derived from a 3-item list)."""
    into = into if into is not None else set()
    if isinstance(obj, bool) or obj is None:
        return into
    if isinstance(obj, (int, float)):
        into.add(abs(float(obj)))
    elif isinstance(obj, str):
        for token in extract_numbers(obj):
            into.add(abs(token.value * token.multiplier))
    elif isinstance(obj, dict):
        into.add(float(len(obj)))
        for key, value in obj.items():
            collect_numbers(str(key), into)
            collect_numbers(value, into)
    elif isinstance(obj, (list, tuple)):
        into.add(float(len(obj)))
        for value in obj:
            collect_numbers(value, into)
    return into


def _tolerance(token: Token, value: float) -> float:
    """Allowed error given the precision the answer itself states.

    "0.85" tolerates +-0.005, "51,987" +-0.5, "52,000" +-500 (but never more than 1%
    of the value, so "100" cannot match 95) and "52k" +-500 (capped at 1.5%).
    """
    if token.decimals > 0:
        return 0.5 * 10 ** -token.decimals * token.multiplier
    digits = str(int(abs(token.value)))
    trailing_zeros = len(digits) - len(digits.rstrip("0")) if digits != "0" else 0
    tol = 0.5 * 10 ** trailing_zeros * token.multiplier
    cap = 0.015 if token.multiplier > 1 else 0.01
    return tol if trailing_zeros == 0 and token.multiplier == 1 else min(tol, cap * abs(value))


def _matches(token: Token, allowed: list[float]) -> bool:
    actual = abs(token.value) * token.multiplier
    candidates = [(actual, _tolerance(token, actual))]
    if token.percent:
        candidates.append((actual / 100, _tolerance(token, actual) / 100))
    for value, tol in candidates:
        for a in allowed:
            if abs(value - a) <= tol + 1e-9:
                return True
    return False


def verify_numbers(answer: str, evidence: object, known_text: list[str] | None = None) -> dict:
    allowed_set = collect_numbers(evidence)
    for text in known_text or []:
        collect_numbers(text, allowed_set)
    allowed = sorted(allowed_set)
    tokens = extract_numbers(answer)
    unverified = [t.raw for t in tokens if not _matches(t, allowed)]
    unique_unverified = list(dict.fromkeys(unverified))
    return {
        "verified": not unique_unverified,
        "numbers_checked": len(tokens),
        "unverified_numbers": unique_unverified[:20],
    }


def render_evidence(evidence: list[dict], max_chars: int = 3000) -> str:
    """Deterministic, human-readable rendering of tool results (used as a fallback)."""
    lines = []
    for item in evidence:
        if item.get("status") != "ok":
            continue
        args = ", ".join(f"{k}={v}" for k, v in (item.get("args") or {}).items())
        lines.append(f"- {item['tool']}({args}): "
                     + json.dumps(item.get("data"), default=str, ensure_ascii=False))
    text = "\n".join(lines)
    return text if len(text) <= max_chars else text[: max_chars - 1] + "…"
