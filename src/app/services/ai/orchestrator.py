"""Chat orchestration: Gemini plans tool calls, deterministic code computes.

    question ──► Gemini ──► tool call(s) ──► validated pandas tools ──► results
                   ▲                                                       │
                   └──────────────── final explanation ◄───────────────────┘
                                          │
                                  grounding verifier
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import pandas as pd

from src.app import metrics
from src.app.config import Settings, get_settings
from src.app.services.ai import prompts
from src.app.services.ai.errors import AIServiceError
from src.app.services.ai.grounding import render_evidence, verify_numbers
from src.app.services.ai.guard import ai_request_slot
from src.app.services.ai.llm import LLMClient, ToolCall, Turn
from src.app.services.ai.tools import (
    ToolLimits,
    execute_tool,
    sanitize_text,
    tool_declarations,
)
from src.app.services.data_io import numeric_columns

logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    answer: str
    evidence: list[dict]
    grounding: dict
    tool_calls_used: int
    model: str
    latency_ms: int
    fallback_used: bool = False
    usage: dict = field(default_factory=dict)


def build_dataset_card(df: pd.DataFrame, filename: str, limit_columns: int = 60) -> str:
    """Schema-only description (names/types/missing counts). No raw rows, no statistics."""
    numeric = set(numeric_columns(df))
    lines = [
        f"Dataset file: {sanitize_text(filename, 80)}",
        f"Rows: {len(df)}; columns: {len(df.columns)}",
        "Columns (name | type | missing values):",
    ]
    for column in list(df.columns)[:limit_columns]:
        kind = "numeric" if column in numeric else "text/categorical"
        lines.append(f"- {sanitize_text(column, 60)} | {kind} | {int(df[column].isna().sum())}")
    if len(df.columns) > limit_columns:
        lines.append(f"(+{len(df.columns) - limit_columns} more columns)")
    return "\n".join(lines)


def _system_prompt(card: str) -> str:
    return (
        prompts.CHAT_SYSTEM_PROMPT
        + "\n<dataset_schema untrusted_labels=\"true\">\n" + card + "\n</dataset_schema>\n"
    )


def answer_question(
    llm: LLMClient,
    df: pd.DataFrame,
    filename: str,
    question: str,
    history: list[tuple[str, str]],
    settings: Settings | None = None,
) -> ChatResult:
    settings = settings or get_settings()
    question = (question or "").strip()
    if not question:
        raise ValueError("Question must not be empty")
    if len(question) > settings.ai_max_message_chars:
        raise ValueError(
            f"Question is too long (limit {settings.ai_max_message_chars} characters)")

    started = time.perf_counter()
    limits = ToolLimits(max_rows=settings.ai_max_tool_rows,
                        max_cell_chars=settings.ai_max_cell_chars)
    card = build_dataset_card(df, filename)
    system = _system_prompt(card)
    tools = tool_declarations()

    turns: list[Turn] = []
    for role, text in history[-settings.ai_max_history_messages:]:
        turns.append(Turn(role="user" if role == "user" else "model", text=text))
    turns.append(Turn(role="user", text=question))

    evidence: list[dict] = []
    calls_used = 0
    usage: dict = {}
    retried = False
    empty_nudges = 0
    answer = ""

    with ai_request_slot():
        metrics.incr("ai_chat_requests")
        guard = 0
        allow_tools = True
        while guard < settings.ai_max_tool_calls + 4:
            guard += 1
            response = llm.generate(system, turns, tools if allow_tools else None)
            for key, value in (response.usage or {}).items():
                if isinstance(value, int):
                    usage[key] = usage.get(key, 0) + value

            if response.tool_calls and allow_tools:
                turns.append(Turn(role="model", text=response.text,
                                  tool_calls=response.tool_calls, raw=response.raw))
                results = []
                for call in response.tool_calls:
                    if calls_used >= settings.ai_max_tool_calls:
                        results.append((call.name, {
                            "status": "error",
                            "error": "Tool-call budget exhausted for this question."}))
                        continue
                    calls_used += 1
                    metrics.incr("ai_tool_calls")
                    outcome = execute_tool(df, call.name, call.args, limits)
                    evidence.append(outcome)
                    results.append((call.name, outcome))
                turns.append(Turn(role="tool", tool_results=results))
                if calls_used >= settings.ai_max_tool_calls:
                    allow_tools = False
                    turns.append(Turn(role="user", text=prompts.BUDGET_EXHAUSTED_MESSAGE))
                continue

            answer = response.text.strip()
            if not answer:
                if empty_nudges < 1:
                    empty_nudges += 1
                    turns.append(Turn(role="user", text=prompts.BUDGET_EXHAUSTED_MESSAGE))
                    continue
                raise AIServiceError("empty", "Gemini returned an empty answer.")

            known = [question, card] + [t for _, t in history]
            grounding = verify_numbers(answer, evidence, known)
            if grounding["verified"] or retried:
                break
            retried = True
            metrics.incr("ai_grounding_retries")
            turns.append(Turn(role="model", text=answer))
            turns.append(Turn(role="user", text=prompts.RETRY_NUMBERS_MESSAGE.format(
                numbers=", ".join(grounding["unverified_numbers"][:8]))))
            allow_tools = calls_used < settings.ai_max_tool_calls
            answer = ""
        else:
            raise AIServiceError("unknown", "The assistant did not finish within the step limit.")

    fallback = False
    known = [question, card] + [t for _, t in history]
    grounding = verify_numbers(answer, evidence, known) if answer else {
        "verified": False, "numbers_checked": 0, "unverified_numbers": []}
    if not grounding["verified"]:
        fallback = True
        metrics.incr("ai_grounding_failures")
        rendered = render_evidence(evidence)
        answer = (
            "I could not verify every number in my explanation against the computed results, "
            "so I am showing the raw computed results instead.\n\n"
            + (rendered or "No analysis results were produced. Try rephrasing your question.")
        )
        grounding = {**grounding, "fallback": True}

    latency_ms = int((time.perf_counter() - started) * 1000)
    metrics.observe("ai_chat", latency_ms / 1000)
    logger.info("chat answered", extra={
        "tool_calls": calls_used, "latency_ms": latency_ms, "fallback": fallback,
        "verified": grounding["verified"], "answer_chars": len(answer)})
    return ChatResult(answer=answer, evidence=evidence, grounding=grounding,
                      tool_calls_used=calls_used, model=llm.model_name,
                      latency_ms=latency_ms, fallback_used=fallback, usage=usage)
