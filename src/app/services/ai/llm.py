"""LLM abstraction + Google Gemini implementation (official ``google-genai`` SDK).

The rest of the application only depends on :class:`LLMClient`, so tests inject a
scripted fake and never call the live API.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from src.app import metrics
from src.app.config import Settings, get_settings
from src.app.services.ai.errors import AIServiceError

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    name: str
    args: dict


@dataclass
class Turn:
    """Provider-neutral conversation turn.

    role: ``user`` | ``model`` | ``tool``. ``raw`` keeps the provider's original
    model content so it can be replayed verbatim (needed for thought signatures).
    """

    role: str
    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[tuple[str, dict]] = field(default_factory=list)
    raw: Any = None


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    raw: Any = None
    usage: dict = field(default_factory=dict)


class LLMClient(Protocol):
    model_name: str

    def generate(
        self, system: str, turns: list[Turn], tools: list[dict] | None = None
    ) -> LLMResponse: ...


# --------------------------------------------------------------------- Gemini
def classify_api_error(code: int | None, message: str) -> AIServiceError:
    text = (message or "").lower()
    if code == 429:
        if "quota" in text or "exceeded your current" in text:
            return AIServiceError(
                "quota",
                "Gemini quota or rate limit reached. Wait a minute and retry, or "
                "check your quota in Google AI Studio.",
                retry_after=30,
            )
        return AIServiceError("rate_limited",
                              "Gemini rate limit reached. Please retry shortly.",
                              retry_after=15)
    if code in (401, 403) or "api key not valid" in text or "api_key_invalid" in text:
        return AIServiceError(
            "auth",
            "Gemini rejected the API key. Check GEMINI_API_KEY in your .env file.")
    if code == 404 or "is not found" in text or "not supported" in text and "model" in text:
        return AIServiceError(
            "model_unavailable",
            "The configured Gemini model is unavailable. Set GEMINI_MODEL to a "
            "model your key can use.")
    if code == 400:
        return AIServiceError("bad_request", "Gemini rejected the request (invalid input).")
    if code is not None and code >= 500:
        return AIServiceError("server",
                              "Gemini is temporarily unavailable. Please retry.",
                              retry_after=5)
    return AIServiceError("unknown", "Unexpected error from the Gemini API.")


RETRYABLE = {"rate_limited", "server", "timeout"}


class GeminiClient:
    def __init__(
        self,
        settings: Settings | None = None,
        client: Any = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.settings = settings or get_settings()
        self.model_name = self.settings.gemini_model
        self._sleep = sleep
        self._client = client
        if self._client is None:
            if not self.settings.gemini_api_key:
                raise AIServiceError(
                    "not_configured",
                    "GEMINI_API_KEY is not set. Add it to your .env file to enable AI features.",
                )
            from google import genai
            from google.genai import types

            self._client = genai.Client(
                api_key=self.settings.gemini_api_key,
                http_options=types.HttpOptions(
                    timeout=int(self.settings.gemini_timeout_seconds * 1000)
                ),
            )

    # -- conversion helpers -------------------------------------------------
    @staticmethod
    def _to_contents(turns: list[Turn]):
        from google.genai import types

        contents = []
        for turn in turns:
            if turn.role == "user":
                contents.append(types.Content(role="user",
                                              parts=[types.Part(text=turn.text or "")]))
            elif turn.role == "model":
                if turn.raw is not None:
                    contents.append(turn.raw)
                    continue
                parts = []
                if turn.text:
                    parts.append(types.Part(text=turn.text))
                for call in turn.tool_calls:
                    parts.append(types.Part.from_function_call(name=call.name, args=call.args))
                contents.append(types.Content(role="model", parts=parts))
            else:  # tool results
                parts = [types.Part.from_function_response(name=n, response=r)
                         for n, r in turn.tool_results]
                contents.append(types.Content(role="user", parts=parts))
        return contents

    def _config(self, system: str, tools: list[dict] | None):
        from google.genai import types

        kwargs = dict(
            system_instruction=system,
            temperature=self.settings.gemini_temperature,
            max_output_tokens=self.settings.gemini_max_output_tokens,
        )
        if tools:
            kwargs["tools"] = [types.Tool(function_declarations=[
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters_json_schema=t["parameters"],
                ) for t in tools
            ])]
            kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
                disable=True)
        return types.GenerateContentConfig(**kwargs)

    # -- main entry ---------------------------------------------------------
    def generate(self, system: str, turns: list[Turn],
                 tools: list[dict] | None = None) -> LLMResponse:
        from google.genai import errors as genai_errors

        contents = self._to_contents(turns)
        config = self._config(system, tools)
        attempts = self.settings.gemini_max_retries + 1
        last: AIServiceError | None = None

        for attempt in range(1, attempts + 1):
            started = time.perf_counter()
            metrics.incr("gemini_requests")
            try:
                response = self._client.models.generate_content(
                    model=self.model_name, contents=contents, config=config)
                metrics.observe("gemini_call", time.perf_counter() - started)
                return self._parse(response)
            except AIServiceError as exc:
                last = exc
            except genai_errors.APIError as exc:
                last = classify_api_error(getattr(exc, "code", None), str(exc))
            except Exception as exc:  # noqa: BLE001 - network/timeouts vary by transport
                name = type(exc).__name__.lower()
                if "timeout" in name or "timed out" in str(exc).lower():
                    last = AIServiceError(
                        "timeout",
                        f"Gemini did not respond within "
                        f"{self.settings.gemini_timeout_seconds:.0f}s.")
                elif "connect" in name or "network" in name:
                    last = AIServiceError(
                        "server", "Could not reach the Gemini API. Check your internet connection.")
                else:
                    last = AIServiceError("unknown", "Unexpected error calling Gemini.")

            metrics.incr("gemini_errors")
            logger.warning("gemini call failed",
                           extra={"kind": last.kind, "attempt": attempt, "of": attempts})
            if last.kind not in RETRYABLE or attempt == attempts:
                break
            delay = min(20.0, (2 ** (attempt - 1)) + random.uniform(0, 0.5))
            if last.retry_after:
                delay = min(20.0, max(delay, min(last.retry_after, 20.0)))
            self._sleep(delay)
        raise last  # type: ignore[misc]

    @staticmethod
    def _parse(response) -> LLMResponse:
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            feedback = getattr(response, "prompt_feedback", None)
            reason = getattr(feedback, "block_reason", None)
            if reason:
                raise AIServiceError(
                    "blocked", "Gemini blocked this request for safety reasons.")
            raise AIServiceError("empty", "Gemini returned an empty response.")
        candidate = candidates[0]
        content = getattr(candidate, "content", None)
        parts = (getattr(content, "parts", None) or []) if content else []
        text = "".join(p.text for p in parts
                       if getattr(p, "text", None) and not getattr(p, "thought", False))
        calls = []
        for part in parts:
            fc = getattr(part, "function_call", None)
            if fc is not None and getattr(fc, "name", None):
                calls.append(ToolCall(name=fc.name, args=dict(fc.args or {})))
        if not text and not calls:
            finish = str(getattr(candidate, "finish_reason", "") or "")
            if "SAFETY" in finish.upper():
                raise AIServiceError("blocked", "Gemini blocked this response for safety reasons.")
            raise AIServiceError("empty", "Gemini returned no usable content.")
        usage_meta = getattr(response, "usage_metadata", None)
        usage = {}
        if usage_meta is not None:
            usage = {
                "prompt_tokens": getattr(usage_meta, "prompt_token_count", None),
                "output_tokens": getattr(usage_meta, "candidates_token_count", None),
                "total_tokens": getattr(usage_meta, "total_token_count", None),
            }
        return LLMResponse(text=text.strip(), tool_calls=calls, raw=content, usage=usage)


def build_default_client() -> LLMClient:
    """Create the production client (raises AIServiceError if not configured)."""
    return GeminiClient()
