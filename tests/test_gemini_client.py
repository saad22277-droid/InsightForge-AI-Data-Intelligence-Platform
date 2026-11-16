"""Gemini integration tests use a fake SDK client: no network, no API key."""
import dataclasses

import pytest
from google.genai import errors, types

from src.app.config import get_settings
from src.app.services.ai.errors import AIServiceError
from src.app.services.ai.llm import GeminiClient, Turn, classify_api_error


class FakeModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append({"model": model, "contents": contents, "config": config})
        item = self.outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeSDK:
    def __init__(self, outcomes):
        self.models = FakeModels(outcomes)


def response(text=None, call=None):
    parts = []
    if text is not None:
        parts.append(types.Part(text=text))
    if call is not None:
        parts.append(types.Part.from_function_call(name=call[0], args=call[1]))
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts))])


def make(outcomes, retries=2):
    settings = dataclasses.replace(get_settings(), gemini_api_key="test-key",
                                   gemini_max_retries=retries, gemini_model="gemini-test")
    sleeps = []
    sdk = FakeSDK(outcomes)
    return GeminiClient(settings, client=sdk, sleep=sleeps.append), sdk, sleeps


def api_error(code, message="x"):
    return errors.ClientError(code, {"error": {"message": message, "status": "X"}})


def test_text_response_and_model_name_forwarded():
    client, sdk, _ = make([response("hello")])
    out = client.generate("sys", [Turn(role="user", text="hi")])
    assert out.text == "hello" and not out.tool_calls
    assert sdk.models.calls[0]["model"] == "gemini-test"
    assert sdk.models.calls[0]["config"].system_instruction == "sys"


def test_tool_declarations_disable_automatic_execution():
    client, sdk, _ = make([response(call=("value_counts", {"column": "a"}))])
    tools = [{"name": "value_counts", "description": "d",
              "parameters": {"type": "object", "properties": {"column": {"type": "string"}}}}]
    out = client.generate("s", [Turn(role="user", text="q")], tools)
    assert out.tool_calls[0].name == "value_counts" and out.tool_calls[0].args == {"column": "a"}
    config = sdk.models.calls[0]["config"]
    assert config.automatic_function_calling.disable is True
    assert config.tools[0].function_declarations[0].name == "value_counts"


def test_tool_results_are_sent_back_as_function_responses():
    client, sdk, _ = make([response("done")])
    turns = [Turn(role="user", text="q"),
             Turn(role="model", tool_calls=[]),
             Turn(role="tool", tool_results=[("value_counts", {"status": "ok"})])]
    client.generate("s", turns)
    last = sdk.models.calls[0]["contents"][-1]
    assert last.parts[0].function_response.name == "value_counts"


def test_retries_rate_limit_with_backoff_then_succeeds():
    client, sdk, sleeps = make([api_error(429, "rate"), api_error(503, "down"), response("ok")])
    assert client.generate("s", [Turn(role="user", text="q")]).text == "ok"
    assert len(sdk.models.calls) == 3 and len(sleeps) == 2 and all(0 < s <= 20 for s in sleeps)


def test_gives_up_after_max_retries():
    client, sdk, _ = make([api_error(429)] * 5, retries=1)
    with pytest.raises(AIServiceError) as exc:
        client.generate("s", [Turn(role="user", text="q")])
    assert exc.value.kind == "rate_limited" and len(sdk.models.calls) == 2


@pytest.mark.parametrize("code,message,kind,retried", [
    (401, "bad", "auth", False),
    (403, "forbidden", "auth", False),
    (404, "model not found", "model_unavailable", False),
    (400, "API key not valid. Please pass a valid API key.", "auth", False),
    (400, "bad request", "bad_request", False),
])
def test_non_retryable_errors_fail_fast(code, message, kind, retried):
    client, sdk, sleeps = make([api_error(code, message), response("never")])
    with pytest.raises(AIServiceError) as exc:
        client.generate("s", [Turn(role="user", text="q")])
    assert exc.value.kind == kind and len(sdk.models.calls) == 1 and not sleeps


def test_quota_exhaustion_is_classified():
    err = classify_api_error(429, "You exceeded your current quota")
    assert err.kind == "quota" and err.status_code == 429


def test_timeout_is_classified_and_retried():
    class ReadTimeout(Exception):
        pass

    client, sdk, _ = make([ReadTimeout("timed out"), response("ok")])
    assert client.generate("s", [Turn(role="user", text="q")]).text == "ok"


def test_safety_block_and_empty_responses():
    blocked = types.GenerateContentResponse(
        candidates=[], prompt_feedback=types.GenerateContentResponsePromptFeedback(
            block_reason=types.BlockedReason.SAFETY))
    client, _, _ = make([blocked])
    with pytest.raises(AIServiceError) as exc:
        client.generate("s", [Turn(role="user", text="q")])
    assert exc.value.kind == "blocked"
    client, _, _ = make([types.GenerateContentResponse(candidates=[])])
    with pytest.raises(AIServiceError) as exc:
        client.generate("s", [Turn(role="user", text="q")])
    assert exc.value.kind == "empty"


def test_missing_api_key_is_a_clear_error():
    settings = dataclasses.replace(get_settings(), gemini_api_key=None)
    with pytest.raises(AIServiceError) as exc:
        GeminiClient(settings)
    assert exc.value.kind == "not_configured" and exc.value.status_code == 503
    assert "GEMINI_API_KEY" in exc.value.message


def test_raw_model_content_is_replayed_for_thought_signatures():
    raw = types.Content(role="model", parts=[types.Part.from_function_call(name="t", args={})])
    client, sdk, _ = make([response("ok")])
    client.generate("s", [Turn(role="user", text="q"), Turn(role="model", raw=raw)])
    assert sdk.models.calls[0]["contents"][1] is raw
