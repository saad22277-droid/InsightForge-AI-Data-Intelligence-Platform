import dataclasses
import json
import os

import pytest

from src.app.config import get_settings


def test_original_health_and_root_contracts_unchanged(client):
    assert client.get("/health").json() == {"status": "healthy"}
    assert client.get("/").json() == {"message": "AI Data Platform is running", "version": "0.1.0"}


def test_readiness_checks_database_and_storage(client):
    r = client.get("/health/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready" and all(r.json()["checks"].values())


def test_metrics_count_requests_and_ai_activity(client, dataset_id):
    client.get(f"/datasets/{dataset_id}/eda")
    client.get("/datasets/999/eda")
    snap = client.get("/metrics").json()
    assert snap["counters"]["http_requests"] >= 3 and snap["counters"]["http_4xx"] >= 1
    assert snap["latency"]["http"]["count"] >= 3 and snap["uptime_seconds"] >= 0


def test_system_status_counts(client, dataset_id):
    status = client.get("/system/status").json()
    assert status["counts"]["datasets"] == 1
    assert status["settings"]["gemini_configured"] is False


def test_ai_status_without_key(client):
    s = client.get("/ai/status").json()
    assert s["configured"] is False and s["reachable"] is None and s["provider"] == "google-gemini"
    assert "free" in s["free_tier_notice"].lower()


def test_api_key_never_appears_in_any_response(client, dataset_id, monkeypatch):
    secret = "AIzaSyTOP-SECRET-KEY-1234567890"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    get_settings.cache_clear()
    for path in ["/ai/status", "/system/status", "/metrics", "/health/ready", "/ml/catalog",
                 f"/datasets/{dataset_id}/metadata"]:
        assert secret not in client.get(path).text, path
    assert get_settings().public_dict()["gemini_configured"] is True
    assert secret not in json.dumps(get_settings().public_dict())
    assert secret not in repr(dataclasses.replace(get_settings(), gemini_api_key=None))


def test_ai_status_live_check_reports_gemini_errors(client, monkeypatch):
    from src.app.routes import ai as ai_routes
    from src.app.services.ai.errors import AIServiceError
    monkeypatch.setenv("GEMINI_API_KEY", "x" * 20)
    get_settings.cache_clear()

    class Boom:
        model_name = "m"

        def generate(self, *a, **k):
            raise AIServiceError("auth", "Gemini rejected the API key.")

    monkeypatch.setattr(ai_routes, "get_llm", lambda: Boom())
    s = client.get("/ai/status", params={"check": True}).json()
    assert s["reachable"] is False and s["error"]["kind"] == "auth"
    assert s["last_error"]["kind"] == "auth"

    class Good:
        model_name = "m"

        def generate(self, *a, **k):
            from src.app.services.ai.llm import LLMResponse
            return LLMResponse(text="OK", tool_calls=[])

    monkeypatch.setattr(ai_routes, "get_llm", lambda: Good())
    assert client.get("/ai/status", params={"check": True}).json()["reachable"] is True


def test_request_id_header_present(client):
    assert len(client.get("/health").headers["x-request-id"]) == 12


def test_logs_never_contain_dataset_values_or_questions(client, dataset_id, use_llm, capsys):
    from tests.helpers import text
    use_llm.queue(text("Nothing numeric here."))
    client.post(f"/datasets/{dataset_id}/chat", json={"message": "SECRET-QUESTION-TEXT-XYZ"})
    out = capsys.readouterr().out
    assert "SECRET-QUESTION-TEXT-XYZ" not in out
