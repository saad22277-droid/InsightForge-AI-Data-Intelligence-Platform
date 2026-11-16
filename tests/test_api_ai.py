import pytest

from src.app.services.ai.errors import AIServiceError
from tests.helpers import call, text


def mean_price(sales_df):
    return f"{sales_df['price'].mean():.2f}"


def test_chat_creates_and_continues_a_persisted_conversation(client, dataset_id, use_llm, sales_df):
    use_llm.queue(call("compute_aggregate", operation="mean", column="price"),
                  text(f"The average price is {mean_price(sales_df)}."),
                  call("compute_aggregate", operation="max", column="price"),
                  text(f"The maximum price is {sales_df['price'].max():.2f}."))
    first = client.post(f"/datasets/{dataset_id}/chat", json={"message": "What is the average price?"})
    assert first.status_code == 200, first.text
    body = first.json()
    cid = body["conversation_id"]
    assert mean_price(sales_df) in body["assistant_message"]["content"]
    assert body["assistant_message"]["grounding"]["verified"] is True
    assert body["assistant_message"]["evidence"][0]["tool"] == "compute_aggregate"

    second = client.post(f"/datasets/{dataset_id}/chat",
                         json={"message": "And the maximum?", "conversation_id": cid}).json()
    assert second["conversation_id"] == cid
    # the second model request received the first exchange as context
    roles = [t.role for t in use_llm.requests[2]["turns"]]
    assert roles[:2] == ["user", "model"]

    conv = client.get(f"/datasets/{dataset_id}/conversations/{cid}").json()
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant", "user", "assistant"]
    assert conv["title"] == "What is the average price?"
    listing = client.get(f"/datasets/{dataset_id}/conversations").json()
    assert listing[0]["message_count"] == 4

    assert client.delete(f"/datasets/{dataset_id}/conversations/{cid}").status_code == 200
    assert client.get(f"/datasets/{dataset_id}/conversations/{cid}").status_code == 404


def test_conversations_are_removed_with_dataset(client, dataset_id, use_llm, env):
    use_llm.queue(text("Hello."))
    cid = client.post(f"/datasets/{dataset_id}/chat", json={"message": "hi"}).json()["conversation_id"]
    client.delete(f"/datasets/{dataset_id}")
    from src.app.models.chat import ChatConversation, ChatMessage
    with env["Session"]() as db:
        assert db.query(ChatConversation).count() == 0 and db.query(ChatMessage).count() == 0
    assert cid


def test_chat_input_validation_and_not_found(client, dataset_id, use_llm):
    assert client.post(f"/datasets/{dataset_id}/chat", json={"message": ""}).status_code == 422
    assert client.post(f"/datasets/{dataset_id}/chat", json={"message": "x" * 5000}).status_code == 422
    r = client.post(f"/datasets/{dataset_id}/chat", json={"message": "x" * 1500})
    assert r.status_code == 400 and "too long" in r.json()["detail"]
    assert client.post("/datasets/999/chat", json={"message": "hi"}).status_code == 404
    assert client.post(f"/datasets/{dataset_id}/chat",
                       json={"message": "hi", "conversation_id": 99}).status_code == 404


def test_chat_without_api_key_returns_clear_503_and_other_features_work(client, dataset_id):
    r = client.post(f"/datasets/{dataset_id}/chat", json={"message": "hi"})
    assert r.status_code == 503
    assert r.json()["detail"]["code"] == "not_configured"
    assert "GEMINI_API_KEY" in r.json()["detail"]["message"]
    assert client.get(f"/datasets/{dataset_id}/eda").status_code == 200       # core features unaffected


@pytest.mark.parametrize("kind,status,code", [
    ("quota", 429, "quota"), ("rate_limited", 429, "rate_limited"), ("timeout", 504, "timeout"),
    ("auth", 502, "auth"), ("model_unavailable", 502, "model_unavailable"), ("blocked", 422, "blocked"),
])
def test_gemini_failures_map_to_http_errors(client, dataset_id, use_llm, kind, status, code):
    use_llm.queue(AIServiceError(kind, f"{kind} problem", retry_after=7))
    r = client.post(f"/datasets/{dataset_id}/chat", json={"message": "hi"})
    assert r.status_code == status and r.json()["detail"]["code"] == code
    assert r.headers.get("retry-after") == "7"
    # a failed request stores nothing
    assert client.get(f"/datasets/{dataset_id}/conversations").json() == []


def test_hallucination_fallback_is_persisted_with_evidence(client, dataset_id, use_llm):
    use_llm.queue(call("compute_aggregate", operation="mean", column="price"),
                  text("It is 1234567."), text("Really 7654321."))
    body = client.post(f"/datasets/{dataset_id}/chat", json={"message": "avg price?"}).json()
    assert body["fallback_used"] is True
    assert "1234567" not in body["assistant_message"]["content"]
    assert body["assistant_message"]["grounding"]["verified"] is False


def test_explain_kinds(client, dataset_id, use_llm, sales_df):
    use_llm.queue(text("The dataset has 200 rows."))
    r = client.post(f"/datasets/{dataset_id}/ai/explain", json={"kind": "dataset"})
    assert r.status_code == 200 and r.json()["grounding"]["verified"] and r.json()["facts"]["overview"]["rows"] == 200
    use_llm.queue(text("Data quality looks fine overall."))
    assert client.post(f"/datasets/{dataset_id}/ai/explain", json={"kind": "quality"}).status_code == 200
    use_llm.queue(text("Shown is a bar chart."))
    r = client.post(f"/datasets/{dataset_id}/ai/explain",
                    json={"kind": "chart", "chart": {"chart_type": "bar", "x_column": "region"}})
    assert r.status_code == 200 and "chart" in r.json()["facts"]


def test_explain_ml_run_uses_stored_metrics(client, dataset_id, use_llm):
    run = client.post(f"/datasets/{dataset_id}/ml/train",
                      json={"task": "regression", "algorithm": "ridge", "target": "sales"}).json()
    r2 = round(run["metrics"]["r2"], 2)
    use_llm.queue(text(f"The model explains about {r2} of variance (R2)."))
    r = client.post(f"/datasets/{dataset_id}/ai/explain", json={"kind": "ml_run", "run_id": run["id"]})
    assert r.status_code == 200 and r.json()["grounding"]["verified"]
    assert r.json()["facts"]["metrics"]["r2"] == run["metrics"]["r2"]


def test_explain_validation(client, dataset_id, use_llm):
    assert client.post(f"/datasets/{dataset_id}/ai/explain", json={"kind": "nonsense"}).status_code == 422
    assert client.post(f"/datasets/{dataset_id}/ai/explain", json={"kind": "ml_run"}).status_code == 400
    assert client.post(f"/datasets/{dataset_id}/ai/explain", json={"kind": "ml_run", "run_id": 9}).status_code == 404
    assert client.post(f"/datasets/{dataset_id}/ai/explain", json={"kind": "chart"}).status_code == 400


def test_explain_ungrounded_text_is_flagged(client, dataset_id, use_llm):
    use_llm.queue(text("There are 31337 rows."), text("Still 31337 rows."))
    body = client.post(f"/datasets/{dataset_id}/ai/explain", json={"kind": "dataset"}).json()
    assert body["grounding"]["verified"] is False and "31337" in body["grounding"]["unverified_numbers"]
