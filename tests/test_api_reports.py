import io

import pytest
from pypdf import PdfReader

from src.app.services.ai.errors import AIServiceError
from tests.helpers import text


def pdf_text(content: bytes) -> str:
    return "\n".join(p.extract_text() for p in PdfReader(io.BytesIO(content)).pages)


def test_report_without_ai_contains_real_dataset_results(client, dataset_id, sales_df):
    r = client.post(f"/datasets/{dataset_id}/reports", json={"title": "My Sales Report"})
    assert r.status_code == 200, r.text
    meta = r.json()
    assert meta["ai_included"] is False and meta["size_bytes"] > 5000
    pdf = client.get(meta["download_url"])
    assert pdf.status_code == 200 and pdf.headers["content-type"] == "application/pdf"
    assert pdf.content.startswith(b"%PDF")
    body = pdf_text(pdf.content)
    for expected in ["My Sales Report", "sales.csv", "200 rows", "Data quality findings",
                     "Exploratory data analysis", "Limitations", "Recommendations"]:
        assert expected in body, expected
    assert f"{int(sales_df['age'].isna().sum())}" in body            # real missing count
    assert len(PdfReader(io.BytesIO(pdf.content)).pages) >= 3
    assert client.get(f"/datasets/{dataset_id}/reports").json()[0]["id"] == meta["id"]


def test_report_includes_trained_models(client, dataset_id):
    run = client.post(f"/datasets/{dataset_id}/ml/train",
                      json={"task": "regression", "algorithm": "ridge", "target": "units"}).json()
    meta = client.post(f"/datasets/{dataset_id}/reports", json={"run_ids": [run["id"]]}).json()
    assert meta["sections"]["models"] == 1
    body = pdf_text(client.get(meta["download_url"]).content)
    assert "Machine learning results" in body and run["model_name"] in body
    assert f"{run['metrics']['rmse']:.4g}"[:4] in body.replace(",", "")
    assert client.post(f"/datasets/{dataset_id}/reports", json={"run_ids": [999]}).status_code == 404


def test_report_with_gemini_explanations(client, dataset_id, use_llm):
    use_llm.queue(text("**Summary:** The dataset contains 200 rows.\n* Check missing values."),
                  text("Quality is acceptable."))
    meta = client.post(f"/datasets/{dataset_id}/reports", json={"include_ai": True}).json()
    assert meta["ai_included"] is True
    body = pdf_text(client.get(meta["download_url"]).content)
    assert "Google Gemini" in body and "The dataset contains 200 rows" in body
    assert "Numeric grounding check" in body


def test_report_survives_gemini_outage(client, dataset_id, use_llm):
    use_llm.queue(AIServiceError("quota", "Gemini quota or rate limit reached."))
    r = client.post(f"/datasets/{dataset_id}/reports", json={"include_ai": True})
    assert r.status_code == 200 and r.json()["ai_included"] is False
    body = pdf_text(client.get(r.json()["download_url"]).content)
    assert "quota" in body.lower()


def test_report_without_api_key_still_generates(client, dataset_id):
    r = client.post(f"/datasets/{dataset_id}/reports", json={"include_ai": True})
    assert r.status_code == 200 and r.json()["ai_included"] is False
    assert "GEMINI_API_KEY" in pdf_text(client.get(r.json()["download_url"]).content)


def test_report_delete_and_not_found(client, dataset_id):
    meta = client.post(f"/datasets/{dataset_id}/reports", json={}).json()
    assert client.delete(f"/reports/{meta['id']}").status_code == 200
    assert client.get(f"/reports/{meta['id']}/download").status_code == 404
    assert client.post("/datasets/999/reports", json={}).status_code == 404


def test_report_path_traversal_is_blocked(client, dataset_id, env):
    from src.app.models.report import Report
    meta = client.post(f"/datasets/{dataset_id}/reports", json={}).json()
    outside = env["tmp"] / "secret.pdf"
    outside.write_bytes(b"%PDF secret")
    with env["Session"]() as db:
        db.get(Report, meta["id"]).file_path = str(outside)
        db.commit()
    assert client.get(f"/reports/{meta['id']}/download").status_code == 404


def test_report_for_tiny_and_text_only_datasets(client):
    for name, data in [("tiny.csv", b"a,b\n1,x\n2,y\n3,x\n"), ("text.csv", b"w\nfoo\nbar\nbaz\n")]:
        did = client.post("/datasets/upload", files={"file": (name, data, "text/csv")}).json()["id"]
        r = client.post(f"/datasets/{did}/reports", json={})
        assert r.status_code == 200, r.text
