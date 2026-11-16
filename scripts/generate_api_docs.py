"""Generate docs/API.md from the FastAPI OpenAPI schema (so it never drifts from code).

Usage: python scripts/generate_api_docs.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("DATABASE_URL", "postgresql://docs:docs@localhost/docs")

from src.app.main import app  # noqa: E402

ORDER = ["datasets", "analysis", "cleaning", "machine-learning", "experiments", "ai", "reports", "system"]
TITLES = {
    "datasets": "Datasets (Stages 1-4, unchanged contracts)",
    "analysis": "Analysis, charts and exports",
    "cleaning": "Data cleaning and dataset versions",
    "machine-learning": "Machine learning studio",
    "experiments": "Experiment tracking",
    "ai": "Gemini AI: chat, explanations, status",
    "reports": "Reports",
    "system": "Health, metrics and system status",
}
schema = app.openapi()
routes = {}
for path, methods in schema["paths"].items():
    for method, op in methods.items():
        tag = (op.get("tags") or ["root"])[0]
        summary = (op.get("description") or op.get("summary") or "").strip().splitlines()[0]
        params = [p["name"] for p in op.get("parameters", []) if p["in"] == "query"]
        body = ""
        if "requestBody" in op:
            ref = (op["requestBody"]["content"].get("application/json", {}).get("schema", {}) or {}).get("$ref", "")
            body = ref.split("/")[-1] if ref else "multipart/form-data"
        routes.setdefault(tag, []).append((method.upper(), path, summary, body, params))

lines = [
    "# API reference",
    "",
    "Generated from the running application's OpenAPI schema (`python scripts/generate_api_docs.py`).",
    "Interactive documentation is always available at **http://127.0.0.1:8000/docs** (Swagger UI) and `/redoc`.",
    "",
    "Conventions: JSON in/out; errors use `{\"detail\": ...}`. AI errors use "
    "`{\"detail\": {\"code\": \"quota|rate_limited|timeout|auth|not_configured|...\", \"message\": \"...\"}}` "
    "with HTTP 429/502/503/504 and a `Retry-After` header where relevant. Every response carries `X-Request-ID`.",
    "",
]
for tag in ORDER + sorted(set(routes) - set(ORDER)):
    if tag not in routes:
        continue
    lines += [f"## {TITLES.get(tag, tag.title())}", "", "| Method | Path | Request body | Query parameters | Description |", "|---|---|---|---|---|"]
    for method, path, summary, body, params in sorted(routes[tag], key=lambda r: (r[1], r[0])):
        lines.append(f"| {method} | `{path}` | {('`' + body + '`') if body else ''} | {', '.join(params)} | {summary.replace('|', '/')} |")
    lines.append("")
out = Path(__file__).resolve().parents[1] / "docs" / "API.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"wrote {out} ({sum(len(v) for v in routes.values())} operations)")
