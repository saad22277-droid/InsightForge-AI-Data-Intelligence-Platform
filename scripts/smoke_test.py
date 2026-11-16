"""End-to-end smoke test against a RUNNING backend (real database, real files).

    python scripts/smoke_test.py                       # http://127.0.0.1:8000
    python scripts/smoke_test.py --base-url http://127.0.0.1:8000 --keep

Exercises: upload -> EDA -> charts -> cleaning (preview/apply/rollback) -> ML
(regression, classification, clustering, compare, predict, reproduce) -> experiments
-> PDF report -> AI status. Gemini calls are only made if GEMINI_API_KEY is configured
on the server (--skip-ai to avoid spending quota). Exits non-zero on any failure.
"""
import argparse
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--csv", default=str(ROOT / "data" / "samples" / "sales_demo.csv"))
    parser.add_argument("--keep", action="store_true", help="do not delete the dataset afterwards")
    parser.add_argument("--skip-ai", action="store_true", help="never call Gemini")
    args = parser.parse_args()
    http = httpx.Client(base_url=args.base_url, timeout=180)

    r = http.get("/health")
    check("health endpoint", r.status_code == 200 and r.json() == {"status": "healthy"})
    r = http.get("/health/ready")
    check("readiness (database + storage)", r.status_code == 200, str(r.json().get("checks")))

    data = Path(args.csv).read_bytes()
    r = http.post("/datasets/upload", files={"file": ("sales_demo.csv", data, "text/csv")})
    check("dataset upload", r.status_code == 200, r.text[:100])
    if r.status_code != 200:
        return 1
    did = r.json()["id"]

    try:
        meta = http.get(f"/datasets/{did}/metadata").json()
        check("metadata", meta["row_count"] > 700 and meta["active_version"] == 1,
              f"{meta['row_count']} rows, sha256 {meta['sha256'][:10]}...")

        eda = http.get(f"/datasets/{did}/eda").json()
        check("EDA computed from data", eda["overview"]["row_count"] == meta["row_count"]
              and eda["duplicates"]["duplicate_rows"] >= 12 and eda["outliers"]["total_outliers"] > 0,
              f"quality {eda['quality']['quality_score']}, dupes {eda['duplicates']['duplicate_rows']}, "
              f"outliers {eda['outliers']['total_outliers']}")

        for chart in ["histogram:Sales", "bar:Region", "scatter:Units", "box:Sales", "heatmap:", "missing:"]:
            kind, col = chart.split(":")
            params = {"chart_type": kind}
            if col:
                params["x_column"] = col
            if kind == "scatter":
                params["y_column"] = "Sales"
            r = http.get(f"/datasets/{did}/charts", params=params)
            check(f"chart: {kind}", r.status_code == 200, "" if r.status_code == 200 else r.text[:120])

        recs = http.get(f"/datasets/{did}/cleaning/recommendations").json()
        check("cleaning recommendations", recs["recommendation_count"] >= 4,
              f"{recs['recommendation_count']} found")
        ops = [x["operation"] for x in recs["recommendations"]]
        prev = http.post(f"/datasets/{did}/cleaning/preview", json={"operations": ops})
        check("cleaning preview (no change saved)", prev.status_code == 200
              and len(http.get(f"/datasets/{did}/versions").json()["versions"]) == 1,
              str(prev.json().get("delta")) if prev.status_code == 200 else prev.text[:150])
        applied = http.post(f"/datasets/{did}/cleaning/apply", json={"operations": ops, "note": "smoke test"})
        check("cleaning apply -> new version", applied.status_code == 200 and applied.json()["new_version"] == 2,
              applied.text[:100] if applied.status_code != 200 else "")
        after = http.get(f"/datasets/{did}/eda").json()
        check("cleaned data has fewer problems",
              after["missing"]["total_missing_cells"] < eda["missing"]["total_missing_cells"]
              and after["duplicates"]["duplicate_rows"] == 0,
              f"missing {eda['missing']['total_missing_cells']} -> {after['missing']['total_missing_cells']}")

        cols = [c["name"] for c in http.get(f"/datasets/{did}/metadata").json()["columns"]]
        sales = next(c for c in cols if c.lower() == "sales")
        profit = next(c for c in cols if c.lower() == "profit")
        reg = http.post(f"/datasets/{did}/ml/train", json={
            "task": "regression", "algorithm": "random_forest", "target": profit,
            "hyperparameters": {"n_estimators": 60}, "model_name": "smoke-profit"})
        ok = reg.status_code == 200
        check("ML regression trains with real metrics", ok,
              f"R2={reg.json()['metrics']['r2']:.3f} (baseline {reg.json()['metrics']['baseline_r2']:.3f})" if ok else reg.text[:200])
        clf = http.post(f"/datasets/{did}/ml/train", json={
            "task": "classification", "algorithm": "gradient_boosting", "target": next(c for c in cols if c.lower() == "category"),
            "hyperparameters": {"n_estimators": 40}})
        check("ML classification trains", clf.status_code == 200,
              f"accuracy={clf.json()['metrics']['accuracy']:.3f}" if clf.status_code == 200 else clf.text[:200])
        clu = http.post(f"/datasets/{did}/ml/train", json={"task": "clustering", "algorithm": "kmeans"})
        check("ML clustering trains", clu.status_code == 200,
              f"k={clu.json()['metrics']['n_clusters']}, silhouette={clu.json()['metrics']['silhouette']:.3f}" if clu.status_code == 200 else clu.text[:200])
        cmp_ = http.post(f"/datasets/{did}/ml/compare", json={
            "task": "regression", "target": sales, "algorithms": ["linear_regression", "ridge", "random_forest"]})
        check("model comparison ranks algorithms", cmp_.status_code == 200 and len(cmp_.json()["runs"]) == 3)
        if reg.status_code == 200:
            rid = reg.json()["id"]
            pred = http.post(f"/ml/runs/{rid}/predict", json={"records": [{"Units": 5, "Unit Price": 100, "Region": "North"}]})
            check("prediction from saved model", pred.status_code == 200, str(pred.json()["predictions"]) if pred.status_code == 200 else pred.text[:150])
            rep = http.post(f"/ml/runs/{rid}/reproduce")
            check("run reproduces identical metrics", rep.status_code == 200 and rep.json()["metrics_identical"])
            reproduced_version = rep.json()["new_run"]["version"]
            v = http.post(f"/datasets/{did}/ml/train", json={"task": "regression", "algorithm": "ridge", "target": profit, "model_name": "smoke-profit"})
            check("model versioning increments (v1 -> reproduce v2 -> new v3)",
                  v.status_code == 200 and reproduced_version == 2 and v.json()["version"] == 3)

        exps = http.get("/experiments").json()
        check("experiment tracker lists runs", len(exps) >= 6, f"{len(exps)} runs")
        mine = [e["id"] for e in exps if e["task"] == "regression" and e["dataset_id"] == did][:3]
        c = http.get("/experiments/compare", params={"ids": ",".join(map(str, mine))})
        check("experiment comparison", c.status_code == 200 and "r2" in c.json()["best_by_metric"],
              c.text[:100] if c.status_code != 200 else f"best r2 = run #{c.json()['best_by_metric']['r2']}")

        ai = http.get("/ai/status").json()
        check("AI status endpoint", "configured" in ai, f"configured={ai['configured']} model={ai['model']}")
        use_ai = ai["configured"] and not args.skip_ai
        if use_ai:
            t = http.get("/ai/status", params={"check": True}).json()
            check("Gemini live connection", t.get("reachable") is True, str(t.get("error") or f"{t.get('latency_ms')} ms"))
            if t.get("reachable"):
                chat = http.post(f"/datasets/{did}/chat", json={"message": f"What is the average {sales}?"})
                body = chat.json() if chat.status_code == 200 else {}
                check("dataset chat answers (live Gemini)", chat.status_code == 200 and body["assistant_message"]["grounding"]["verified"],
                      body.get("assistant_message", {}).get("content", chat.text)[:140])
                follow = http.post(f"/datasets/{did}/chat", json={"message": "And the median?", "conversation_id": body.get("conversation_id")})
                check("conversation persists across turns", follow.status_code == 200 and
                      len(http.get(f"/datasets/{did}/conversations/{body['conversation_id']}").json()["messages"]) == 4)
        else:
            print("[SKIP] Gemini chat/explain (no GEMINI_API_KEY on server, or --skip-ai)")

        rep = http.post(f"/datasets/{did}/reports", json={"include_ai": bool(use_ai)})
        ok = rep.status_code == 200
        check("PDF report generated", ok, rep.text[:150] if not ok else f"{rep.json()['size_bytes']} bytes, ai={rep.json()['ai_included']}")
        if ok:
            pdf = http.get(rep.json()["download_url"])
            out = ROOT / "data" / "reports" / "_smoke_report.pdf"
            out.write_bytes(pdf.content)
            check("report downloads as valid PDF", pdf.content.startswith(b"%PDF") and len(pdf.content) > 20000, str(out))

        r = http.post(f"/datasets/{did}/versions/1/activate")
        check("rollback to original version", r.status_code == 200)
        check("system metrics endpoint", http.get("/metrics").json()["counters"]["http_requests"] > 20)
    finally:
        if not args.keep:
            http.delete(f"/datasets/{did}")
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed" + (f"; FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    t0 = time.time()
    code = main()
    print(f"({time.time() - t0:.1f}s)")
    sys.exit(code)
