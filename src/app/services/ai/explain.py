"""AI explanations of already-computed analytical results ("FACTS")."""
from __future__ import annotations

import json
import logging
import time

import pandas as pd

from src.app import metrics
from src.app.services import cleaning as cleaning_service
from src.app.services import eda as eda_service
from src.app.services.ai import prompts
from src.app.services.ai.errors import AIServiceError
from src.app.services.ai.grounding import verify_numbers
from src.app.services.ai.guard import ai_request_slot
from src.app.services.ai.llm import LLMClient, Turn
from src.app.services.ai.tools import sanitize

logger = logging.getLogger(__name__)
MAX_FACT_CHARS = 14000
KINDS = {"dataset", "quality", "eda", "correlations", "ml_run", "chart", "insights"}


def _trim(obj, depth=0):
    """Keep FACTS compact: cap list lengths."""
    if isinstance(obj, dict):
        return {k: _trim(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_trim(v, depth + 1) for v in obj[:15]]
    return obj


def build_facts(kind: str, df: pd.DataFrame | None, filename: str,
                run: dict | None = None, chart: dict | None = None) -> dict:
    if kind == "ml_run":
        if run is None:
            raise ValueError("ml_run explanations need a run")
        details = run.get("details") or {}
        return {
            "model": {k: run.get(k) for k in ("model_name", "version", "task", "algorithm",
                                               "target", "primary_metric")},
            "metrics": run.get("metrics"),
            "features_used": run.get("features"),
            "feature_importance": details.get("feature_importance"),
            "split": details.get("split"),
            "excluded_features": details.get("excluded_features"),
            "warnings": details.get("warnings"),
            "cluster_profiles": details.get("cluster_profiles"),
            "confusion_matrix": details.get("confusion_matrix"),
            "classes": details.get("classes"),
            "note": "baseline_* metrics come from a naive model; compare against them.",
        }
    if kind == "chart":
        if chart is None:
            raise ValueError("chart explanations need chart data")
        return {"chart": chart}
    if df is None:
        raise ValueError("A dataset is required")

    eda = eda_service.generate_eda(df)
    base = {"filename": filename, "overview": {
        "rows": eda["overview"]["row_count"], "columns": eda["overview"]["column_count"],
        "numeric_columns": eda["overview"]["numeric_columns"],
        "categorical_columns": eda["overview"]["categorical_columns"]}}
    quality = {k: eda["quality"][k] for k in (
        "quality_score", "score_components", "missing_cells", "duplicate_rows",
        "empty_columns", "warnings", "recommendations")}
    top_missing = [c for c in eda["missing"]["columns"] if c["missing"]][:8]

    if kind == "dataset":
        return {**base, "missing": top_missing,
                "numeric_statistics": eda["numeric_statistics"],
                "categorical_top_values": {
                    c: v["top_values"][:3] for c, v in eda["categorical_statistics"].items()}}
    if kind == "quality":
        recs = cleaning_service.analyze_cleaning(df)["recommendations"]
        return {**base, "quality": quality, "missing_by_column": top_missing,
                "duplicates": eda["duplicates"],
                "outlier_summary": {c: {"count": v["outlier_count"],
                                        "percentage": v["outlier_percentage"]}
                                    for c, v in eda["outliers"]["columns"].items()
                                    if v["outlier_count"]},
                "column_insights": eda["column_insights"][:15],
                "cleaning_recommendations": [r["title"] + ": " + r["reason"] for r in recs[:10]]}
    if kind == "correlations":
        return {**base, "top_correlations": eda["correlations"]["top_pairs"]}
    # eda / insights
    return {**base, "quality": quality, "missing_by_column": top_missing,
            "numeric_statistics": {c: {k: v[k] for k in ("mean", "median", "std", "min", "max", "skewness")}
                                   for c, v in eda["numeric_statistics"].items()},
            "top_correlations": eda["correlations"]["top_pairs"][:6],
            "outlier_summary": {c: {"count": v["outlier_count"], "percentage": v["outlier_percentage"]}
                                for c, v in eda["outliers"]["columns"].items() if v["outlier_count"]},
            "column_insights": eda["column_insights"][:12]}


def explain(llm: LLMClient, kind: str, facts: dict, focus: str | None = None) -> dict:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of: {', '.join(sorted(KINDS))}")
    facts_json = json.dumps(sanitize(_trim(facts), 100), default=str, ensure_ascii=False)
    if len(facts_json) > MAX_FACT_CHARS:
        facts_json = facts_json[:MAX_FACT_CHARS]
    task = {
        "dataset": "Explain the main characteristics of this dataset.",
        "quality": "Explain the data quality problems and how to address them.",
        "eda": "Summarize the most important exploratory-analysis findings.",
        "insights": "Generate the most useful insights and actionable recommendations.",
        "correlations": "Explain the strongest relationships between numeric columns.",
        "ml_run": "Explain these machine learning results to a non-expert, including whether "
                  "the model is meaningfully better than the baseline and its limitations.",
        "chart": "Explain what this chart shows.",
    }[kind]
    if focus:
        task += f" Focus: {focus[:200]}"
    prompt = f"{task}\n\nFACTS:\n{facts_json}"

    started = time.perf_counter()
    with ai_request_slot():
        metrics.incr("ai_explain_requests")
        turns = [Turn(role="user", text=prompt)]
        response = llm.generate(prompts.EXPLAIN_SYSTEM_PROMPT, turns, None)
        text = response.text.strip()
        grounding = verify_numbers(text, facts, [focus or ""])
        if not grounding["verified"]:
            metrics.incr("ai_grounding_retries")
            turns += [Turn(role="model", text=text),
                      Turn(role="user", text=prompts.RETRY_NUMBERS_MESSAGE.format(
                          numbers=", ".join(grounding["unverified_numbers"][:8])))]
            retry = llm.generate(prompts.EXPLAIN_SYSTEM_PROMPT, turns, None)
            text = retry.text.strip()
            grounding = verify_numbers(text, facts, [focus or ""])
    if not text:
        raise AIServiceError("empty", "Gemini returned an empty explanation.")
    if not grounding["verified"]:
        metrics.incr("ai_grounding_failures")
    return {
        "kind": kind,
        "explanation": text,
        "grounding": grounding,
        "model": llm.model_name,
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "facts": facts,
    }
