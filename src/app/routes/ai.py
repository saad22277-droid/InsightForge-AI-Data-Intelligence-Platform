import logging
import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.app.config import get_settings
from src.app import metrics
from src.app.database import get_db
from src.app.models.ml_run import MLRun
from src.app.routes.common import get_dataset_or_404, load_df_or_http
from src.app.schemas.chat import ChatRequest, ExplainRequest
from src.app.services import charts as chart_service
from src.app.services.ai import conversations as conv_service
from src.app.services.ai.errors import AIServiceError
from src.app.services.ai.explain import build_facts, explain
from src.app.services.ai.llm import LLMClient, Turn
from src.app.services.ai.orchestrator import answer_question
from src.app.services.ai.provider import get_llm
from src.app.services.ml.service import run_to_dict

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ai"])

_last_error: dict | None = None


def _remember_error(exc: AIServiceError) -> None:
    global _last_error
    _last_error = {"kind": exc.kind, "message": exc.message, "at": time.time()}


@router.get("/ai/status")
def ai_status(check: bool = False):
    """AI service status. ``check=true`` makes one tiny live Gemini call."""
    settings = get_settings()
    status = {
        "provider": "google-gemini",
        "configured": settings.ai_configured,
        "model": settings.gemini_model,
        "limits": {
            "max_tool_calls_per_question": settings.ai_max_tool_calls,
            "requests_per_minute": settings.ai_requests_per_minute,
            "max_concurrent": settings.ai_max_concurrent,
            "timeout_seconds": settings.gemini_timeout_seconds,
            "retries": settings.gemini_max_retries,
        },
        "free_tier_notice": (
            "Gemini free-tier quotas are limited, can change, and free-tier prompts may be "
            "used by Google to improve its products. Only aggregated results and short value "
            "samples are sent; the full dataset never is."),
        "last_error": _last_error,
        "reachable": None,
    }
    if check and settings.ai_configured:
        try:
            llm = get_llm()
            started = time.perf_counter()
            llm.generate("Reply with the single word OK.", [Turn(role="user", text="ping")], None)
            status["reachable"] = True
            status["latency_ms"] = int((time.perf_counter() - started) * 1000)
        except AIServiceError as exc:
            _remember_error(exc)
            status["reachable"] = False
            status["error"] = {"kind": exc.kind, "message": exc.message}
            status["last_error"] = _last_error
    return status


@router.post("/datasets/{dataset_id}/chat")
def chat(dataset_id: int, body: ChatRequest, db: Session = Depends(get_db),
         llm: LLMClient = Depends(get_llm)):
    settings = get_settings()
    dataset = get_dataset_or_404(db, dataset_id)
    df = load_df_or_http(dataset)

    conversation = None
    history: list[tuple[str, str]] = []
    if body.conversation_id is not None:
        conversation = conv_service.get_conversation(db, dataset_id, body.conversation_id)
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        history = conv_service.history_for_llm(conversation, settings.ai_max_history_messages)

    try:
        result = answer_question(llm, df, dataset.filename, body.message, history, settings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except AIServiceError as exc:
        _remember_error(exc)
        raise

    if conversation is None:
        conversation = conv_service.create_conversation(db, dataset_id, body.message)
    user, assistant = conv_service.save_exchange(db, conversation, body.message, result)
    return {
        "conversation_id": conversation.id,
        "title": conversation.title,
        "user_message": conv_service.message_to_dict(user),
        "assistant_message": conv_service.message_to_dict(assistant),
        "tool_calls_used": result.tool_calls_used,
        "fallback_used": result.fallback_used,
        "usage": result.usage,
    }


@router.get("/datasets/{dataset_id}/conversations")
def list_conversations(dataset_id: int, db: Session = Depends(get_db)):
    get_dataset_or_404(db, dataset_id)
    return [conv_service.conversation_to_dict(c)
            for c in conv_service.list_conversations(db, dataset_id)]


@router.get("/datasets/{dataset_id}/conversations/{conversation_id}")
def get_conversation(dataset_id: int, conversation_id: int, db: Session = Depends(get_db)):
    get_dataset_or_404(db, dataset_id)
    conversation = conv_service.get_conversation(db, dataset_id, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv_service.conversation_to_dict(conversation, with_messages=True)


@router.delete("/datasets/{dataset_id}/conversations/{conversation_id}")
def delete_conversation(dataset_id: int, conversation_id: int, db: Session = Depends(get_db)):
    get_dataset_or_404(db, dataset_id)
    conversation = conv_service.get_conversation(db, dataset_id, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.delete(conversation)
    db.commit()
    return {"message": "Conversation deleted", "id": conversation_id}


@router.post("/datasets/{dataset_id}/ai/explain")
def explain_endpoint(dataset_id: int, body: ExplainRequest, db: Session = Depends(get_db),
                     llm: LLMClient = Depends(get_llm)):
    dataset = get_dataset_or_404(db, dataset_id)
    df = load_df_or_http(dataset)
    run_dict = chart = None
    if body.kind == "ml_run":
        if body.run_id is None:
            raise HTTPException(status_code=400, detail="run_id is required for ml_run")
        run = db.query(MLRun).filter(MLRun.id == body.run_id,
                                     MLRun.dataset_id == dataset_id).first()
        if run is None:
            raise HTTPException(status_code=404, detail="Model run not found")
        run_dict = run_to_dict(run)
    if body.kind == "chart":
        if body.chart is None:
            raise HTTPException(status_code=400, detail="chart is required for kind=chart")
        try:
            chart = chart_service.generate_chart(
                df, body.chart.chart_type, body.chart.x_column, body.chart.y_column,
                body.chart.group_by, body.chart.agg, body.chart.bins)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
    try:
        facts = build_facts(body.kind, df, dataset.filename, run=run_dict, chart=chart)
        return explain(llm, body.kind, facts, body.focus)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except AIServiceError as exc:
        _remember_error(exc)
        metrics.incr("ai_errors")
        raise
