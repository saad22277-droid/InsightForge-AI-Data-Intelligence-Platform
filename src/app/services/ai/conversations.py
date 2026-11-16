"""Persistence of chat conversations in PostgreSQL."""
from __future__ import annotations

from sqlalchemy.orm import Session

from src.app.models.chat import ChatConversation, ChatMessage, _now
from src.app.services.ai.orchestrator import ChatResult


def message_to_dict(message: ChatMessage) -> dict:
    return {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "evidence": message.evidence,
        "grounding": message.grounding,
        "model": message.model,
        "latency_ms": message.latency_ms,
        "created_at": message.created_at.isoformat() if message.created_at else None,
    }


def conversation_to_dict(conv: ChatConversation, with_messages: bool = False) -> dict:
    data = {
        "id": conv.id,
        "dataset_id": conv.dataset_id,
        "title": conv.title,
        "created_at": conv.created_at.isoformat() if conv.created_at else None,
        "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
    }
    if with_messages:
        data["messages"] = [message_to_dict(m) for m in conv.messages]
    else:
        data["message_count"] = len(conv.messages)
    return data


def get_conversation(db: Session, dataset_id: int, conversation_id: int) -> ChatConversation | None:
    return (
        db.query(ChatConversation)
        .filter(ChatConversation.id == conversation_id,
                ChatConversation.dataset_id == dataset_id)
        .first()
    )


def list_conversations(db: Session, dataset_id: int) -> list[ChatConversation]:
    return (
        db.query(ChatConversation)
        .filter(ChatConversation.dataset_id == dataset_id)
        .order_by(ChatConversation.updated_at.desc(), ChatConversation.id.desc())
        .all()
    )


def history_for_llm(conv: ChatConversation, limit: int) -> list[tuple[str, str]]:
    """Only plain user/assistant text is replayed. Tool results are re-computed per
    question so answers always reflect the current dataset version."""
    pairs = [(m.role, m.content) for m in conv.messages if m.role in ("user", "assistant")]
    return pairs[-limit:]


def create_conversation(db: Session, dataset_id: int, first_question: str) -> ChatConversation:
    title = " ".join(first_question.split())[:60] or "New conversation"
    conv = ChatConversation(dataset_id=dataset_id, title=title)
    db.add(conv)
    db.flush()
    return conv


def save_exchange(db: Session, conv: ChatConversation, question: str,
                  result: ChatResult) -> tuple[ChatMessage, ChatMessage]:
    user = ChatMessage(conversation_id=conv.id, role="user", content=question)
    assistant = ChatMessage(
        conversation_id=conv.id, role="assistant", content=result.answer,
        evidence=result.evidence, grounding=result.grounding,
        model=result.model, latency_ms=result.latency_ms)
    db.add_all([user, assistant])
    conv.updated_at = _now()
    db.commit()
    db.refresh(user)
    db.refresh(assistant)
    return user, assistant
