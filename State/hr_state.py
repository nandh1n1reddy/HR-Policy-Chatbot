from __future__ import annotations

from typing import Any, TypedDict


class HRState(TypedDict, total=False):
    session_id: str
    query: str
    intent: str
    intent_reason: str
    in_scope: bool
    retrieved_chunks: list[dict[str, Any]]
    draft_answer: str
    answer: str
    confidence: float
    confidence_reason: str
    escalated: bool
    needs_review: bool
    status: str
    review_id: str
    reviewer_id: str
    reviewer_note: str
    reviewer_action: str
    source_citations: list[dict[str, Any]]
    history: list[dict[str, Any]]
    audit_entry: dict[str, Any]
    local_tokens: int
    cloud_tokens: int
    raw_model_output: str
    token_source: str
