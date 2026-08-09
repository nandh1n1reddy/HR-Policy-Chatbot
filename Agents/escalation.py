from __future__ import annotations

import uuid

from backend.audit import log_interaction, update_interaction


def queue_for_review(state):
    interaction_id = state.get("review_id") or uuid.uuid4().hex
    state["review_id"] = interaction_id
    state["escalated"] = True
    state["status"] = "pending_review"

    log_interaction(
        {
            "interaction_id": interaction_id,
            "session_id": state.get("session_id"),
            "user_query": state.get("query", ""),
            "intent": state.get("intent"),
            "in_scope": state.get("in_scope"),
            "status": "pending_review",
            "draft_answer": state.get("draft_answer"),
            "confidence": state.get("confidence"),
            "confidence_reason": state.get("confidence_reason"),
            "escalated": True,
            "source_citations": state.get("source_citations", []),
            "retrieved_chunks": state.get("retrieved_chunks", []),
            "local_tokens": state.get("local_tokens", 0),
            "cloud_tokens": state.get("cloud_tokens", 0),
        }
    )
    return state


def decline_query(state):
    interaction_id = state.get("audit_entry", {}).get("interaction_id") or uuid.uuid4().hex
    answer = "I can only help with HR policy questions. Please contact IT support or your department head for other inquiries."

    state["answer"] = answer
    state["status"] = "declined"
    state["escalated"] = False
    state["review_id"] = interaction_id
    state["audit_entry"] = {"interaction_id": interaction_id}

    log_interaction(
        {
            "interaction_id": interaction_id,
            "session_id": state.get("session_id"),
            "user_query": state.get("query", ""),
            "intent": state.get("intent", "general_hr"),
            "in_scope": False,
            "status": "declined",
            "answer": answer,
            "escalated": False,
            "local_tokens": state.get("local_tokens", 0),
            "cloud_tokens": state.get("cloud_tokens", 0),
        }
    )
    return state


def finalize_answer(state):
    interaction_id = state.get("audit_entry", {}).get("interaction_id") or uuid.uuid4().hex
    citations = state.get("source_citations", [])
    answer = state.get("answer", "").strip()

    if citations:
        citation_lines = []
        for citation in citations:
            line = f"- {citation.get('source')} :: {citation.get('heading')}"
            if citation.get("page_number"):
                line += f" (page {citation['page_number']})"
            citation_lines.append(line)
        answer = f"{answer}\n\nSources:\n" + "\n".join(citation_lines)

    state["answer"] = answer
    state["status"] = "answered"
    state["escalated"] = False
    state["audit_entry"] = {"interaction_id": interaction_id}

    log_interaction(
        {
            "interaction_id": interaction_id,
            "session_id": state.get("session_id"),
            "user_query": state.get("query", ""),
            "intent": state.get("intent"),
            "in_scope": state.get("in_scope"),
            "status": "answered",
            "answer": answer,
            "draft_answer": state.get("draft_answer"),
            "confidence": state.get("confidence"),
            "confidence_reason": state.get("confidence_reason"),
            "escalated": False,
            "source_citations": citations,
            "retrieved_chunks": state.get("retrieved_chunks", []),
            "local_tokens": state.get("local_tokens", 0),
            "cloud_tokens": state.get("cloud_tokens", 0),
        }
    )
    return state


def resolve_review(interaction_id, reviewer_id, reviewer_action, reviewer_note, revised_answer=None):
    update_interaction(
        interaction_id,
        status=reviewer_action,
        reviewer_id=reviewer_id,
        reviewer_action=reviewer_action,
        reviewer_note=reviewer_note,
        answer=revised_answer,
        escalated=False,
    )
