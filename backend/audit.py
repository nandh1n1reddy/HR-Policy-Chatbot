from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from backend.config import AUDIT_DB_PATH


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(str(AUDIT_DB_PATH))
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id TEXT PRIMARY KEY,
                session_id TEXT,
                user_query TEXT NOT NULL,
                intent TEXT,
                in_scope INTEGER,
                status TEXT NOT NULL,
                answer TEXT,
                draft_answer TEXT,
                confidence REAL,
                confidence_reason TEXT,
                escalated INTEGER DEFAULT 0,
                reviewer_id TEXT,
                reviewer_action TEXT,
                reviewer_note TEXT,
                source_citations TEXT,
                retrieved_chunks TEXT,
                local_tokens INTEGER DEFAULT 0,
                cloud_tokens INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_interactions_status ON interactions(status)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_interactions_session ON interactions(session_id)"
        )
        connection.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, default=str)


def log_interaction(payload: dict[str, Any]) -> str:
    init_db()
    interaction_id = payload.get("interaction_id")
    if not interaction_id:
        raise ValueError("interaction_id is required")

    record = {
        "interaction_id": interaction_id,
        "session_id": payload.get("session_id"),
        "user_query": payload.get("user_query", ""),
        "intent": payload.get("intent"),
        "in_scope": 1 if payload.get("in_scope") else 0,
        "status": payload.get("status", "logged"),
        "answer": payload.get("answer"),
        "draft_answer": payload.get("draft_answer"),
        "confidence": payload.get("confidence"),
        "confidence_reason": payload.get("confidence_reason"),
        "escalated": 1 if payload.get("escalated") else 0,
        "reviewer_id": payload.get("reviewer_id"),
        "reviewer_action": payload.get("reviewer_action"),
        "reviewer_note": payload.get("reviewer_note"),
        "source_citations": _serialize(payload.get("source_citations", [])),
        "retrieved_chunks": _serialize(payload.get("retrieved_chunks", [])),
        "local_tokens": int(payload.get("local_tokens", 0)),
        "cloud_tokens": int(payload.get("cloud_tokens", 0)),
        "created_at": payload.get("created_at", _now()),
        "updated_at": payload.get("updated_at", _now()),
    }

    with _connect() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO interactions (
                interaction_id, session_id, user_query, intent, in_scope, status,
                answer, draft_answer, confidence, confidence_reason, escalated,
                reviewer_id, reviewer_action, reviewer_note, source_citations,
                retrieved_chunks, local_tokens, cloud_tokens, created_at, updated_at
            ) VALUES (
                :interaction_id, :session_id, :user_query, :intent, :in_scope, :status,
                :answer, :draft_answer, :confidence, :confidence_reason, :escalated,
                :reviewer_id, :reviewer_action, :reviewer_note, :source_citations,
                :retrieved_chunks, :local_tokens, :cloud_tokens, :created_at, :updated_at
            )
            """,
            record,
        )
        connection.commit()
    return interaction_id


def update_interaction(interaction_id: str, **updates: Any) -> None:
    if not updates:
        return

    init_db()
    updates["updated_at"] = _now()
    assignments = ", ".join(f"{field} = ?" for field in updates)
    values: list[Any] = []

    for field, value in updates.items():
        if field in {"source_citations", "retrieved_chunks"}:
            values.append(_serialize(value))
        elif field in {"in_scope", "escalated"}:
            values.append(1 if value else 0)
        else:
            values.append(value)

    values.append(interaction_id)

    with _connect() as connection:
        connection.execute(
            f"UPDATE interactions SET {assignments} WHERE interaction_id = ?",
            values,
        )
        connection.commit()


def fetch_pending_reviews(limit: int = 25) -> list[dict[str, Any]]:
    init_db()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT * FROM interactions
            WHERE status = 'pending_review'
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def fetch_recent_interactions(limit: int = 50) -> list[dict[str, Any]]:
    init_db()
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT * FROM interactions
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def fetch_interaction(interaction_id: str) -> dict[str, Any] | None:
    init_db()
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM interactions WHERE interaction_id = ?",
            (interaction_id,),
        ).fetchone()
    return dict(row) if row else None
