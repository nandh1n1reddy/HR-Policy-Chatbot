from __future__ import annotations

import json
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Agents.escalation import resolve_review
from Graph.workflow import run_query
from backend.audit import fetch_interaction, fetch_pending_reviews, fetch_recent_interactions
from backend.chunk_generator import ChunkGenerator
from backend.indexing import index_documents
from backend.ingest import _load_document
from backend.preprocess import TextPreprocessor
from backend.vector_store import VectorStore


st.set_page_config(
    page_title="HR Policy Compliance Q&A",
    layout="wide",
)


def _inject_styles():
    st.markdown(
        """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .block-container {
                padding-top: 1rem;
                padding-bottom: 2rem;
                max-width: 1440px;
            }
            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(circle at top left, rgba(239, 68, 68, 0.12), transparent 28%),
                    radial-gradient(circle at top right, rgba(59, 130, 246, 0.10), transparent 22%),
                    linear-gradient(180deg, #111318 0%, #0b0c10 100%);
            }
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #12141a 0%, #0f1116 100%);
                border-right: 1px solid rgba(255,255,255,0.08);
            }
            .hero-shell {
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 22px;
                padding: 1.15rem 1.35rem;
                background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
                box-shadow: 0 10px 40px rgba(0,0,0,0.25);
                margin-bottom: 1rem;
            }
            .hero-kicker {
                text-transform: uppercase;
                letter-spacing: 0.18em;
                font-size: 0.72rem;
                opacity: 0.72;
                margin-bottom: 0.35rem;
            }
            .hero-title {
                font-size: 2rem;
                font-weight: 700;
                margin: 0;
            }
            .hero-copy {
                color: rgba(255,255,255,0.72);
                margin-top: 0.35rem;
                margin-bottom: 0;
                max-width: 70ch;
            }
            .quick-chip {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                padding: 0.5rem 0.85rem;
                margin: 0.25rem 0.35rem 0.25rem 0;
                border-radius: 999px;
                border: 1px solid rgba(255,255,255,0.12);
                background: rgba(255,255,255,0.04);
                color: inherit;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_state():
    st.session_state.setdefault("session_id", uuid.uuid4().hex)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("local_tokens", 0)
    st.session_state.setdefault("cloud_tokens", 0)
    st.session_state.setdefault("last_query", "")
    # interaction_ids for questions that were escalated to HR and haven't
    # been resolved (or shown to the employee) yet.
    st.session_state.setdefault("pending_review_ids", [])


def _add_message(role: str, content: str):
    st.session_state.messages.append(
        {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


def _render_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def _check_resolved_reviews():
    """Look up any escalated questions this session is waiting on and, if HR
    has since resolved them, add the outcome as a new assistant message.

    This runs on every script rerun (main() always executes both the chat
    tab and the review tab's functions regardless of which tab is visually
    active), so submitting a review in the HR Review tab surfaces the result
    here on the very next rerun -- no manual refresh needed, as long as it's
    the same browser session.
    """
    pending_ids = st.session_state.get("pending_review_ids", [])
    if not pending_ids:
        return

    still_pending = []
    for interaction_id in pending_ids:
        record = fetch_interaction(interaction_id)
        if not record or record.get("status") == "pending_review":
            still_pending.append(interaction_id)
            continue

        query_text = record.get("user_query", "")
        if record.get("status") == "policy_gap":
            note = record.get("reviewer_note") or "This isn't addressed by current policy."
            _add_message(
                "assistant",
                f"HR reviewed your question — *\"{query_text}\"* — and found it isn't "
                f"covered by current policy. {note}",
            )
        else:
            answer = record.get("answer") or "HR reviewed this but did not leave a final answer."
            _add_message(
                "assistant",
                f"HR approved an answer to your question — *\"{query_text}\"*:\n\n{answer}",
            )

    st.session_state.pending_review_ids = still_pending


def _process_query(prompt: str):
    if not prompt.strip():
        return

    _add_message("user", prompt)
    with st.chat_message("assistant"):
        with st.spinner("Checking policy..."):
            result = run_query(
                query=prompt,
                session_id=st.session_state.session_id,
                history=st.session_state.messages,
            )

        st.session_state.local_tokens += int(result.get("local_tokens", 0) or 0)
        st.session_state.cloud_tokens += int(result.get("cloud_tokens", 0) or 0)
        st.session_state.last_query = prompt

        if result.get("status") == "pending_review":
            st.info("This question was escalated to HR review because the confidence score was below the threshold.")
            st.markdown("A draft answer was generated for review, but it is not shown to the employee until approved.")
            _add_message(
                "assistant",
                "Your question has been sent to HR review because the policy answer was not confident enough to publish automatically.",
            )
            review_id = result.get("review_id")
            if review_id and review_id not in st.session_state.pending_review_ids:
                st.session_state.pending_review_ids.append(review_id)
        else:
            answer = result.get("answer", "No answer was produced.")
            st.markdown(answer)
            _add_message("assistant", answer)


def _ingest_uploaded_file(uploaded_file, category: str = "Uploads") -> int:
    suffix = Path(uploaded_file.name).suffix or ".md"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = Path(temp_file.name)

    try:
        document = _load_document(temp_path, temp_path.parent)
        documents = [document] if document is not None else []
        for document in documents:
            document.category = category
            document.content = TextPreprocessor.preprocess(document.content)
            chunks = ChunkGenerator().create_chunks(document)
            VectorStore().add_chunks(chunks)
        return len(documents)
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


def employee_chat_tab():
    _check_resolved_reviews()

    st.subheader("Employee Chat")
    st.caption("Ask HR policy questions. Sensitive or ambiguous items are sent to human review.")

    st.markdown(
        """
        <div class="hero-shell">
            <div class="hero-kicker">Policy assistant</div>
            <h2 class="hero-title">HR policy answers, grounded in your document library.</h2>
            <p class="hero-copy">
                Use the quick prompts below or ask your own question. Answers are retrieved from the indexed policy corpus,
                and uncertain cases are routed to human review instead of being guessed.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    quick_prompts = [
        "How many casual leaves do I get per year?",
        "Can I work from home on Friday?",
        "How do I claim travel expense reimbursement?",
    ]
    quick_cols = st.columns(len(quick_prompts))
    for index, label in enumerate(quick_prompts):
        if quick_cols[index].button(label, use_container_width=True):
            _process_query(label)
            st.rerun()

    col1, col2, col3 = st.columns(3)
    col1.metric("Local tokens", st.session_state.local_tokens)
    col2.metric("Cloud tokens", st.session_state.cloud_tokens)
    col3.metric("Messages", len(st.session_state.messages))

    st.markdown("### Conversation")
    chat_container = st.container()
    with chat_container:
        if st.session_state.messages:
            _render_messages()
        else:
            st.caption("Your question and the assistant response will appear here.")

    prompt = st.chat_input("Ask a policy question")
    if not prompt:
        return

    _process_query(prompt)
    st.rerun()


def review_tab():
    st.subheader("HR Review Queue")
    pending = fetch_pending_reviews(limit=50)

    if not pending:
        st.info("No pending reviews right now.")
        return

    options = {
        f"{row['interaction_id'][:8]} - {row['user_query'][:60]}": row
        for row in pending
    }
    selected_label = st.selectbox("Pending escalations", list(options.keys()))
    selected = options[selected_label]

    detail = fetch_interaction(selected["interaction_id"]) or selected

    st.markdown(f"**Query:** {detail['user_query']}")
    st.markdown(f"**Intent:** {detail.get('intent')}")
    st.markdown(f"**Confidence:** {detail.get('confidence')}")
    st.markdown(f"**Reasoning:** {detail.get('confidence_reason') or ''}")
    st.markdown(f"**Draft answer:** {detail.get('draft_answer') or ''}")

    retrieved_chunks = detail.get("retrieved_chunks")
    if isinstance(retrieved_chunks, str):
        try:
            retrieved_chunks = json.loads(retrieved_chunks)
        except Exception:
            retrieved_chunks = []

    if retrieved_chunks:
        st.markdown("**Retrieved chunks**")
        for chunk in retrieved_chunks:
            with st.expander(f"{chunk.get('source')} - {chunk.get('heading')}"):
                st.write(chunk.get("text"))

    reviewer_id = st.text_input("Reviewer name", value="HR Reviewer")
    reviewer_action = st.selectbox("Review action", ["approve", "rewrite", "policy_gap"])
    reviewer_note = st.text_area("Reviewer note")
    revised_answer = st.text_area("Revised answer", value=detail.get("draft_answer") or "")

    if st.button("Submit review", type="primary"):
        final_answer = revised_answer if reviewer_action in {"approve", "rewrite"} else None
        resolve_review(
            interaction_id=detail["interaction_id"],
            reviewer_id=reviewer_id,
            reviewer_action=reviewer_action,
            reviewer_note=reviewer_note,
            revised_answer=final_answer,
        )
        st.success("Review saved.")
        st.rerun()


def audit_tab():
    st.subheader("Audit Trail")
    interactions = fetch_recent_interactions(limit=100)
    if not interactions:
        st.info("No audit records yet.")
        return

    frame = pd.DataFrame(interactions)
    display_columns = [
        "created_at",
        "interaction_id",
        "session_id",
        "status",
        "intent",
        "confidence",
        "escalated",
        "reviewer_action",
        "user_query",
    ]
    available_columns = [column for column in display_columns if column in frame.columns]
    st.dataframe(frame[available_columns], use_container_width=True, hide_index=True)


def ingestion_tab():
    st.subheader("Runtime Policy Ingestion")
    st.caption("Upload markdown, text, Word, or PDF policy files and add them to the vector store without restarting.")

    uploader = st.file_uploader(
        "Upload policy files",
        type=["md", "markdown", "txt", "docx", "pdf"],
        accept_multiple_files=True,
    )

    if uploader and st.button("Ingest uploaded files"):
        ingested = 0
        for uploaded_file in uploader:
            ingested += _ingest_uploaded_file(uploaded_file)
        st.success(f"Ingested {ingested} file(s).")

    if st.button("Rebuild index from documents folder"):
        index_documents(reset=True)
        st.success("Index rebuilt from the document library.")


def sidebar():
    st.sidebar.title("HR Policy Compliance Q&A")
    st.sidebar.write("Three-agent LangGraph workflow with confidence gating and human review.")
    st.sidebar.metric("Local tokens", st.session_state.local_tokens)
    st.sidebar.metric("Cloud tokens", st.session_state.cloud_tokens)
    st.sidebar.write(f"Session ID: `{st.session_state.session_id[:12]}`")
    if st.session_state.last_query:
        st.sidebar.caption(f"Last query: {st.session_state.last_query[:70]}")


def main():
    _inject_styles()
    _init_state()
    sidebar()

    tab_chat, tab_review, tab_audit, tab_ingest = st.tabs(
        ["Employee Chat", "HR Review", "Audit Trail", "Ingestion"]
    )

    with tab_chat:
        employee_chat_tab()
    with tab_review:
        review_tab()
    with tab_audit:
        audit_tab()
    with tab_ingest:
        ingestion_tab()


if __name__ == "__main__":
    main()
