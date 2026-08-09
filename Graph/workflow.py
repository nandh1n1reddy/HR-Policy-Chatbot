from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from Agents.confidence import evaluate_confidence
from Agents.escalation import decline_query, finalize_answer, queue_for_review
from Agents.intent_classifier import classify_intent
from Agents.retriever import retrieve_policy
from Agents.writer import generate_answer
from State.hr_state import HRState


def route_after_classification(state: HRState) -> str:
    return "decline" if not state.get("in_scope", True) else "retrieve"


def route_after_confidence(state: HRState) -> str:
    return "review" if state.get("needs_review") else "finalize"


def build_graph():
    workflow = StateGraph(HRState)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("decline", decline_query)
    workflow.add_node("retrieve", retrieve_policy)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("evaluate_confidence", evaluate_confidence)
    workflow.add_node("queue_review", queue_for_review)
    workflow.add_node("finalize_answer", finalize_answer)

    workflow.set_entry_point("classify_intent")
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_classification,
        {"decline": "decline", "retrieve": "retrieve"},
    )
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", "evaluate_confidence")
    workflow.add_conditional_edges(
        "evaluate_confidence",
        route_after_confidence,
        {"review": "queue_review", "finalize": "finalize_answer"},
    )
    workflow.add_edge("decline", END)
    workflow.add_edge("queue_review", END)
    workflow.add_edge("finalize_answer", END)

    return workflow.compile(checkpointer=MemorySaver())


graph = build_graph()


def run_query(
    query: str,
    session_id: str,
    history: list[dict[str, Any]] | None = None,
) -> HRState:
    state: HRState = {
        "session_id": session_id,
        "query": query,
        "history": history or [],
        "local_tokens": 0,
        "cloud_tokens": 0,
        "retrieved_chunks": [],
        "source_citations": [],
    }
    return graph.invoke(
        state,
        config={"configurable": {"thread_id": session_id}},
    )


if __name__ == "__main__":
    test_state = run_query(
        query="How many casual leaves do I get?",
        session_id="debug-session",
    )
    print(f"Query: {test_state['query']}")
    print(f"Intent: {test_state.get('intent')}")
    print(f"In-scope: {test_state.get('in_scope')}")
    print(f"Confidence: {test_state.get('confidence')}")
    print(f"Status: {test_state.get('status')}")
    print(f"Answer: {test_state.get('answer')}")
