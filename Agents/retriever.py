from __future__ import annotations

from backend.config import TOP_K_RESULTS
from backend.search import get_search_engine


def retrieve_policy(state):
    search_engine = get_search_engine()
    results = search_engine.search(
        query=state["query"],
        top_k=state.get("top_k", TOP_K_RESULTS),
        intent=state.get("intent"),
    )

    state["retrieved_chunks"] = [
        {
            "text": result["text"],
            "source": result["filename"],
            "category": result["category"],
            "heading": result["heading"],
            "source_path": result.get("source_path"),
            "page_number": result.get("page_number"),
            "distance": result.get("distance"),
            "ranking_score": result.get("ranking_score"),
            "rerank_score": result.get("rerank_score"),
            "id": result.get("id"),
        }
        for result in results
    ]
    return state
