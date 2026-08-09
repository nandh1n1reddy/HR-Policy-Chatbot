from __future__ import annotations

from backend.config import CONFIDENCE_THRESHOLD


def _heuristic_confidence(retrieved_chunks):
    if not retrieved_chunks:
        return 0.0

    # rerank_score comes from the cross-encoder in backend/reranker.py and is
    # already a calibrated 0-1 relevance score (higher = better). Use the
    # single strongest match, not an average across several -- a couple of
    # weaker supporting chunks shouldn't drag down confidence when the top
    # match genuinely answers the question.
    rerank_scores = [chunk.get("rerank_score") for chunk in retrieved_chunks if chunk.get("rerank_score") is not None]
    if rerank_scores:
        return max(0.0, min(0.98, max(rerank_scores)))

    ranking_scores = [chunk.get("ranking_score") for chunk in retrieved_chunks if chunk.get("ranking_score") is not None]
    if ranking_scores:
        best_score = min(ranking_scores)
        # Lower ranking score means a stronger match.
        return max(0.0, min(0.98, 1.0 - max(0.0, best_score)))

    distances = [chunk.get("distance") for chunk in retrieved_chunks if chunk.get("distance") is not None]
    if not distances:
        return 0.5

    avg_distance = sum(distances) / len(distances)
    return max(0.0, min(1.0, 1.0 - (avg_distance / 1.5)))


def evaluate_confidence(state):
    model_confidence = state.get("confidence")
    retrieval_confidence = _heuristic_confidence(state.get("retrieved_chunks", []))
    if model_confidence is None or model_confidence == 0:
        model_confidence = retrieval_confidence

    try:
        model_confidence_value = float(model_confidence)
    except (TypeError, ValueError):
        model_confidence_value = retrieval_confidence

    # The heuristic is now backed by a cross-encoder reranker, so it is a
    # more reliable signal than the LLM's uncalibrated self-reported
    # confidence -- weight it accordingly.
    confidence_value = (model_confidence_value * 0.25) + (retrieval_confidence * 0.75)
    confidence_value = max(confidence_value, retrieval_confidence)
    if retrieval_confidence < 0.5:
        confidence_value = min(confidence_value, 0.7)

    # Keep the constituent scores visible so a low final score can be traced
    # back to "retrieval didn't find a strong match" vs. "the model itself
    # was unsure" without needing to inspect raw state.
    reason = state.get("confidence_reason", "")
    debug_note = f"[retrieval_confidence={retrieval_confidence:.2f}, model_confidence={model_confidence_value:.2f}]"
    state["confidence_reason"] = f"{reason} {debug_note}".strip()
    state["retrieval_confidence"] = retrieval_confidence
    state["model_confidence"] = model_confidence_value
    state["confidence"] = confidence_value
    state["needs_review"] = confidence_value < CONFIDENCE_THRESHOLD
    state["escalated"] = state["needs_review"]
    state["status"] = "pending_review" if state["needs_review"] else "answered"
    return state
