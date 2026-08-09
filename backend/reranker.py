from __future__ import annotations

import importlib
import math
from functools import lru_cache

from backend.config import RERANK_MODEL


class Reranker:
    """Scores (query, passage) pairs with a cross-encoder.

    A cross-encoder reads the query and passage together, which is a much
    stronger relevance signal than comparing two independently-embedded
    vectors. This is what lets the confidence gate in Agents/confidence.py
    trust a high score enough to auto-answer at a strict 0.75 threshold.
    """

    def __init__(self, model_name: str = RERANK_MODEL):
        sentence_transformers = importlib.import_module("sentence_transformers")
        self.model = sentence_transformers.CrossEncoder(model_name)

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return a relevance score in [0, 1] per passage (higher = more relevant)."""
        if not passages:
            return []

        raw_scores = self.model.predict([(query, passage) for passage in passages])
        return [1.0 / (1.0 + math.exp(-float(raw_score))) for raw_score in raw_scores]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker | None:
    """Return a single shared Reranker, loaded once.

    Returns None if the cross-encoder model can't be loaded (e.g. no network
    access on first run and the model isn't cached locally yet), so search
    can fall back to vector-distance-only ranking instead of crashing.
    """
    try:
        return Reranker()
    except Exception:
        return None
