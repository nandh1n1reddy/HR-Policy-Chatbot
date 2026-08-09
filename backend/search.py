from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from backend.config import INTENT_CATEGORY_HINTS, INTENT_QUERY_HINTS, MAX_QUERY_VARIANTS
from backend.reranker import get_reranker
from backend.vector_store import VectorStore


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _keyword_hits(text: str, keywords: list[str]) -> int:
    normalized = _normalize(text)
    return sum(1 for keyword in keywords if _normalize(keyword) in normalized)


class SearchEngine:
    def __init__(self):
        self.vector_store = VectorStore()

    def _collect_variants(self, query: str, intent: str | None):
        variants = [query]
        if intent and intent != "general_hr":
            variants.append(f"{intent.replace('_', ' ')} {query}")
            for keyword in INTENT_QUERY_HINTS.get(intent, []):
                variants.append(f"{query} {keyword}")
        variants = list(dict.fromkeys(variants))
        # Each variant is a real vector search call (times each candidate
        # category filter), so this list is capped rather than left to grow
        # with however many keywords an intent happens to have.
        return variants[:MAX_QUERY_VARIANTS]

    def _candidate_categories(self, intent: str | None) -> list[str]:
        if not intent or intent == "general_hr":
            return []
        return INTENT_CATEGORY_HINTS.get(intent, [])

    def _score_result(self, result: dict[str, Any], query: str, intent: str | None) -> float:
        distance = result.get("distance")
        distance_score = float(distance) if distance is not None else 1.0

        heading = result.get("heading", "")
        text = result.get("text", "")
        category = result.get("category", "")
        keywords = INTENT_QUERY_HINTS.get(intent or "", [])

        score = distance_score
        if intent and category in self._candidate_categories(intent):
            score -= 0.25

        if keywords:
            hit_count = _keyword_hits(f"{heading} {text}", keywords)
            score -= min(0.18, hit_count * 0.05)

        query_terms = [term for term in re.split(r"[^a-z0-9]+", _normalize(query)) if len(term) >= 3]
        if query_terms:
            normalized_text = _normalize(f"{heading} {text}")
            term_hits = sum(1 for term in query_terms if term in normalized_text)
            score -= min(0.12, term_hits * 0.02)

        return score

    def _build_items(
        self,
        raw_results: dict[str, Any],
        query: str,
        intent: str | None,
        score_penalty: float = 0.0,
    ) -> list[dict[str, Any]]:
        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        ids = raw_results.get("ids", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        items = []
        for index, doc in enumerate(documents):
            item = {
                "id": ids[index],
                "text": doc,
                "filename": metadatas[index].get("filename"),
                "category": metadatas[index].get("category"),
                "heading": metadatas[index].get("heading"),
                "source_path": metadatas[index].get("source_path"),
                "page_number": metadatas[index].get("page_number"),
                "distance": distances[index] if distances else None,
            }
            item["ranking_score"] = self._score_result(item, query, intent) - score_penalty
            items.append(item)
        return items

    def _merge_candidates(
        self,
        candidate_results: dict[str, dict[str, Any]],
        items: list[dict[str, Any]],
    ) -> None:
        for item in items:
            existing = candidate_results.get(item["id"])
            if existing is None or item["ranking_score"] < existing["ranking_score"]:
                candidate_results[item["id"]] = item

    def search(self, query: str, top_k: int = 3, intent: str | None = None):
        variants = self._collect_variants(query, intent)
        categories = self._candidate_categories(intent)
        candidate_results: dict[str, dict[str, Any]] = {}

        search_size = max(top_k * 4, 8)

        for variant in variants:
            # Embed once per variant and reuse it below -- previously this
            # ran the embedding model again for every category filter of the
            # same variant text, which was pure repeated work.
            embedding = self.vector_store.embedding_generator.generate_embedding(variant)

            raw_results = self.vector_store.search_by_embedding(embedding, top_k=search_size)
            self._merge_candidates(
                candidate_results,
                self._build_items(raw_results, query, intent),
            )

            for category in categories:
                filtered_results = self.vector_store.search_by_embedding(
                    embedding,
                    top_k=max(top_k * 2, 6),
                    where={"category": category},
                )
                self._merge_candidates(
                    candidate_results,
                    self._build_items(filtered_results, query, intent, score_penalty=0.05),
                )

        # Vector distance alone is a noisy relevance signal. Rerank the
        # strongest candidates with a cross-encoder so the ranking_score
        # (and, downstream, the confidence gate) reflects true relevance
        # rather than embedding-space distance.
        self._rerank(candidate_results, query, top_k)

        ranked = sorted(
            candidate_results.values(),
            key=lambda item: (
                item.get("ranking_score", 1.0),
                item.get("distance") if item.get("distance") is not None else 1.0,
            ),
        )

        return ranked[:top_k]

    def _rerank(
        self,
        candidate_results: dict[str, dict[str, Any]],
        query: str,
        top_k: int,
    ) -> None:
        reranker = get_reranker()
        if reranker is None or not candidate_results:
            return

        pre_ranked = sorted(candidate_results.values(), key=lambda item: item["ranking_score"])
        rerank_pool = pre_ranked[: max(top_k * 2, 10)]

        relevance_scores = reranker.score(query, [item["text"] for item in rerank_pool])
        for item, relevance in zip(rerank_pool, relevance_scores):
            item["rerank_score"] = relevance
            # Blend: mostly trust the cross-encoder (expressed as a
            # distance-like 1 - relevance so lower still means "better"),
            # but keep some weight on the existing vector/keyword score so a
            # single reranker miss can't flip a strong match to the bottom.
            item["ranking_score"] = (0.3 * item["ranking_score"]) + (0.7 * (1.0 - relevance))


@lru_cache(maxsize=1)
def get_search_engine() -> SearchEngine:
    """Return a single shared SearchEngine instead of reconnecting to
    ChromaDB and rebuilding the vector store wrapper on every query."""
    return SearchEngine()


if __name__ == "__main__":
    engine = SearchEngine()
    query = input("Ask a question : ")
    results = engine.search(query)

    print("\nTop Matches\n")
    for index, result in enumerate(results, start=1):
        print("=" * 60)
        print(f"Match #{index}")
        print(f"ID        : {result['id']}")
        print(f"File      : {result['filename']}")
        print(f"Category  : {result['category']}")
        print(f"Heading   : {result['heading']}")
        print(f"Distance  : {result['distance']}")
        print(f"Score     : {result.get('ranking_score')}")
        print("\nContent\n")
        print(result["text"])
        print()
