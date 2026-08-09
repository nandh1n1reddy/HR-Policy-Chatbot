from __future__ import annotations

import chromadb

from backend.config import CHROMA_DB_PATH, COLLECTION_NAME
from backend.embeddings import get_embedding_generator


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        self.embedding_generator = get_embedding_generator()

    def clear_collection(self):
        existing = self.collection.get()
        if existing.get("ids"):
            self.collection.delete(ids=existing["ids"])

    def add_chunk(self, chunk):
        embedding = self.embedding_generator.generate_chunk_embedding(chunk)
        self.collection.upsert(
            ids=[str(chunk.chunk_id)],
            documents=[chunk.text],
            embeddings=[embedding.tolist()],
            metadatas=[
                {
                    "filename": chunk.filename,
                    "category": chunk.category,
                    "heading": chunk.heading,
                    "source_path": chunk.source_path or "",
                    "page_number": int(chunk.page_number or 0),
                }
            ],
        )

    def add_chunks(self, chunks):
        for chunk in chunks:
            self.add_chunk(chunk)

    def search(self, query: str, top_k: int = 3, where: dict | None = None):
        embedding = self.embedding_generator.generate_embedding(query)
        return self.search_by_embedding(embedding, top_k=top_k, where=where)

    def search_by_embedding(self, embedding, top_k: int = 3, where: dict | None = None):
        """Query using an already-computed embedding.

        Lets callers that run several filtered queries against the same
        query text (e.g. SearchEngine's per-category searches) embed once
        and reuse it, instead of re-running the embedding model for every
        filter variant.
        """
        if self.collection.count() == 0:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        query_kwargs = {
            "query_embeddings": [embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        results = self.collection.query(**query_kwargs)
        return results


if __name__ == "__main__":
    store = VectorStore()
    print("Vector Store Initialized Successfully!")
