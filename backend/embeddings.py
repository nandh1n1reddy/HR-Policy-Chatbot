from __future__ import annotations

import importlib
from functools import lru_cache

from backend.chunk import Chunk
from backend.config import EMBEDDING_MODEL


class EmbeddingGenerator:
    def __init__(self):
        sentence_transformers = importlib.import_module("sentence_transformers")
        self.model = sentence_transformers.SentenceTransformer(EMBEDDING_MODEL)

    def generate_embedding(self, text: str):
        return self.model.encode(text, normalize_embeddings=True)

    def generate_chunk_embedding(self, chunk: Chunk):
        return self.generate_embedding(chunk.text)


@lru_cache(maxsize=1)
def get_embedding_generator() -> EmbeddingGenerator:
    """Return a single shared EmbeddingGenerator instead of reloading the
    model from disk every time a SearchEngine/VectorStore is created."""
    return EmbeddingGenerator()


if __name__ == "__main__":
    generator = EmbeddingGenerator()
    sample = "Employees receive 12 casual leaves."
    embedding = generator.generate_embedding(sample)
    print(type(embedding))
    print(len(embedding))
    print(embedding[:10])
