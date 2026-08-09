from __future__ import annotations

from backend.chunk_generator import ChunkGenerator
from backend.ingest import load_documents
from backend.preprocess import TextPreprocessor
from backend.vector_store import VectorStore


def index_documents(source_path=None, reset=True):
    print("=" * 60)
    print("Starting HR Policy Indexing")
    print("=" * 60)

    chunk_generator = ChunkGenerator()
    vector_store = VectorStore()

    if reset:
        vector_store.clear_collection()

    documents = load_documents(source_path)
    print(f"\nLoaded {len(documents)} documents.\n")

    total_chunks = 0
    for document in documents:
        print(f"Processing: {document.filename}")
        document.content = TextPreprocessor.preprocess(document.content)
        chunks = chunk_generator.create_chunks(document)
        print(f"Generated {len(chunks)} chunks")
        vector_store.add_chunks(chunks)
        total_chunks += len(chunks)

    print("\n" + "=" * 60)
    print("Indexing Completed Successfully")
    print("=" * 60)
    print(f"Documents Indexed : {len(documents)}")
    print(f"Chunks Created    : {total_chunks}")
    return {"documents_indexed": len(documents), "chunks_created": total_chunks}


if __name__ == "__main__":
    index_documents()
