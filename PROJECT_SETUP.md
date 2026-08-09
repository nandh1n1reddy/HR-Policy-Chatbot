# Project Setup

## What this project now contains

- A LangGraph-based HR policy Q&A workflow
- Local intent classification with Ollama
- ChromaDB-backed policy retrieval
- Confidence gating with human review routing
- Streamlit chat, review queue, audit trail, and ingestion UI

## How to run

1. Activate the virtual environment.
2. Make sure Ollama is running locally with the `mistral` model available.
3. Optionally set `GEMINI_API_KEY` if you want cloud answer generation.
4. Rebuild the index from the document library if needed:

```powershell
python -m backend.indexing
```

5. Start the Streamlit app:

```powershell
streamlit run UI/app.py
```

## Notes

- Markdown, text, Word, and PDF policy files are supported.
- The audit log is stored in `hr_policy_audit.sqlite3`.
- The vector store lives in `chromadb_data/`.
