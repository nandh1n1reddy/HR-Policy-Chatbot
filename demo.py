from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Graph.workflow import run_query
from backend.audit import fetch_recent_interactions
from backend.indexing import index_documents


def main():
    print("=" * 70)
    print("HR POLICY COMPLIANCE Q&A - SMOKE TEST")
    print("=" * 70)
    print("Rebuilding the index from the document library...")
    summary = index_documents(reset=True)
    print(summary)

    print("\nRunning a sample query...\n")
    result = run_query(
        query="Can I work from home on Friday?",
        session_id="demo-session",
    )

    print("Status:", result.get("status"))
    print("Intent:", result.get("intent"))
    print("Confidence:", result.get("confidence"))
    print("Answer:\n", result.get("answer"))

    print("\nRecent audit rows:")
    for row in fetch_recent_interactions(limit=5):
        print(f"- {row['status']} | {row['intent']} | {row['user_query'][:60]}")


if __name__ == "__main__":
    main()
