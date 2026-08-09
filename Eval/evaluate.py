from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Graph.workflow import run_query


def load_test_questions(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def evaluate(csv_path: Path | None = None) -> list[dict[str, str]]:
    csv_path = csv_path or Path(__file__).with_name("test_questions.csv")
    rows = load_test_questions(csv_path)
    if not rows:
        print("No evaluation rows found.")
        return []

    results: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        query = row.get("question", "").strip()
        if not query:
            continue

        outcome = run_query(query=query, session_id=f"eval-{index}")
        results.append(
            {
                "question": query,
                "expected_intent": row.get("expected_intent", ""),
                "expected_route": row.get("expected_route", ""),
                "predicted_intent": outcome.get("intent", ""),
                "predicted_status": outcome.get("status", ""),
                "confidence": str(outcome.get("confidence", "")),
            }
        )

    return results


if __name__ == "__main__":
    results = evaluate()
    print(f"Evaluated {len(results)} questions.")
    for row in results[:10]:
        print(
            f"- {row['question'][:50]} | intent={row['predicted_intent']} | "
            f"status={row['predicted_status']} | confidence={row['confidence']}"
        )
