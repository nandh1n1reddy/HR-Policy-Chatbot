from __future__ import annotations

from typing import Any

from backend.config import LOCAL_LLM_MAX_TOKENS, LOCAL_LLM_MODEL, USE_LLM_INTENT_CLASSIFIER
from backend.llm_service import extract_json

try:
    from ollama import chat as ollama_chat
except Exception: 
    ollama_chat = None


INTENT_KEYWORDS: dict[str, list[str]] = {
    "leave": ["leave", "casual", "sick", "earned", "maternity", "annual leave"],
    "wfh": ["work from home", "wfh", "remote work", "telework"],
    "overtime_comp_off": ["overtime", "comp off", "comp-off", "compensatory"],
    "whistleblower": ["whistleblower", "report misconduct", "anonymous report"],
    "harassment_grievance": ["harassment", "grievance", "discrimination", "posh"],
    "anti_bribery": ["bribery", "corruption", "gift", "kickback"],
    "ethics_conduct": ["ethics", "code of conduct", "conduct", "behavior"],
    "data_privacy": ["privacy", "data protection", "personal data", "confidential"],
    "travel_expense": ["travel", "expense", "reimbursement", "per diem"],
    "disciplinary": ["disciplinary", "discipline", "warning", "termination"],
}

OUT_OF_SCOPE_KEYWORDS = [
    "laptop",
    "wifi",
    "internet",
    "payroll system",
    "server",
    "printer",
    "software install",
    "it support",
    "finance",
    "accounting",
    "bank",
]


def _heuristic_classify(query: str) -> dict[str, Any]:
    normalized = query.lower()

    if any(keyword in normalized for keyword in OUT_OF_SCOPE_KEYWORDS):
        return {
            "intent": "general_hr",
            "in_scope": False,
            "reason": "Query appears to be outside HR policy support.",
        }

    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return {
                "intent": intent,
                "in_scope": True,
                "reason": f"Matched {intent.replace('_', ' ')} keywords.",
            }

    return {
        "intent": "general_hr",
        "in_scope": True,
        "reason": "No strong topic match found, but the question still appears HR-related.",
    }


def classify_intent(state):
    query = state["query"]

    # The heuristic classifier is near-instant and covers this fixed set of
    # HR categories well, so it's the default. Calling the local LLM here
    # adds a full extra Ollama round trip before retrieval even starts --
    # opt in via USE_LLM_INTENT_CLASSIFIER only if you need finer-grained
    # classification and can tolerate the extra latency.
    candidate = _heuristic_classify(query)

    if USE_LLM_INTENT_CLASSIFIER and ollama_chat is not None:
        system_prompt = """
You are an HR policy classifier.
Classify the employee query into exactly one of these categories:
leave, wfh, overtime_comp_off, whistleblower, harassment_grievance,
anti_bribery, ethics_conduct, data_privacy, travel_expense, disciplinary, general_hr

Also determine whether the query is in_scope for HR policy support.
Return only JSON in this format:
{"intent":"category_name","in_scope":true,"reason":"brief reason"}
""".strip()

        try:
            response = ollama_chat(
                model=LOCAL_LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                options={"num_predict": LOCAL_LLM_MAX_TOKENS},
            )
            llm_candidate = extract_json(response["message"]["content"])
            if llm_candidate:
                candidate = llm_candidate
        except Exception:
            pass

    state["intent"] = candidate.get("intent", "general_hr")
    state["in_scope"] = bool(candidate.get("in_scope", True))
    state["intent_reason"] = candidate.get("reason", "")
    state["token_source"] = "local"
    state["local_tokens"] = state.get("local_tokens", 0) + max(1, len(query) // 4)
    return state


if __name__ == "__main__":
    test_queries = [
        "How many casual leaves do I get per year?",
        "How do I fix my laptop?",
        "Can I work from home?",
    ]

    for query in test_queries:
        state = {"query": query}
        result = classify_intent(state)
        print(f"Q: {query}")
        print(f"  Intent: {result['intent']}, In-scope: {result['in_scope']}\n")
