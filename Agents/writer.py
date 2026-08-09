from __future__ import annotations

import textwrap

from backend.config import MAX_CONTEXT_CHARS
from backend.llm_service import LLMService


def _format_chunks(retrieved_chunks):
    formatted = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        citation = f"{chunk.get('source')} - {chunk.get('heading')}"
        if chunk.get("page_number"):
            citation += f" (page {chunk['page_number']})"
        formatted.append(
            f"[{index}] {citation}\n{chunk.get('text', '')}"
        )
    return "\n\n".join(formatted)


def _fallback_answer(state):
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return (
            "I could not find policy text that clearly answers this question. "
            "It has been flagged for human review."
        )

    top = chunks[0]
    citation = f"{top.get('source')} - {top.get('heading')}"
    return (
        "Based on the retrieved policy excerpts, the most relevant guidance is:\n"
        f"{textwrap.shorten(top.get('text', '').replace(chr(10), ' '), width=500, placeholder='...')}\n\n"
        f"Source: {citation}"
    )


def generate_answer(state):
    retrieved_chunks = state.get("retrieved_chunks", [])
    context = _format_chunks(retrieved_chunks)[:MAX_CONTEXT_CHARS]

    prompt = f"""
You are an HR policy compliance assistant.
Answer the employee's question strictly using the policy excerpts below.
If the excerpts contain an explicit entitlement, limit, exception, or approval rule, answer directly with that detail.
Do not claim the policy is unclear just because the wording includes conditions or caveats.
Only say the policy is unclear when the excerpts truly do not contain a usable answer.
Prefer concise answers that quote the exact policy meaning in plain English.

Score "confidence" using this rubric, calibrated against a 0.75 auto-answer threshold:
- 0.90-1.00: the excerpts state the answer explicitly and unambiguously, with no interpretation needed.
- 0.75-0.89: the excerpts clearly support the answer, though light interpretation or combining two excerpts was needed.
- 0.40-0.74: the excerpts are only partially relevant, cover a related but different scenario, or leave a meaningful gap.
- 0.00-0.39: the excerpts do not meaningfully address the question.
Be conservative: only score 0.75 or above if an HR reviewer reading the same excerpts would reach your exact conclusion without hesitation.

Employee question:
{state['query']}

Policy excerpts:
{context}

Return only JSON with this schema:
{{
  "answer": "plain English answer",
  "confidence": 0.0,
  "reasoning": "short explanation of coverage",
  "citations": [
    {{"source": "file name", "heading": "section heading", "page_number": 1}}
  ]
}}
""".strip()

    service = LLMService()
    payload, local_tokens, cloud_tokens, mode, raw_text = service.generate_json(prompt, use_cloud=True)

    answer = payload.get("answer") or _fallback_answer(state)
    confidence = payload.get("confidence")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0

    citations = payload.get("citations") or []
    if not citations and retrieved_chunks:
        citations = [
            {
                "source": chunk.get("source"),
                "heading": chunk.get("heading"),
                "page_number": chunk.get("page_number"),
            }
            for chunk in retrieved_chunks[:3]
        ]

    state["draft_answer"] = answer
    state["answer"] = answer
    state["confidence"] = float(confidence)
    state["confidence_reason"] = payload.get("reasoning", "")
    state["source_citations"] = citations
    state["local_tokens"] = state.get("local_tokens", 0) + local_tokens
    state["cloud_tokens"] = state.get("cloud_tokens", 0) + cloud_tokens
    state["raw_model_output"] = raw_text
    state["token_source"] = mode

    if not answer.strip():
        state["answer"] = _fallback_answer(state)
    return state
