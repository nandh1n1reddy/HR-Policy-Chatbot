from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from ollama import chat as ollama_chat

from backend.config import CLOUD_LLM_MODEL, GEMINI_API_KEY, LOCAL_LLM_MAX_TOKENS, LOCAL_LLM_MODEL

try:
    from google import genai
except Exception:  # pragma: no cover - optional dependency
    genai = None


load_dotenv()


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


class LLMService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY)
        self.local_model = LOCAL_LLM_MODEL
        self.cloud_model = CLOUD_LLM_MODEL
        self._client = None

        if self.api_key and genai is not None:
            self._client = genai.Client(api_key=self.api_key)

    @property
    def cloud_available(self) -> bool:
        return self._client is not None

    def generate_text(self, prompt: str, use_cloud: bool = True) -> tuple[str, int, int, str]:
        if use_cloud and self._client is not None:
            try:
                response = self._client.models.generate_content(
                    model=self.cloud_model,
                    contents=prompt,
                )
                text = response.text or ""
                return text, 0, estimate_tokens(prompt) + estimate_tokens(text), "cloud"
            except Exception:
                pass

        try:
            response = ollama_chat(
                model=self.local_model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                # Without a cap, a local model can keep generating far past
                # what's needed for a short JSON answer. This is the single
                # biggest lever on perceived response time for local runs.
                options={"num_predict": LOCAL_LLM_MAX_TOKENS},
            )
            text = response["message"]["content"]
            return text, estimate_tokens(prompt) + estimate_tokens(text), 0, "local"
        except Exception:
            return "", estimate_tokens(prompt), 0, "fallback"

    def generate_json(self, prompt: str, use_cloud: bool = True) -> tuple[dict[str, Any], int, int, str, str]:
        text, local_tokens, cloud_tokens, mode = self.generate_text(prompt, use_cloud=use_cloud)
        payload = extract_json(text) or {"raw_text": text}
        return payload, local_tokens, cloud_tokens, mode, text
