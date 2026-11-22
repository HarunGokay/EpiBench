"""EpiBench Reader LLM.

Standardized reader model (Gemini 3.5 Flash-Lite, temperature 0.0)
receiving retrieved context from any memory architecture to generate
final conversational answers. Metered under identical accounting controls.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger("epibench_reader")


class ReaderLLM:
    """Standardized reader model across all evaluated memory systems."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3.5-flash-lite",
        spend_tracker: Optional[Any] = None,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.spend_tracker = spend_tracker
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    async def answer_question(
        self,
        context: str,
        question: str,
        question_date: str = "",
    ) -> str:
        prompt = f"""You are a helpful and accurate personal AI assistant. Answer the user's question directly, accurately, and concisely based strictly on the provided context retrieved from past conversations.

TODAY'S REFERENCE DATE: {question_date}

RETRIEVED CONVERSATION CONTEXT:
{context if context.strip() else "[No past context retrieved]"}

USER'S QUESTION:
{question}

Provide a direct, factual answer. If the context does not contain enough information to answer, state that clearly."""

        url = f"{self.base_url}/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0},
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Reader LLM API error {resp.status_code}: {resp.text}")
            data = resp.json()

        # Record usage
        usage = data.get("usageMetadata", {})
        in_tok = usage.get("promptTokenCount", 0)
        out_tok = usage.get("candidatesTokenCount", 0)
        if self.spend_tracker and (in_tok > 0 or out_tok > 0):
            self.spend_tracker.record(in_tok, out_tok, stage="reader_llm")

        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return parts[0].get("text", "").strip() if parts else ""
