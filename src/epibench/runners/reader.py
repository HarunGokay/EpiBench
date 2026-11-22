"""EpiBench Standardized Reader LLM.

Receives retrieved context from any memory architecture and produces answers
following a uniform, structured prompt protocol across all evaluated systems.
Programmatic question types require a final line `FINAL ANSWER: <answer>`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

logger = logging.getLogger("epibench_reader")

READER_PROMPT_VERSION = "2.0.0"


def build_reader_prompt(
    context: str,
    question: str,
    question_date: str = "",
    scoring_type: str = "llm_rubric",
) -> str:
    """Build the standardized reader prompt identical across all memory systems."""
    if scoring_type == "string_match":
        format_rule = (
            "FORMAT REQUIREMENT:\n"
            "- You may briefly reason first if needed.\n"
            "- End your response with a final line strictly in the format:\n"
            "FINAL ANSWER: <exact value only>\n"
            "- If the retrieved context does not contain enough information to answer, end with:\n"
            "FINAL ANSWER: UNKNOWN"
        )
    elif scoring_type == "integer_match":
        format_rule = (
            "FORMAT REQUIREMENT:\n"
            "- You may briefly reason first (for calendar day calculations, count exact calendar days including leap years).\n"
            "- End your response with a final line strictly in the format:\n"
            "FINAL ANSWER: <single integer only>\n"
            "- If the retrieved context does not contain enough information to answer, end with:\n"
            "FINAL ANSWER: UNKNOWN"
        )
    elif scoring_type == "yes_no":
        format_rule = (
            "FORMAT REQUIREMENT:\n"
            "- You may briefly reason first if needed.\n"
            "- End your response with a final line strictly in the format:\n"
            "FINAL ANSWER: YES\n"
            "or\n"
            "FINAL ANSWER: NO\n"
            "- If the retrieved context does not contain enough information to answer, end with:\n"
            "FINAL ANSWER: UNKNOWN"
        )
    elif scoring_type == "set_overlap":
        format_rule = (
            "FORMAT REQUIREMENT:\n"
            "- You may briefly reason first if needed.\n"
            "- On the final line, list ONLY the exact names/labels of the affected decisions or commitments from the conversation, separated by semicolons, strictly in the format:\n"
            "FINAL ANSWER: <item 1>; <item 2>\n"
            "- Do NOT include unaffected decisions on the FINAL ANSWER line.\n"
            "- If no decisions are affected, end with:\n"
            "FINAL ANSWER: NONE\n"
            "- If the retrieved context does not contain enough information to answer, end with:\n"
            "FINAL ANSWER: UNKNOWN"
        )
    else:
        format_rule = (
            "FORMAT REQUIREMENT:\n"
            "- Provide a direct, factual, concise answer citing the specific evidence from the retrieved conversation context.\n"
            "- If the retrieved context does not contain enough information to answer, state clearly that the context does not contain enough information."
        )

    return (
        "You are a helpful and accurate personal AI assistant. Answer the user's question "
        "strictly based on the provided context retrieved from past conversations. "
        "Do not guess or fabricate facts that are not supported by the retrieved context.\n\n"
        f"TODAY'S REFERENCE DATE: {question_date}\n\n"
        "RETRIEVED CONVERSATION CONTEXT:\n"
        f"{context if context.strip() else '[No past context retrieved]'}\n\n"
        "USER'S QUESTION:\n"
        f"{question}\n\n"
        f"{format_rule}"
    )


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
        scoring_type: str = "llm_rubric",
    ) -> str:
        prompt = build_reader_prompt(
            context=context,
            question=question,
            question_date=question_date,
            scoring_type=scoring_type,
        )

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
