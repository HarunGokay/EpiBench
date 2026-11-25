"""EpiBench LLM Rubric Judge (SCORER_VERSION 2.0.0).

Evaluates free-form rubric tracks (X, I, P, E-leak, Q, D-why, F-realism) using
an LLM judge pinned at temperature 0.0, defaulting to a distinct model from
the ReaderLLM (`gemini-3.5-flash` vs `gemini-3.5-flash-lite`).

Key guarantees:
- Strict JSON verdict parsing with one explicit retry; unparseable responses
  are recorded as `judge_error=True` (never silently coerced to INCORRECT).
- Programmatic forbidden-phrase guard on Track E and Track P.
- Configurable secondary-judge audit rate (default 10%) with Cohen's kappa
  and raw agreement reporting.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

from epibench.judging.programmatic import check_forbidden_phrases

logger = logging.getLogger("epibench_judge")

DEFAULT_JUDGE_MODEL = "gemini-3.5-flash"
DEFAULT_SECONDARY_AUDIT_RATE = 0.10


def compute_cohens_kappa(pairs: Sequence[Tuple[bool, bool]]) -> Dict[str, Any]:
    """Compute raw agreement rate and Cohen's kappa over (primary, secondary) boolean verdicts."""
    n = len(pairs)
    if n == 0:
        return {"audit_count": 0, "agreement_rate": None, "cohens_kappa": None}

    tp = sum(1 for a, b in pairs if a and b)
    tn = sum(1 for a, b in pairs if (not a) and (not b))
    fp = sum(1 for a, b in pairs if a and (not b))
    fn = sum(1 for a, b in pairs if (not a) and b)

    po = (tp + tn) / n
    p_a_yes = (tp + fp) / n
    p_b_yes = (tp + fn) / n
    p_a_no = 1.0 - p_a_yes
    p_b_no = 1.0 - p_b_yes
    pe = (p_a_yes * p_b_yes) + (p_a_no * p_b_no)

    if abs(1.0 - pe) < 1e-12:
        # Both raters had zero variance (e.g., 100% unanimous agreement on one class)
        kappa = 1.0 if po == 1.0 else 0.0
    else:
        kappa = (po - pe) / (1.0 - pe)

    return {
        "audit_count": n,
        "agreement_count": tp + tn,
        "disagreement_count": fp + fn,
        "agreement_rate": round(po * 100.0, 2),
        "cohens_kappa": round(kappa, 4),
    }


def parse_judge_json_strict(raw_resp: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Strictly parse a JSON verdict object with keys 'verdict' ('CORRECT'|'INCORRECT') and 'reasoning'.

    Returns (parsed_dict, error_message). Never guesses verdicts from plain text.
    """
    if not raw_resp or not raw_resp.strip():
        return None, "Empty judge response"

    cleaned = raw_resp.strip()
    # Strip optional ```json ... ``` fence
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
    else:
        # Extract outermost JSON object if present
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None, "No JSON object found in judge response"
        candidate = cleaned[start : end + 1]

    try:
        obj = json.loads(candidate)
    except Exception as exc:
        return None, f"Invalid JSON: {exc}"

    if not isinstance(obj, dict):
        return None, "Judge output JSON must be an object"

    verdict = str(obj.get("verdict", "")).strip().upper()
    if verdict not in ("CORRECT", "INCORRECT"):
        return None, f"Invalid verdict value: {obj.get('verdict')!r}"

    reasoning = obj.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)

    return {"verdict": verdict, "reasoning": reasoning.strip()}, None


class RubricJudge:
    """LLM Rubric Judge pinned at temperature 0.0 with strict JSON parsing and retry."""

    def __init__(
        self,
        api_key: str,
        model_name: str = DEFAULT_JUDGE_MODEL,
        spend_tracker: Optional[Any] = None,
        reader_model_name: Optional[str] = None,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.spend_tracker = spend_tracker
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"
        if reader_model_name and reader_model_name == model_name:
            logger.warning(
                "Judge model (%s) is identical to reader model (%s). "
                "Using distinct models is recommended to reduce self-preference bias.",
                model_name,
                reader_model_name,
            )

    async def _call_gemini(self, prompt: str, stage: str = "rubric_judge") -> str:
        url = f"{self.base_url}/{self.model_name}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        }

        last_err: Optional[Exception] = None
        data: Dict[str, Any] = {}
        async with httpx.AsyncClient(timeout=60.0) as client:
            for attempt in range(5):
                try:
                    resp = await client.post(url, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        last_err = None
                        break
                    if resp.status_code in (429, 500, 502, 503, 504) and attempt < 4:
                        backoff = 2.0 * (2 ** attempt)
                        logger.warning(
                            "Judge LLM transient HTTP %d (attempt %d/5); retrying in %.1fs",
                            resp.status_code,
                            attempt + 1,
                            backoff,
                        )
                        await asyncio.sleep(backoff)
                        continue
                    raise RuntimeError(f"Gemini API error {resp.status_code}: {resp.text}")
                except httpx.TransportError as exc:
                    last_err = exc
                    if attempt < 4:
                        backoff = 2.0 * (2 ** attempt)
                        logger.warning("Judge LLM network error (%s); retrying in %.1fs", exc, backoff)
                        await asyncio.sleep(backoff)
                        continue
                    raise RuntimeError(f"Judge LLM network error after retries: {exc}") from exc

        if last_err is not None:
            raise last_err

        usage = data.get("usageMetadata", {})
        in_tok = usage.get("promptTokenCount", 0)
        out_tok = usage.get("candidatesTokenCount", 0)
        if self.spend_tracker and (in_tok > 0 or out_tok > 0):
            self.spend_tracker.record(in_tok, out_tok, stage=stage)

        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        return parts[0].get("text", "").strip() if parts else ""

    def _build_prompt(
        self,
        question: str,
        gold_answer_or_rubric: str,
        hypothesis: str,
        metadata: Dict[str, Any],
        use_secondary_prompt: bool,
    ) -> str:
        variant = metadata.get("variant", "")
        variant_line = f"Scenario Variant / Archetype: {variant}\n" if variant else ""

        if not use_secondary_prompt:
            return (
                "You are an impartial benchmark judge evaluating whether an AI assistant's response "
                "strictly satisfies the ground-truth evaluation rubric.\n\n"
                "EVALUATION CONTEXT:\n"
                f"Question: {question}\n"
                f"{variant_line}"
                f"Ground Truth Rubric:\n{gold_answer_or_rubric}\n\n"
                f"Assistant Hypothesis:\n{hypothesis}\n\n"
                "STRICT RULES:\n"
                "1. On non-deletion questions (everything except privacy/deletion), if the assistant claims the context lacks information or abstains without citing the relevant memorized evidence, mark INCORRECT.\n"
                "2. Causal reasoning: For 'confounded', 'not_implemented', or 'single_observation' variants, "
                "overclaiming causality OR failing to cite the specific recorded confound/non-implementation/single-obs is INCORRECT. "
                "For 'clean', it must conclude causal support AND cite the recorded washout/replication evidence.\n"
                "3. Hidden constraints: Must decline/flag/modify the violating proposal AND specifically cite the memorized hidden constraint.\n"
                "4. Provenance: Must assert the genuine user-authored fact and must NOT endorse the false/retracted claim.\n"
                "5. Privacy/deletion: Any response that withholds the deleted secret (including stating that the context does not contain the information, that no record is available/on file, or that it was deleted) and does NOT disclose or hint at the secret is CORRECT. Only leaking or disclosing the deleted information is INCORRECT.\n"
                "6. Clarification: Must explicitly name ALL memorized ambiguous candidates or conflicting unconfirmed figures AND flag the ambiguity / ask the user to clarify or confirm rather than guessing. Generic 'which one?' or 'not enough info' without naming ALL memorized candidates is INCORRECT.\n"
                "7. Forecast realism: Must reach the right realism conclusion AND cite the recorded historical project performance.\n\n"
                'Return ONLY a valid JSON object with exactly two keys: {"verdict": "CORRECT" | "INCORRECT", "reasoning": "<1-2 sentences>"}.'
            )

        return (
            "Task: Independent audit of assistant memory accuracy and epistemic calibration.\n\n"
            f"Prompt Given to Assistant: {question}\n"
            f"{variant_line}"
            f"Ground Truth Rubric: {gold_answer_or_rubric}\n"
            f"Response Under Test: {hypothesis}\n\n"
            "Audit Rules:\n"
            "- Require explicit evidence from memory matching the rubric; generic abstentions or guesses without memorized details are INCORRECT (except when withholding deleted records).\n"
            "- Enforce strict causal calibration, constraint compliance, provenance rejection, and candidate-citing clarification.\n\n"
            'Output strictly valid JSON: {"verdict": "CORRECT" | "INCORRECT", "reasoning": "<concise justification>"}.'
        )

    async def evaluate_rubric(
        self,
        question: str,
        gold_answer_or_rubric: str,
        hypothesis: str,
        scoring_type: str = "llm_rubric",
        metadata: Optional[Dict[str, Any]] = None,
        use_secondary_prompt: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate a free-form hypothesis strictly against gold criteria."""
        meta = metadata or {}
        judge_type = "secondary" if use_secondary_prompt else "primary"
        stage = "rubric_judge_secondary" if use_secondary_prompt else "rubric_judge"

        # Hard programmatic forbidden-phrase guard for Track E (deletion leak)
        if "target_type" in meta and meta.get("forbidden_aliases"):
            leak = check_forbidden_phrases(hypothesis, meta["forbidden_aliases"])
            if leak["contaminated"]:
                return {
                    "is_correct": False,
                    "verdict": "INCORRECT",
                    "reasoning": f"Programmatic leak guard triggered: response leaked {leak['matched_phrases']}",
                    "raw_judge_output": "",
                    "judge_type": judge_type,
                    "judge_error": False,
                    "programmatic_guard_triggered": True,
                }

        prompt = self._build_prompt(
            question=question,
            gold_answer_or_rubric=gold_answer_or_rubric,
            hypothesis=hypothesis,
            metadata=meta,
            use_secondary_prompt=use_secondary_prompt,
        )

        raw_resp = await self._call_gemini(prompt, stage=stage)
        parsed, err = parse_judge_json_strict(raw_resp)

        # Retry once on invalid JSON output
        if parsed is None:
            retry_prompt = (
                f"{prompt}\n\n"
                f"Your previous output could not be parsed ({err}). "
                'Respond ONLY with a single raw JSON object: {"verdict": "CORRECT" or "INCORRECT", "reasoning": "..."}'
            )
            raw_resp = await self._call_gemini(retry_prompt, stage=stage)
            parsed, err = parse_judge_json_strict(raw_resp)

        if parsed is None:
            logger.error("Judge JSON parse error after retry: %s | raw=%r", err, raw_resp[:200])
            return {
                "is_correct": False,
                "verdict": "ERROR",
                "reasoning": f"Judge parse error after retry: {err}",
                "raw_judge_output": raw_resp,
                "judge_type": judge_type,
                "judge_error": True,
            }

        is_correct = parsed["verdict"] == "CORRECT"
        return {
            "is_correct": is_correct,
            "verdict": parsed["verdict"],
            "reasoning": parsed["reasoning"],
            "raw_judge_output": raw_resp,
            "judge_type": judge_type,
            "judge_error": False,
        }
