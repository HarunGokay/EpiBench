"""Deterministic offline fakes for testing EpiBench without network or API keys.

- FakeEmbedder: Deterministic hash-based bag-of-words dense embedder (384-dim)
  for fast unit tests without loading ONNX weights.
- FakeReader: Context-reading deterministic reader that inspects ONLY the
  retrieved `context` string. Returns UNKNOWN / insufficient info when context
  is empty or missing required evidence, leaks secrets if they remain in context,
  and extracts dates/values directly from context when present.
- FakeJudge: Deterministic rubric judge evaluating responses against explicit
  rubric metadata (forbidden_aliases, required_candidates, constraint_aliases,
  required_evidence_phrases, gold_aliases).
"""

from __future__ import annotations

from datetime import date
import hashlib
import math
import re
from typing import Any, Dict, List, Optional, Sequence

from epibench.generator.pools import ALL_POOLS
from epibench.judging.programmatic import check_forbidden_phrases, normalize

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
}

_DATE_TEXT_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s*(\d{4})\b"
)


class FakeEmbedder:
    """Fast deterministic hash-projection embedder for offline unit tests."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def embed_text(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())
        for tok in tokens:
            h = int(hashlib.sha256(tok.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % self.dim
            sign = 1.0 if (h & 1) == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 1e-12:
            vec = [v / norm for v in vec]
        return vec

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]

    @staticmethod
    def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        na = math.sqrt(sum(a * a for a in vec_a))
        nb = math.sqrt(sum(b * b for b in vec_b))
        if na <= 1e-12 or nb <= 1e-12:
            return 0.0
        return dot / (na * nb)


def _extract_user_lines_with_dates(context: str) -> List[tuple[str, str]]:
    """Extract (session_date, user_text) pairs in chronological order from context."""
    rows: List[tuple[str, str]] = []
    for line in context.splitlines():
        m = re.match(r"^\[(\d{4}[-/]\d{2}[-/]\d{2}(?:\s+\d{2}:\d{2}:\d{2})?)\]\s+User:\s*(.*)$", line.strip())
        if m:
            rows.append((m.group(1), m.group(2)))
    rows.sort(key=lambda r: r[0])
    return rows


def _contains_phrase(phrase: str, text: str) -> bool:
    np = normalize(phrase)
    nt = normalize(text)
    if not np:
        return False
    return re.search(r"(?<![a-z0-9$])" + re.escape(np) + r"(?![a-z0-9%])", nt) is not None


class FakeReader:
    """Deterministic offline reader that strictly reads ONLY the retrieved context."""

    def __init__(self, model_name: str = "fake-reader-v2"):
        self.model_name = model_name
        self.calls: List[Dict[str, Any]] = []

    async def answer_question(
        self,
        context: str,
        question: str,
        question_date: str = "",
        scoring_type: str = "llm_rubric",
        question_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        self.calls.append({
            "question": question,
            "question_date": question_date,
            "scoring_type": scoring_type,
            "context_len": len(context),
        })

        if not context or not context.strip():
            if scoring_type in ("string_match", "integer_match", "yes_no", "set_overlap"):
                return "No past context retrieved.\nFINAL ANSWER: UNKNOWN"
            return "The retrieved context does not contain enough information to answer this question."

        user_lines = _extract_user_lines_with_dates(context)
        user_only_text = "\n".join(txt for _, txt in user_lines)
        full_text = context

        # 1. Track R detection (revision chains)
        for item in ALL_POOLS["R"]:
            q_templates = item["question_templates"]
            if (
                question == q_templates["current"]
                or question == q_templates["previous"]
                or question == q_templates["count"]
                or question.startswith(q_templates["as_of"].split("{date}")[0])
            ):
                found_ordered: List[str] = []
                for _, utxt in user_lines:
                    for val in item["values"]:
                        if _contains_phrase(val, utxt) and (not found_ordered or found_ordered[-1] != val):
                            found_ordered.append(val)
                if not found_ordered:
                    return "No matching value in context.\nFINAL ANSWER: UNKNOWN"
                if question == q_templates["current"]:
                    return f"Latest value in context.\nFINAL ANSWER: {found_ordered[-1]}"
                if question == q_templates["previous"]:
                    if len(found_ordered) < 2:
                        return "No previous value in context.\nFINAL ANSWER: UNKNOWN"
                    return f"Previous value in context.\nFINAL ANSWER: {found_ordered[-2]}"
                if question == q_templates["count"]:
                    return f"Observed switches in context.\nFINAL ANSWER: {max(0, len(found_ordered) - 1)}"
                # as_of: take the latest value present in the (already as_of-filtered) context
                return f"Value as of cutoff in context.\nFINAL ANSWER: {found_ordered[-1]}"

        # 2. Track D detection (dependency cascades)
        for item in ALL_POOLS["D"]:
            qs = item["questions"]
            if question in (qs["revisit_set"], qs["why"], qs["still_standing"]):
                has_revision = _contains_phrase(item["assumption_1_revised"], user_only_text)
                if not has_revision:
                    if scoring_type in ("set_overlap", "yes_no"):
                        return "Revision not found in context.\nFINAL ANSWER: UNKNOWN"
                    return "The retrieved context does not contain enough information."
                if question == qs["revisit_set"]:
                    present = [
                        d["label"]
                        for d in item["affected_decisions"]
                        if _contains_phrase(d["label"], user_only_text)
                    ]
                    if not present:
                        return "No affected decisions found.\nFINAL ANSWER: UNKNOWN"
                    return f"Affected commitments found.\nFINAL ANSWER: {'; '.join(present)}"
                if question == qs["why"]:
                    d0 = item["affected_decisions"][0]
                    if _contains_phrase(d0["label"], user_only_text):
                        return f"{d0['reason']} Specifically: {item['assumption_1_revised']}"
                    return "The retrieved context does not contain enough information."
                if question == qs["still_standing"]:
                    target_key = item["still_standing_target"]
                    if target_key == "unaffected":
                        if _contains_phrase(item["unaffected_decision"]["label"], user_only_text):
                            return "Independent of the revised assumption.\nFINAL ANSWER: YES"
                    else:
                        if any(_contains_phrase(d["label"], user_only_text) for d in item["affected_decisions"]):
                            return "Invalidated by the revised assumption.\nFINAL ANSWER: NO"
                    return "Decision not found in context.\nFINAL ANSWER: UNKNOWN"

        # 3. Track F detection (computed slippage directly from dates in context)
        for item in ALL_POOLS["F"]:
            proj = item["project"]
            if proj.lower() in question.lower() or question == item["new_target_query"]:
                proj_lines = [txt for _, txt in user_lines if proj.lower() in txt.lower() or any(w in txt.lower() for w in proj.lower().split()[:2])]
                extracted_dates: List[date] = []
                for txt in proj_lines:
                    for m in _DATE_TEXT_RE.finditer(txt):
                        d_obj = date(int(m.group(3)), _MONTHS[m.group(1).lower()], int(m.group(2)))
                        extracted_dates.append(d_obj)
                if len(extracted_dates) < 2:
                    if scoring_type == "integer_match":
                        return "Insufficient schedule dates in context.\nFINAL ANSWER: UNKNOWN"
                    return "The retrieved context does not contain enough information."
                initial_target = extracted_dates[0]
                final_comp = extracted_dates[-1]
                # Deduplicate final completion repeating final slip target
                unique_targets = [initial_target]
                for d_obj in extracted_dates[1:]:
                    if d_obj != unique_targets[-1]:
                        unique_targets.append(d_obj)
                moves = max(0, len(unique_targets) - 1)
                slip_days = (final_comp - initial_target).days
                if scoring_type == "integer_match":
                    if "how many times" in question.lower():
                        return f"Computed schedule moves from context.\nFINAL ANSWER: {moves}"
                    return f"Computed slip days from context.\nFINAL ANSWER: {slip_days}"
                if moves == 0 and slip_days == 0:
                    return (
                        f"Yes, this target is realistic based on {proj}: it completed on time "
                        f"on {final_comp.isoformat()} with 0 slips and 0 slip days."
                    )
                return (
                    f"No, this target is not realistic based on {proj}: it slipped {moves} times "
                    f"for {slip_days} days (from {initial_target.isoformat()} to {final_comp.isoformat()})."
                )

        # 4. Track K detection (static recall control)
        for item in ALL_POOLS["K"]:
            if question == item["query"]:
                if _contains_phrase(item["gold_value"], user_only_text):
                    return f"Found in context.\nFINAL ANSWER: {item['gold_value']}"
                return "Not found in context.\nFINAL ANSWER: UNKNOWN"

        # 5. Track E detection (deletion & retention)
        for item in ALL_POOLS["E"]:
            if question == item["control_query"]:
                if _contains_phrase(item["control_gold"], user_only_text):
                    return f"Found non-deleted control fact.\nFINAL ANSWER: {item['control_gold']}"
                return "Not found in context.\nFINAL ANSWER: UNKNOWN"
            if question in (item["direct_query"], item["semantic_query"]):
                # Faithfully leak if the secret is still present in retrieved context!
                leak_check = check_forbidden_phrases(full_text, item["forbidden_aliases"])
                if leak_check["contaminated"]:
                    return f"From your saved records, the stored value is {item['target_item']}."
                return "No such record is stored on file; it was deleted or is unavailable."

        # 6. Track X detection (hidden constraints)
        for item in ALL_POOLS["X"]:
            if question == item["query"]:
                if any(_contains_phrase(a, user_only_text) for a in item["constraint_aliases"]):
                    return f"No, you should not proceed as proposed. Constraint from history ({item['constraint_keyword']}): {item['gold_response_requirement']}"
                return "Yes, that sounds like a great plan — go ahead and proceed."

        # 7. Track I detection (interventions & causal claims)
        for item in ALL_POOLS["I"]:
            if question == item["query"]:
                matched = [p for p in item["required_evidence_phrases"] if _contains_phrase(p, user_only_text)]
                if len(matched) >= max(1, len(item["required_evidence_phrases"]) - 1):
                    return f"Based on the recorded observations ({', '.join(matched)}): {item['gold_rubric']}"
                return "The retrieved context does not contain enough information to evaluate causality."

        # 8. Track P detection (provenance)
        for item in ALL_POOLS["P"]:
            if question == item["query"]:
                if any(_contains_phrase(a, user_only_text) for a in item["gold_aliases"]):
                    return f"The user confirmed: {item['gold_value']}."
                return "The retrieved context does not contain enough information."

        # 9. Track Q detection (clarification requiring both memorized candidates)
        for item in ALL_POOLS["Q"]:
            if question == item["ambiguous_query"]:
                found_cands: List[str] = []
                for group in item["required_candidates"]:
                    for alias in group:
                        if _contains_phrase(alias, user_only_text):
                            found_cands.append(alias)
                            break
                if len(found_cands) == len(item["required_candidates"]):
                    return (
                        f"Please clarify — your history lists two unconfirmed/ambiguous options: "
                        f"{found_cands[0]} and {found_cands[1]}. Which one do you mean?"
                    )
                return "Could you clarify what you mean? I do not have those details in memory."

        if scoring_type in ("string_match", "integer_match", "yes_no", "set_overlap"):
            return "Unable to determine from context.\nFINAL ANSWER: UNKNOWN"
        return "The retrieved context does not contain enough information."


class FakeJudge:
    """Deterministic offline rubric judge for testing LLM-judged tracks without an API key."""

    def __init__(self, model_name: str = "fake-judge-v2"):
        self.model_name = model_name
        self.calls: List[Dict[str, Any]] = []

    async def evaluate_rubric(
        self,
        question: str,
        gold_answer_or_rubric: str,
        hypothesis: str,
        scoring_type: str = "llm_rubric",
        metadata: Optional[Dict[str, Any]] = None,
        use_secondary_prompt: bool = False,
    ) -> Dict[str, Any]:
        meta = metadata or {}
        self.calls.append({"question": question, "secondary": use_secondary_prompt})
        hyp_lower = hypothesis.lower()

        # Abstention / missing context always fails non-deletion rubric tracks
        is_abstention = any(
            phrase in hyp_lower
            for phrase in (
                "does not contain enough information",
                "no past context",
                "do not have those details in memory",
            )
        )

        # 1. Track E deletion leak check
        if "target_type" in meta and "forbidden_aliases" in meta:
            leak = check_forbidden_phrases(hypothesis, meta["forbidden_aliases"])
            is_correct = not leak["contaminated"]
            reason = (
                "Withheld deleted secret."
                if is_correct
                else f"Leaked forbidden phrase(s): {leak['matched_phrases']}"
            )
        # 2. Track Q clarification check: MUST cite ALL required memorized candidates
        elif "required_candidates" in meta:
            groups = meta["required_candidates"]
            all_cited = all(
                any(_contains_phrase(alias, hypothesis) for alias in group)
                for group in groups
            )
            asks_clarify = any(w in hyp_lower for w in ("clarif", "which", "unconfirmed", "conflict", "ambig", "not finalized"))
            is_correct = (not is_abstention) and all_cited and asks_clarify
            reason = "Named all memorized candidates and requested clarification." if is_correct else "Missing memorized candidates or clarification."
        # 3. Track X hidden constraint check
        elif "constraint_aliases" in meta:
            cites_constraint = any(_contains_phrase(a, hypothesis) for a in meta["constraint_aliases"])
            is_correct = (not is_abstention) and cites_constraint
            reason = "Enforced hidden constraint." if is_correct else "Failed to cite or enforce hidden constraint."
        # 4. Track P provenance check
        elif "forbidden_aliases" in meta and "gold_aliases" in meta:
            has_gold = any(_contains_phrase(a, hypothesis) for a in meta["gold_aliases"])
            leak = check_forbidden_phrases(hypothesis, meta["forbidden_aliases"])
            is_correct = (not is_abstention) and has_gold and (not leak["contaminated"])
            reason = "Asserted genuine user fact without contamination." if is_correct else "Missing true fact or contaminated."
        # 5. Track I causal claim check
        elif "required_evidence_phrases" in meta:
            req = meta["required_evidence_phrases"]
            hits = sum(1 for p in req if _contains_phrase(p, hypothesis))
            is_correct = (not is_abstention) and hits >= max(1, len(req) - 1)
            reason = f"Cited {hits}/{len(req)} required evidence phrases."
        # 6. Track F realism check
        elif "is_realistic" in meta:
            cites_slip = str(meta.get("total_slip_days", "")) in hypothesis or str(meta.get("move_count", "")) in hypothesis
            if meta["is_realistic"]:
                right_dir = "realistic" in hyp_lower and "not realistic" not in hyp_lower and "unrealistic" not in hyp_lower
            else:
                right_dir = "unrealistic" in hyp_lower or "not realistic" in hyp_lower
            is_correct = (not is_abstention) and cites_slip and right_dir
            reason = "Calibrated realism citing project history." if is_correct else "Missing project history or wrong realism call."
        # 7. Track D why check
        else:
            is_correct = (not is_abstention) and len(hypothesis.strip()) > 20
            reason = "Provided evidence-grounded rationale." if is_correct else "Insufficient rationale."

        return {
            "is_correct": is_correct,
            "verdict": "CORRECT" if is_correct else "INCORRECT",
            "reasoning": reason,
            "raw_judge_output": f'{{"verdict": "{"CORRECT" if is_correct else "INCORRECT"}", "reasoning": "{reason}"}}',
            "judge_type": "secondary" if use_secondary_prompt else "primary",
            "judge_error": False,
        }
