"""EpiBench Programmatic Judging Suite.

Deterministic rule-based checkers for exact values, counts, dates, and set membership:
- string_match (exact / normalized / alias matching)
- integer_match (digits and number words)
- set_overlap (precision, recall, F1 over decision sets)
- stale_answer detection (returning superseded values)

Strictly decoupled from all memory architectures. Pure Python evaluation.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

WORD_TO_NUMBER = {
    "zero": 0, "one": 1, "two": 2, "twice": 2, "three": 3, "thrice": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "forty": 40, "forty-four": 44, "fifty": 50, "fifty-one": 51, "fifty-two": 52,
}


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, stripping punctuation, and compressing whitespace."""
    t = text.lower()
    t = re.sub(r"[^\w\s\$]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def check_string_match(hypothesis: str, gold: str) -> bool:
    """Check if the gold value or essential tokens are captured in the hypothesis."""
    norm_hyp = normalize_text(hypothesis)
    norm_gold = normalize_text(gold)

    # 1. Direct substring
    if norm_gold in norm_hyp:
        return True

    # 2. Key token presence (e.g., currency amounts, proper nouns)
    gold_tokens = [tok for tok in norm_gold.split() if len(tok) > 2 or tok.startswith("$")]
    if gold_tokens and all(tok in norm_hyp for tok in gold_tokens):
        return True

    # 3. Special case for Yes/No
    if norm_gold in ["yes", "yes it stands", "stands"]:
        if "stands" in norm_hyp or ("yes" in norm_hyp and "no" not in norm_hyp.split()):
            return True

    return False


def check_integer_match(hypothesis: str, gold: int) -> bool:
    """Check if the integer count or equivalent word appears in the hypothesis."""
    norm_hyp = normalize_text(hypothesis)

    # 1. Check exact digits as standalone words
    digit_pattern = rf"\b{gold}\b"
    if re.search(digit_pattern, norm_hyp):
        return True

    # 2. Check number words
    for word, num in WORD_TO_NUMBER.items():
        if num == gold:
            if re.search(rf"\b{word}\b", norm_hyp):
                return True

    return False


def check_stale_answer(hypothesis: str, stale_value: str) -> bool:
    """Check if the hypothesis returned an outdated/stale value."""
    norm_hyp = normalize_text(hypothesis)
    norm_stale = normalize_text(stale_value)
    return norm_stale in norm_hyp


def check_set_overlap(
    hypothesis: str,
    gold_set: List[str],
    unaffected_items: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute precision, recall, and F1 on a set of affected decisions (Track D)."""
    norm_hyp = normalize_text(hypothesis)

    true_positives = 0
    detected_golds = []
    for item in gold_set:
        norm_item = normalize_text(item)
        item_keywords = [w for w in norm_item.split() if len(w) > 3]
        found = False
        if norm_item in norm_hyp:
            found = True
        elif any(sig in norm_item and sig in norm_hyp for sig in ["surround sound", "home theater", "aspen", "ski", "countertop", "plumb", "ad campaign", "press embargo", "advertising"]):
            found = True
        elif item_keywords and sum(1 for kw in item_keywords if kw in norm_hyp) / len(item_keywords) >= 0.4:
            found = True

        if found:
            true_positives += 1
            detected_golds.append(item)

    false_negatives = len(gold_set) - true_positives

    false_positives = 0
    detected_unaffected = []
    if unaffected_items:
        for u in unaffected_items:
            norm_u = normalize_text(u)
            u_kws = [w for w in norm_u.split() if len(w) > 3]
            if norm_u in norm_hyp or (u_kws and sum(1 for kw in u_kws if kw in norm_hyp) / len(u_kws) >= 0.5):
                # Only a false positive if claimed to need revisiting
                revisit_indicators = ["revisit", "cancel", "reschedule", "postpone", "affected", "update"]
                if any(ri in norm_hyp for ri in revisit_indicators):
                    false_positives += 1
                    detected_unaffected.append(u)

    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, len(gold_set))
    f1 = (2 * precision * recall) / max(1e-6, precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "detected_golds": detected_golds,
        "detected_unaffected": detected_unaffected,
        "is_correct": recall >= 0.9 and precision >= 0.9,
    }
