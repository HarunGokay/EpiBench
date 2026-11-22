"""EpiBench Strict Programmatic Judging Suite (SCORER_VERSION 2.0.0).

Deterministic, structured rule-based evaluation reading ONLY the extracted
`FINAL ANSWER:` field for programmatic question types:
- normalize(): NFKC, lowercase, currency/thousands normalization, never drops short tokens
- score_string_match / check_string_match: exact whole-token match against gold or explicit aliases
- score_integer_match / check_integer_match: strict single-integer parser (digits or number word)
- score_yes_no / check_yes_no: exact YES or NO
- score_set_overlap / check_set_overlap: semicolon-separated exact alias matching (F1 == 1.0)
- check_forbidden_phrases / check_stale_answer: whole-phrase boundary match against explicit aliases

Zero pool-specific hard-coded keywords.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

SCORER_VERSION = "2.0.0"

_FINAL_ANSWER_RE = re.compile(
    r"(?im)(?:^|\n)\s*(?:\*\*|__)?final\s+answer(?:\*\*|__)?\s*:\s*([^\n]*)"
)

_UNITS: Dict[str, int] = {
    "zero": 0, "one": 1, "once": 1, "two": 2, "twice": 2, "three": 3, "thrice": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}

_TENS: Dict[str, int] = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def normalize(text: str) -> str:
    """Normalize text deterministically without ever dropping short tokens.

    Steps:
    1. Unicode NFKC normalization and lowercasing.
    2. Strip thousands commas inside numeric literals (e.g. "$1,500" -> "$1500").
    3. Canonicalize explicit USD currency forms ("1500 dollars", "1500 usd", "usd 1500" -> "$1500").
    4. Preserve digits, "$", "%", decimal points inside numbers, and single-letter trailing
       polarity signs (e.g. "o-"); replace other punctuation with spaces.
    5. Collapse whitespace. Never drops short tokens.
    """
    t = unicodedata.normalize("NFKC", str(text)).lower()

    # Remove thousands separators inside numbers (repeated for 1,000,000+)
    prev = None
    while prev != t:
        prev = t
        t = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", t)

    # Canonicalize explicit dollar/USD phrasing to $<amount>
    t = re.sub(r"\b(\d+(?:\.\d+)?)\s+(?:us\s+)?(?:dollars?|usd)\b", r"$\1", t)
    t = re.sub(r"\busd\s*\$?\s*(\d+(?:\.\d+)?)\b", r"$\1", t)
    t = re.sub(r"\$\s+(\d+(?:\.\d+)?)\b", r"$\1", t)

    # Replace hyphens between word characters with spaces (e.g., "o-negative" -> "o negative")
    t = re.sub(r"(?<=\w)-(?=\w)", " ", t)

    # Protect decimal points inside numbers and trailing polarity on single-letter blood types (e.g. "o-", "a+")
    t = re.sub(r"(?<=\d)\.(?=\d)", "__EPIDOT__", t)
    t = re.sub(
        r"(?<![a-z0-9])([a-z])([+-])(?![a-z0-9])",
        lambda m: f"{m.group(1)}__EPISIGN{'PLUS' if m.group(2) == '+' else 'MINUS'}__",
        t,
    )

    # Replace remaining punctuation (preserving letters, digits, $, %, whitespace) with spaces
    t = re.sub(r"[^\w\s\$%]", " ", t)

    # Restore protected decimals and polarity signs
    t = t.replace("__EPIDOT__", ".")
    t = t.replace("__EPISIGNPLUS__", "+").replace("__EPISIGNMINUS__", "-")

    t = re.sub(r"\s+", " ", t).strip()
    return t


# Backwards-compatible alias
normalize_text = normalize


def extract_final_answer(hypothesis: str) -> Tuple[Optional[str], bool]:
    """Extract the answer strictly after the LAST `FINAL ANSWER:` marker.

    Returns (extracted_text, parse_ok).
    Missing marker or empty value yields (None, False).
    """
    if not hypothesis:
        return None, False

    matches = list(_FINAL_ANSWER_RE.finditer(hypothesis))
    if not matches:
        return None, False

    raw = matches[-1].group(1).strip()
    # Strip outer markdown emphasis/backticks and a single trailing period
    raw = raw.strip("`*")
    if raw.endswith(".") and not re.search(r"\d\.$", raw):
        raw = raw[:-1].strip()
    raw = raw.strip()

    if not raw:
        return None, False
    return raw, True


def _parse_number_words(norm_text: str) -> Optional[int]:
    """Parse an exact English integer word/phrase (0..999) or return None."""
    if not norm_text:
        return None
    tokens = norm_text.split()
    if len(tokens) == 1:
        tok = tokens[0]
        if tok in _UNITS:
            return _UNITS[tok]
        if tok in _TENS:
            return _TENS[tok]
        return None
    if len(tokens) == 2:
        a, b = tokens
        if a in _TENS and b in _UNITS and 1 <= _UNITS[b] <= 9:
            return _TENS[a] + _UNITS[b]
        if a in _UNITS and 1 <= _UNITS[a] <= 9 and b == "hundred":
            return _UNITS[a] * 100
        return None
    if len(tokens) == 3 and tokens[1] == "hundred" and tokens[0] in _UNITS and 1 <= _UNITS[tokens[0]] <= 9:
        rest = _parse_number_words(tokens[2])
        if rest is not None and 0 < rest < 100:
            return _UNITS[tokens[0]] * 100 + rest
    if len(tokens) == 4 and tokens[1] == "hundred" and tokens[0] in _UNITS and 1 <= _UNITS[tokens[0]] <= 9:
        rest = _parse_number_words(f"{tokens[2]} {tokens[3]}")
        if rest is not None and 0 < rest < 100:
            return _UNITS[tokens[0]] * 100 + rest
    return None


def parse_single_integer(raw_answer: str) -> Optional[int]:
    """Parse an extracted answer strictly as a single integer (digits or number word)."""
    s = unicodedata.normalize("NFKC", raw_answer).strip()
    # Allow thousands commas inside a standalone integer literal (e.g. "1,200")
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    norm = normalize(raw_answer)
    if re.fullmatch(r"\d+", norm):
        return int(norm)
    return _parse_number_words(norm)


def score_string_match(
    hypothesis: str,
    gold: str,
    aliases: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Strict whole-token-sequence match on the extracted FINAL ANSWER."""
    extracted, ok = extract_final_answer(hypothesis)
    if not ok or extracted is None:
        return {
            "is_correct": False,
            "parse_failure": True,
            "extracted_answer": None,
            "matched_alias": None,
        }

    norm_ext = normalize(extracted)
    candidates = list(aliases) if aliases else [gold]
    if gold not in candidates:
        candidates.insert(0, gold)

    for cand in candidates:
        if norm_ext == normalize(cand):
            return {
                "is_correct": True,
                "parse_failure": False,
                "extracted_answer": extracted,
                "matched_alias": cand,
            }

    return {
        "is_correct": False,
        "parse_failure": False,
        "extracted_answer": extracted,
        "matched_alias": None,
    }


def check_string_match(
    hypothesis: str,
    gold: str,
    aliases: Optional[Sequence[str]] = None,
) -> bool:
    """Boolean convenience wrapper around score_string_match."""
    return bool(score_string_match(hypothesis, gold, aliases=aliases)["is_correct"])


def score_integer_match(hypothesis: str, gold: int) -> Dict[str, Any]:
    """Strict single-integer match on the extracted FINAL ANSWER."""
    extracted, ok = extract_final_answer(hypothesis)
    if not ok or extracted is None:
        return {
            "is_correct": False,
            "parse_failure": True,
            "extracted_answer": None,
            "parsed_int": None,
        }

    parsed = parse_single_integer(extracted)
    return {
        "is_correct": (parsed is not None and parsed == int(gold)),
        "parse_failure": False,
        "extracted_answer": extracted,
        "parsed_int": parsed,
    }


def check_integer_match(hypothesis: str, gold: int) -> bool:
    """Boolean convenience wrapper around score_integer_match."""
    return bool(score_integer_match(hypothesis, gold)["is_correct"])


def score_yes_no(hypothesis: str, gold: str) -> Dict[str, Any]:
    """Strict YES/NO match on the extracted FINAL ANSWER."""
    norm_gold = normalize(gold)
    if norm_gold not in ("yes", "no"):
        raise ValueError(f"yes_no gold must be 'YES' or 'NO', got {gold!r}")

    extracted, ok = extract_final_answer(hypothesis)
    if not ok or extracted is None:
        return {
            "is_correct": False,
            "parse_failure": True,
            "extracted_answer": None,
        }

    norm_ext = normalize(extracted)
    is_correct = norm_ext == norm_gold
    return {
        "is_correct": is_correct,
        "parse_failure": False,
        "extracted_answer": extracted,
    }


def check_yes_no(hypothesis: str, gold: str) -> bool:
    """Boolean convenience wrapper around score_yes_no."""
    return bool(score_yes_no(hypothesis, gold)["is_correct"])


def _normalize_decision_specs(
    items: Optional[Sequence[Union[str, Dict[str, Any]]]]
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    if not items:
        return specs
    for idx, it in enumerate(items):
        if isinstance(it, str):
            specs.append({"id": f"item_{idx}", "label": it, "aliases": [it]})
        else:
            label = it["label"]
            aliases = list(it.get("aliases") or [label])
            if label not in aliases:
                aliases.insert(0, label)
            specs.append({
                "id": it.get("id", f"item_{idx}"),
                "label": label,
                "aliases": aliases,
            })
    return specs


def score_set_overlap(
    hypothesis: str,
    gold_decisions: Sequence[Union[str, Dict[str, Any]]],
    unaffected_decisions: Optional[Sequence[Union[str, Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Strict semicolon-separated set evaluation using ONLY explicit per-decision aliases.

    Correct iff F1 == 1.0 (every gold decision matched, zero false positives).
    """
    gold_specs = _normalize_decision_specs(gold_decisions)
    unaff_specs = _normalize_decision_specs(unaffected_decisions)

    extracted, ok = extract_final_answer(hypothesis)
    if not ok or extracted is None:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "true_positives": 0,
            "false_negatives": len(gold_specs),
            "false_positives": 0,
            "detected_golds": [],
            "detected_unaffected": [],
            "unrecognized_items": [],
            "extracted_answer": None,
            "parse_failure": True,
            "is_correct": False,
        }

    norm_ext = normalize(extracted)
    if norm_ext in ("none", "unknown"):
        pred_items: List[str] = []
    else:
        pred_items = [part.strip() for part in extracted.split(";") if part.strip()]

    # Build exact normalized alias -> spec lookup maps
    gold_alias_map: Dict[str, Dict[str, Any]] = {}
    for spec in gold_specs:
        for alias in spec["aliases"]:
            gold_alias_map[normalize(alias)] = spec

    unaff_alias_map: Dict[str, Dict[str, Any]] = {}
    for spec in unaff_specs:
        for alias in spec["aliases"]:
            unaff_alias_map[normalize(alias)] = spec

    matched_gold_ids = set()
    detected_golds: List[str] = []
    detected_unaffected: List[str] = []
    unrecognized_items: List[str] = []
    false_positives = 0

    for raw_item in pred_items:
        norm_item = normalize(raw_item)
        if not norm_item:
            continue
        if norm_item in gold_alias_map:
            spec = gold_alias_map[norm_item]
            if spec["id"] not in matched_gold_ids:
                matched_gold_ids.add(spec["id"])
                detected_golds.append(spec["label"])
        elif norm_item in unaff_alias_map:
            false_positives += 1
            detected_unaffected.append(unaff_alias_map[norm_item]["label"])
        else:
            false_positives += 1
            unrecognized_items.append(raw_item)

    true_positives = len(matched_gold_ids)
    false_negatives = len(gold_specs) - true_positives

    if len(gold_specs) == 0 and false_positives == 0 and norm_ext == "none":
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    else:
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / len(gold_specs) if len(gold_specs) > 0 else 0.0
        f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    is_correct = (f1 == 1.0 and true_positives == len(gold_specs) and false_positives == 0)

    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "true_positives": true_positives,
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "detected_golds": detected_golds,
        "detected_unaffected": detected_unaffected,
        "unrecognized_items": unrecognized_items,
        "extracted_answer": extracted,
        "parse_failure": False,
        "is_correct": is_correct,
    }


def check_set_overlap(
    hypothesis: str,
    gold_set: Sequence[Union[str, Dict[str, Any]]],
    unaffected_items: Optional[Sequence[Union[str, Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Alias for score_set_overlap."""
    return score_set_overlap(hypothesis, gold_set, unaffected_decisions=unaffected_items)


def check_forbidden_phrases(
    text: str,
    forbidden_aliases: Sequence[str],
) -> Dict[str, Any]:
    """Whole-phrase boundary match against explicit forbidden_aliases after normalization.

    Used for provenance contamination (Track P) and deletion leak checks (Track E).
    Never uses partial-token or loose substring matching.
    """
    norm_text = normalize(text)
    matched: List[str] = []
    for alias in forbidden_aliases:
        norm_alias = normalize(alias)
        if not norm_alias:
            continue
        pattern = r"(?<![a-z0-9$])" + re.escape(norm_alias) + r"(?![a-z0-9%])"
        if re.search(pattern, norm_text):
            matched.append(alias)
    return {
        "contaminated": len(matched) > 0,
        "matched_phrases": matched,
    }


def check_stale_answer(
    hypothesis: str,
    stale_value: str,
    stale_aliases: Optional[Sequence[str]] = None,
) -> bool:
    """Check if the extracted FINAL ANSWER equals the superseded stale value (or an explicit alias)."""
    res = score_string_match(hypothesis, stale_value, aliases=stale_aliases)
    return bool(res["is_correct"])
