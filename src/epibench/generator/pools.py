"""EpiBench Curated World-Model Pools and Pool Validation.

Exports curated, deterministic templates across all 9 tracks:
- Track R: Revision & Contradiction (16 templates)
- Track D: Dependency Cascades (14 templates)
- Track X: Hidden Constraints (12 templates)
- Track F: Forecasts & Patterns (12 templates, computed gold)
- Track I: Interventions & Causal Claims (12 templates)
- Track P: Provenance & False Memory Injection (12 templates)
- Track K: Recall Control (12 templates)
- Track E: Forgetting & Deletion (12 templates)
- Track Q: Uncertainty & Clarification (12 templates)
"""

from __future__ import annotations

from datetime import date
import re
from typing import Any, Dict, List, Optional, Set

from epibench.generator._pools_pkeq import (
    CLARIFICATION_POOLS,
    DELETION_POOLS,
    PROVENANCE_POOLS,
    RECALL_CONTROL_POOLS,
)
from epibench.generator._pools_rd import DEPENDENCY_POOLS, REVISION_POOLS
from epibench.generator._pools_xfi import (
    FORECAST_POOLS,
    HIDDEN_CONSTRAINT_POOLS,
    INTERVENTION_POOLS,
    compute_forecast_gold,
)
from epibench.generator.distractor_pool import DistractorPool

DOMAINS: List[str] = [
    "software_engineering",
    "personal_finance",
    "health_and_fitness",
    "home_renovation",
    "travel_and_logistics",
    "operations",
    "education_and_career",
    "legal_and_compliance",
    "personal_facts",
    "personal_background",
    "security_and_credentials",
    "health_and_privacy",
]

MIN_TEMPLATES_PER_TRACK = 12

ALL_POOLS: Dict[str, List[Dict[str, Any]]] = {
    "R": REVISION_POOLS,
    "D": DEPENDENCY_POOLS,
    "X": HIDDEN_CONSTRAINT_POOLS,
    "F": FORECAST_POOLS,
    "I": INTERVENTION_POOLS,
    "P": PROVENANCE_POOLS,
    "K": RECALL_CONTROL_POOLS,
    "E": DELETION_POOLS,
    "Q": CLARIFICATION_POOLS,
}


def _norm_basic(text: str) -> str:
    t = text.lower().replace(",", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _whole_phrase_present(phrase: str, haystack: str) -> bool:
    p = _norm_basic(phrase)
    h = _norm_basic(haystack)
    if not p:
        return False
    pattern = r"(?<![a-z0-9])" + re.escape(p) + r"(?![a-z0-9])"
    return re.search(pattern, h) is not None


def validate_pools(distractor_pool: Optional[DistractorPool] = None) -> Dict[str, int]:
    """Validate all curated track pools for structural, arithmetic, alias, and filler integrity.

    Raises AssertionError or ValueError on any violation.
    Returns a summary mapping track -> template count.
    """
    dp = distractor_pool or DistractorPool.get_instance()
    filler_texts: List[str] = []
    for f_sess in dp._pool:
        joined = " ".join(
            (t.get("content") or t.get("text") or "") for t in f_sess.get("turns", [])
        )
        filler_texts.append(joined)

    counts: Dict[str, int] = {}
    all_ids: Set[str] = set()
    entity_golds_to_check_against_fillers: List[tuple[str, str]] = []

    for track, pool in ALL_POOLS.items():
        if len(pool) < MIN_TEMPLATES_PER_TRACK:
            raise AssertionError(
                f"Track {track} has only {len(pool)} templates; minimum is {MIN_TEMPLATES_PER_TRACK}"
            )
        counts[track] = len(pool)
        for item in pool:
            tid = item.get("id")
            if not tid or tid in all_ids:
                raise AssertionError(f"Track {track}: missing or duplicate template id {tid!r}")
            all_ids.add(tid)

    # 1. Track R validation
    for item in REVISION_POOLS:
        tid = item["id"]
        values = item["values"]
        if len(values) < 3:
            raise AssertionError(f"{tid}: Track R requires at least 3 values")
        norm_vals = [_norm_basic(v) for v in values]
        if len(set(norm_vals)) != len(norm_vals):
            raise AssertionError(f"{tid}: Track R values are not pairwise distinct: {values}")
        # Current, previous, as_of (values[-1], values[-2], values[0]) must all be distinct
        if len({_norm_basic(values[-1]), _norm_basic(values[-2]), _norm_basic(values[0])}) != 3:
            raise AssertionError(f"{tid}: current/previous/as_of values must be pairwise distinct")
        phrasings = item["user_phrasings"]
        if len(phrasings) != len(values):
            raise AssertionError(f"{tid}: user_phrasings length must match values length")
        val_aliases = item.get("value_aliases", {})
        seen_alias_norms: Dict[str, str] = {}
        for val, phrasing in zip(values, phrasings):
            formatted = phrasing.format(val=val)
            if val not in formatted:
                raise AssertionError(f"{tid}: value {val!r} does not appear in phrasing {formatted!r}")
            aliases = val_aliases.get(val, [])
            if not aliases or _norm_basic(val) not in [_norm_basic(a) for a in aliases]:
                raise AssertionError(f"{tid}: missing value_aliases for {val!r}")
            for a in aliases:
                na = _norm_basic(a)
                if na in seen_alias_norms and seen_alias_norms[na] != val:
                    raise AssertionError(
                        f"{tid}: alias {a!r} collides between {seen_alias_norms[na]!r} and {val!r}"
                    )
                seen_alias_norms[na] = val
            entity_golds_to_check_against_fillers.append((tid, val))

    # 2. Track D validation
    d_yes = 0
    d_no = 0
    for item in DEPENDENCY_POOLS:
        tid = item["id"]
        affected = item["affected_decisions"]
        unaffected = item["unaffected_decision"]
        if len(affected) < 2:
            raise AssertionError(f"{tid}: Track D requires at least 2 affected decisions")
        aff_alias_set: Set[str] = set()
        for dec in affected:
            aliases = dec.get("aliases", [])
            if not aliases:
                raise AssertionError(f"{tid}: decision {dec['id']} missing aliases")
            if not any(_whole_phrase_present(a, dec["text"]) for a in [dec["label"], *aliases]):
                raise AssertionError(f"{tid}: decision {dec['label']!r} not present in evidence text")
            for a in aliases:
                aff_alias_set.add(_norm_basic(a))
            entity_golds_to_check_against_fillers.append((tid, dec["label"]))
        unaff_aliases = unaffected.get("aliases", [])
        if not unaff_aliases:
            raise AssertionError(f"{tid}: unaffected decision missing aliases")
        if not any(_whole_phrase_present(a, unaffected["text"]) for a in [unaffected["label"], *unaff_aliases]):
            raise AssertionError(f"{tid}: unaffected decision {unaffected['label']!r} not in text")
        unaff_alias_set = {_norm_basic(a) for a in unaff_aliases}
        overlap = aff_alias_set & unaff_alias_set
        if overlap:
            raise AssertionError(f"{tid}: alias overlap between affected and unaffected: {overlap}")
        gold_yn = item.get("still_standing_gold")
        if gold_yn == "YES":
            d_yes += 1
        elif gold_yn == "NO":
            d_no += 1
        else:
            raise AssertionError(f"{tid}: invalid still_standing_gold {gold_yn!r}")
    if d_yes == 0 or d_no == 0:
        raise AssertionError(f"Track D must contain both YES and NO still_standing items (got YES={d_yes}, NO={d_no})")

    # 3. Track X validation
    for item in HIDDEN_CONSTRAINT_POOLS:
        tid = item["id"]
        kw = item.get("constraint_keyword", "")
        if not kw or not _whole_phrase_present(kw, item["constraint_statement"]):
            raise AssertionError(f"{tid}: constraint_keyword {kw!r} missing from constraint_statement")
        if not item.get("constraint_aliases"):
            raise AssertionError(f"{tid}: missing constraint_aliases")

    # 4. Track F validation (computed gold + Titan leap year check)
    f_realistic = 0
    f_unrealistic = 0
    for item in FORECAST_POOLS:
        tid = item["id"]
        if "move_count" in item or "total_slip_days" in item or "gold_answers" in item:
            raise AssertionError(f"{tid}: Track F pool must compute gold from events, not hand-type it")
        events = item["events"]
        prev_d: Optional[date] = None
        for ev in events:
            ev_d = date.fromisoformat(ev["date"])
            _ = date.fromisoformat(ev["target_date"])
            if prev_d is not None and ev_d < prev_d:
                raise AssertionError(f"{tid}: event dates not chronological: {ev_d} < {prev_d}")
            prev_d = ev_d
        computed = compute_forecast_gold(events)
        if tid == "f_00_titan_alpha":
            if computed["total_slip_days"] != 53 or computed["move_count"] != 2:
                raise AssertionError(f"Titan computed gold wrong: {computed}")
        if item["is_realistic"]:
            f_realistic += 1
        else:
            f_unrealistic += 1
    if f_realistic == 0 or f_unrealistic == 0:
        raise AssertionError("Track F must contain both realistic and unrealistic templates")

    # 5. Track I validation
    variant_counts: Dict[str, int] = {}
    valid_variants = {"clean", "confounded", "not_implemented", "single_observation"}
    for item in INTERVENTION_POOLS:
        tid = item["id"]
        v = item["variant"]
        if v not in valid_variants:
            raise AssertionError(f"{tid}: invalid variant {v!r}")
        variant_counts[v] = variant_counts.get(v, 0) + 1
        if not item.get("required_evidence_phrases"):
            raise AssertionError(f"{tid}: missing required_evidence_phrases")
    if set(variant_counts.keys()) != valid_variants:
        raise AssertionError(f"Track I missing variants: {valid_variants - set(variant_counts.keys())}")

    # 6. Track P validation
    for item in PROVENANCE_POOLS:
        tid = item["id"]
        user_text = " ".join(
            t["content"] for t in item["timeline"] if t.get("role") == "user"
        )
        gold_aliases = item.get("gold_aliases", [])
        forb_aliases = item.get("forbidden_aliases", [])
        if not gold_aliases or not forb_aliases:
            raise AssertionError(f"{tid}: Track P requires non-empty gold_aliases and forbidden_aliases")
        if not any(_whole_phrase_present(a, user_text) for a in [item["gold_value"], *gold_aliases]):
            raise AssertionError(f"{tid}: gold_value {item['gold_value']!r} not found in user turns")
        if {_norm_basic(a) for a in gold_aliases} & {_norm_basic(a) for a in forb_aliases}:
            raise AssertionError(f"{tid}: overlap between gold_aliases and forbidden_aliases")
        entity_golds_to_check_against_fillers.append((tid, item["gold_value"]))

    # 7. Track K validation
    for item in RECALL_CONTROL_POOLS:
        tid = item["id"]
        aliases = item.get("gold_aliases", [])
        if not aliases:
            raise AssertionError(f"{tid}: Track K requires gold_aliases")
        if not any(_whole_phrase_present(a, item["statement"]) for a in [item["gold_value"], *aliases]):
            raise AssertionError(f"{tid}: gold_value {item['gold_value']!r} not found in statement")
        entity_golds_to_check_against_fillers.append((tid, item["gold_value"]))

    # 8. Track E validation
    for item in DELETION_POOLS:
        tid = item["id"]
        target = item["target_item"]
        ctrl = item["control_gold"]
        if not _whole_phrase_present(target, item["session_1"]):
            raise AssertionError(f"{tid}: target_item {target!r} missing from session_1")
        if _whole_phrase_present(target, item["session_2_non_deleted"]):
            raise AssertionError(f"{tid}: target_item {target!r} leaked into session_2_non_deleted")
        if not any(_whole_phrase_present(a, item["session_2_non_deleted"]) for a in [ctrl, *item.get("control_aliases", [])]):
            raise AssertionError(f"{tid}: control_gold {ctrl!r} missing from session_2_non_deleted")
        if not item.get("forbidden_aliases") or not item.get("control_aliases"):
            raise AssertionError(f"{tid}: Track E requires forbidden_aliases and control_aliases")
        entity_golds_to_check_against_fillers.append((tid, target))
        entity_golds_to_check_against_fillers.append((tid, ctrl))

    # 9. Track Q validation
    for item in CLARIFICATION_POOLS:
        tid = item["id"]
        cands = item.get("required_candidates", [])
        if len(cands) < 2:
            raise AssertionError(f"{tid}: Track Q requires at least 2 required_candidates groups")
        in_query_count = 0
        for group in cands:
            if not any(_whole_phrase_present(alias, item["context_statement"]) for alias in group):
                raise AssertionError(f"{tid}: candidate group {group} missing from context_statement")
            if any(_whole_phrase_present(alias, item["ambiguous_query"]) for alias in group):
                in_query_count += 1
        if in_query_count >= len(cands):
            raise AssertionError(f"{tid}: all candidate options already revealed in ambiguous_query")

    # 10. Filler contamination check: no entity gold may appear in any filler session
    for tid, gold_str in entity_golds_to_check_against_fillers:
        for f_idx, f_text in enumerate(filler_texts):
            if _whole_phrase_present(gold_str, f_text):
                raise AssertionError(
                    f"Filler leak detected: gold {gold_str!r} (from {tid}) appears in filler {f_idx}"
                )

    return counts


__all__ = [
    "ALL_POOLS",
    "CLARIFICATION_POOLS",
    "DELETION_POOLS",
    "DEPENDENCY_POOLS",
    "DOMAINS",
    "FORECAST_POOLS",
    "HIDDEN_CONSTRAINT_POOLS",
    "INTERVENTION_POOLS",
    "MIN_TEMPLATES_PER_TRACK",
    "PROVENANCE_POOLS",
    "RECALL_CONTROL_POOLS",
    "REVISION_POOLS",
    "compute_forecast_gold",
    "validate_pools",
]
