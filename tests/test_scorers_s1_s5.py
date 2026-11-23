"""Regression and property tests for Phase 3 strict structured scorers (S1-S5)."""

from __future__ import annotations

from pathlib import Path
import pytest

from epibench.generator.pools import ALL_POOLS
from epibench.generator.scenario_generator import ScenarioGenerator
from epibench.judging.programmatic import (
    SCORER_VERSION,
    check_forbidden_phrases,
    check_integer_match,
    check_stale_answer,
    check_string_match,
    check_yes_no,
    normalize,
    score_integer_match,
    score_set_overlap,
    score_string_match,
    score_yes_no,
)
from epibench.runners.reader import READER_PROMPT_VERSION, build_reader_prompt

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "hypothesis,gold,aliases,expected",
    [
        ("FINAL ANSWER: PostgreSQL 14", "PostgreSQL 16", ["PostgreSQL 16", "Postgres 16"], False),
        ("FINAL ANSWER: PostgreSQL 16", "PostgreSQL 16", ["PostgreSQL 16", "Postgres 16"], True),
        ("We previously ran PostgreSQL 14 and switched.\nFINAL ANSWER: PostgreSQL 16", "PostgreSQL 16", None, True),
        ("We previously ran PostgreSQL 16 and rolled back.\nFINAL ANSWER: PostgreSQL 14", "PostgreSQL 16", None, False),
        ("PostgreSQL 16 is our current database.", "PostgreSQL 16", None, False),
        ("FINAL ANSWER: $1,500", "$1,500", ["$1,500", "$1500", "1500 dollars"], True),
        ("FINAL ANSWER: $1500", "$1,500", None, True),
        ("FINAL ANSWER: 1500 dollars", "$1,500", None, True),
        ("FINAL ANSWER: 1,500 dollars", "$1,500", None, True),
        ("FINAL ANSWER: USD 1500", "$1,500", None, True),
        ("FINAL ANSWER: $1,200", "$1,500", None, False),
    ],
)
def test_S1_short_token_and_currency_normalization(hypothesis, gold, aliases, expected):
    """S1: Never drop short tokens (e.g. '14' vs '16'); normalize currency consistently."""
    assert SCORER_VERSION == "2.0.0"
    assert check_string_match(hypothesis, gold, aliases=aliases) is expected


@pytest.mark.parametrize(
    "hypothesis,gold,expected,expect_parse_fail",
    [
        ("FINAL ANSWER: NO", "YES", False, False),
        ("FINAL ANSWER: YES", "YES", True, False),
        ("The decision no longer stands.\nFINAL ANSWER: NO", "YES", False, False),
        ("The decision no longer stands.\nFINAL ANSWER: NO", "NO", True, False),
        ("Yes, it still stands.", "YES", False, True),
        ("FINAL ANSWER: MAYBE", "YES", False, False),
        ("FINAL ANSWER: UNKNOWN", "YES", False, False),
    ],
)
def test_S2_yes_no_strict_no_negation_leak(hypothesis, gold, expected, expect_parse_fail):
    """S2: 'No longer stands' / NO fails for gold YES; missing marker is parse failure."""
    res = score_yes_no(hypothesis, gold)
    assert res["is_correct"] is expected
    assert res["parse_failure"] is expect_parse_fail
    assert check_yes_no(hypothesis, gold) is expected


@pytest.mark.parametrize(
    "hypothesis,gold,expected,expect_parse_fail",
    [
        ("It changed 3 times: 1. a 2. b 3. c\nFINAL ANSWER: 3", 2, False, False),
        ("We switched twice.\nFINAL ANSWER: 2", 2, True, False),
        ("We switched twice.\nFINAL ANSWER: two", 2, True, False),
        ("It was not 2, it was 7.\nFINAL ANSWER: 7", 2, False, False),
        ("FINAL ANSWER: not 2, it was 7", 2, False, False),
        ("FINAL ANSWER: 53", 53, True, False),
        ("FINAL ANSWER: 52", 53, False, False),
        ("FINAL ANSWER: fifty-three", 53, True, False),
        ("The team slipped 53 days.", 53, False, True),
        ("FINAL ANSWER: UNKNOWN", 2, False, False),
    ],
)
def test_S3_integer_match_strict_single_integer(hypothesis, gold, expected, expect_parse_fail):
    """S3: Integer match reads only FINAL ANSWER as a single integer, immune to list bullets and prose."""
    res = score_integer_match(hypothesis, gold)
    assert res["is_correct"] is expected
    assert res["parse_failure"] is expect_parse_fail
    assert check_integer_match(hypothesis, gold) is expected


def test_S4_set_overlap_exact_alias_and_no_phantom_penalty():
    """S4: Exact alias set matching, zero hardcoded keywords, no phantom 'revisit' penalty."""
    gold_decisions = [
        {"id": "d1", "label": "downhill ski package rental", "aliases": ["downhill ski package rental", "ski trip", "ski rental"]},
        {"id": "d2", "label": "winter snow tire chains order", "aliases": ["winter snow tire chains order", "snow chains"]},
    ]
    unaffected = [
        {"id": "u1", "label": "routine passport renewal application", "aliases": ["routine passport renewal application", "passport renewal"]},
    ]

    # Exact set -> True
    exact = (
        "The routine passport renewal application does NOT need to be revisited.\n"
        "FINAL ANSWER: downhill ski package rental; winter snow tire chains order"
    )
    r_exact = score_set_overlap(exact, gold_decisions, unaffected)
    assert r_exact["is_correct"] is True
    assert r_exact["f1"] == 1.0
    assert r_exact["false_positives"] == 0
    assert r_exact["parse_failure"] is False

    # Subset -> False
    subset = "FINAL ANSWER: downhill ski package rental"
    r_sub = score_set_overlap(subset, gold_decisions, unaffected)
    assert r_sub["is_correct"] is False
    assert r_sub["recall"] == 0.5

    # Superset including unaffected -> False
    superset = "FINAL ANSWER: downhill ski package rental; winter snow tire chains order; routine passport renewal application"
    r_sup = score_set_overlap(superset, gold_decisions, unaffected)
    assert r_sup["is_correct"] is False
    assert r_sup["false_positives"] == 1

    # Substring false-positive guard: "asking about skills" must NEVER match "ski trip"
    skills = "FINAL ANSWER: asking about skills; snow chains"
    r_skills = score_set_overlap(skills, gold_decisions, unaffected)
    assert r_skills["is_correct"] is False
    assert r_skills["true_positives"] == 1
    assert r_skills["false_positives"] == 1

    # Missing marker -> parse failure
    r_missing = score_set_overlap("downhill ski package rental; winter snow tire chains order", gold_decisions, unaffected)
    assert r_missing["is_correct"] is False
    assert r_missing["parse_failure"] is True


def test_S5_forbidden_phrase_boundary_matching():
    """S5: Contamination/forbidden check uses whole-phrase boundary match, not lenient string match."""
    res_leak = check_forbidden_phrases(
        "Your API key is sk-live-alpha-8821.",
        ["sk-live-alpha-8821"],
    )
    assert res_leak["contaminated"] is True

    res_clean = check_forbidden_phrases(
        "No API key is stored on file; it was permanently deleted.",
        ["sk-live-alpha-8821"],
    )
    assert res_clean["contaminated"] is False

    # Substring guard: "ski" must not match "skills" or "asking"
    res_sub = check_forbidden_phrases("Asking about skills.", ["ski"])
    assert res_sub["contaminated"] is False

    # Stale answer checks ONLY FINAL ANSWER, never prose mentioning prior value
    assert check_stale_answer(
        "Previously CockroachDB.\nFINAL ANSWER: PostgreSQL 16",
        "CockroachDB",
    ) is False
    assert check_stale_answer(
        "FINAL ANSWER: CockroachDB",
        "CockroachDB",
    ) is True


def test_no_hardcoded_pool_keywords_in_judging():
    """Phase 3 Gate: grep confirms zero pool-specific keywords remain in judging/."""
    judging_dir = REPO_ROOT / "src" / "epibench" / "judging"
    banned = ["surround sound", "home theater", "aspen", "countertop", "plumb", "ad campaign", "press embargo"]
    for py_file in judging_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8").lower()
        for kw in banned:
            assert kw not in text, f"Hard-coded pool keyword {kw!r} found in {py_file.name}"


def test_property_all_generated_scenarios_gold_passes_wrong_fails():
    """Phase 3 Gate property test: every scenario's gold passes its scorer and wrong/stale values fail."""
    gen = ScenarioGenerator()
    for track, pool in ALL_POOLS.items():
        for idx in range(len(pool)):
            sc = gen.generate_scenario(track, scenario_idx=idx, split="dev", seed=20261122, filler_count=2)
            for q in sc.questions:
                if q.scoring_type == "string_match":
                    aliases = q.metadata.get("gold_aliases")
                    res_ok = score_string_match(f"FINAL ANSWER: {q.gold_answer}", str(q.gold_answer), aliases=aliases)
                    assert res_ok["is_correct"] is True, f"{q.question_id}: gold failed"
                    assert res_ok["parse_failure"] is False

                    for wrong in q.metadata.get("wrong_values", []):
                        res_bad = score_string_match(f"FINAL ANSWER: {wrong}", str(q.gold_answer), aliases=aliases)
                        assert res_bad["is_correct"] is False, f"{q.question_id}: wrong {wrong!r} falsely passed"

                    res_unk = score_string_match("FINAL ANSWER: UNKNOWN", str(q.gold_answer), aliases=aliases)
                    assert res_unk["is_correct"] is False

                elif q.scoring_type == "integer_match":
                    gold_int = int(q.gold_answer)
                    assert check_integer_match(f"FINAL ANSWER: {gold_int}", gold_int) is True
                    for wrong in q.metadata.get("wrong_values", [gold_int + 1]):
                        assert check_integer_match(f"FINAL ANSWER: {wrong}", gold_int) is False

                elif q.scoring_type == "yes_no":
                    gold_yn = str(q.gold_answer)
                    other = "NO" if gold_yn == "YES" else "YES"
                    assert check_yes_no(f"FINAL ANSWER: {gold_yn}", gold_yn) is True
                    assert check_yes_no(f"FINAL ANSWER: {other}", gold_yn) is False
                    assert check_yes_no("FINAL ANSWER: UNKNOWN", gold_yn) is False

                elif q.scoring_type == "set_overlap":
                    gold_decs = q.metadata["gold_decisions"]
                    unaff_decs = q.metadata["unaffected_decisions"]
                    all_labels = "; ".join(d["label"] for d in gold_decs)
                    assert score_set_overlap(f"FINAL ANSWER: {all_labels}", gold_decs, unaff_decs)["is_correct"] is True
                    # Subset must fail
                    assert score_set_overlap(f"FINAL ANSWER: {gold_decs[0]['label']}", gold_decs, unaff_decs)["is_correct"] is False
                    # Superset with unaffected must fail
                    sup = all_labels + "; " + unaff_decs[0]["label"]
                    assert score_set_overlap(f"FINAL ANSWER: {sup}", gold_decs, unaff_decs)["is_correct"] is False


def test_reader_prompt_versions_and_structured_instructions():
    """Verify standardized reader prompts include FINAL ANSWER instructions for structured scoring types."""
    assert READER_PROMPT_VERSION == "2.0.0"
    for st in ("string_match", "integer_match", "yes_no", "set_overlap"):
        p = build_reader_prompt("ctx", "q", question_date="2024-05-01", scoring_type=st)
        assert "FINAL ANSWER:" in p
        assert "2024-05-01" in p
    p_rubric = build_reader_prompt("ctx", "q", question_date="2024-05-01", scoring_type="llm_rubric")
    assert "2024-05-01" in p_rubric
    assert "FINAL ANSWER:" not in p_rubric
