"""Regression and invariant tests for EpiBench v2.2.0 epistemic-stress profile & offline eval."""

from __future__ import annotations

import json
from pathlib import Path
import re
import pytest

from epibench.adapters.base import count_tokens, format_session_text
from epibench.cli import main as cli_main
from epibench.generator._pools_stress import (
    STRESS_FORECAST_EVENT_COUNT,
    STRESS_MULTIHOP_DEPENDENCY_POOLS,
    STRESS_PORTFOLIO_INTERVENTION_POOLS,
    STRESS_PORTFOLIO_SESSION_COUNT,
    STRESS_REVISION_COUNT,
    STRESS_ZERO_OVERLAP_CONSTRAINT_POOLS,
)
from epibench.generator.manifest import generate_manifest
from epibench.generator.pools import validate_pools
from epibench.generator.scenario_generator import (
    BENCHMARK_PROFILES,
    GENERATOR_VERSION,
    VALID_TRACKS,
    ScenarioGenerator,
    parse_session_date,
    resolve_profile,
)
from epibench.runners.offline_eval import format_offline_report, run_offline_epistemic_eval
from epibench.runners.validate import run_validity_gate

REPO_ROOT = Path(__file__).resolve().parent.parent

_STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are",
    "was", "were", "be", "my", "our", "we", "i", "me", "you", "it", "that",
    "this", "with", "at", "by", "from", "as", "not", "no", "any", "all",
    "can", "should", "want", "wants", "have", "has", "had", "do", "does",
    "so", "if", "but", "into", "up", "out", "about", "per", "just", "now",
    "us", "her", "his", "their", "them", "would", "could", "will", "go",
    "ahead", "using", "because", "than", "before", "after", "which", "what",
    "who", "how", "when", "where", "why", "its", "him", "she", "he", "they",
}


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 2}


def test_v2_2_profile_resolution_and_alias():
    """Stress profile and epistemic_stress alias resolve identically with 240 mixed fillers."""
    assert GENERATOR_VERSION == "2.2.0"
    assert "stress" in BENCHMARK_PROFILES
    p1 = resolve_profile(profile="stress")
    p2 = resolve_profile(profile="epistemic_stress")
    assert p1 == p2
    assert p1["profile"] == "stress"
    assert p1["filler_count"] == 240
    assert p1["distractor_mode"] == "mixed"
    assert p1["token_budget"] == 1500
    assert p1["epistemic_stress"] is True

    # Ensure dev & hard remain non-stress
    assert resolve_profile(profile="dev")["epistemic_stress"] is False
    assert resolve_profile(profile="hard")["epistemic_stress"] is False


def test_v2_2_upgrade1_evidence_exceeds_budget_tracks_r_and_f():
    """Upgrade #1: Track R (28 steps) and Track F (24 steps) pure evidence physically overflows 1,500 tokens."""
    gen = ScenarioGenerator(distractor_mode="mixed", epistemic_stress=True)

    # Track R: 28 distinct timestamped revisions -> pure evidence > 1,500 tokens
    sc_r = gen.generate_scenario("R", scenario_idx=0, split="dev", seed=20261122, filler_count=240)
    ev_r = [s for s in sc_r.sessions if s.get("is_evidence")]
    assert len(ev_r) == STRESS_REVISION_COUNT == 28
    ev_r_tokens = count_tokens("\n\n".join(format_session_text(s) for s in ev_r))
    assert ev_r_tokens > 1500, f"Stress Track R pure evidence ({ev_r_tokens}) must exceed 1,500-token budget"
    q_count = next(q for q in sc_r.questions if q.question_subkind == "count")
    assert q_count.gold_answer == 27
    assert len(sc_r.metadata["all_stale_values"]) == 27

    # Track F: 24 dated sessions -> pure evidence > 1,500 tokens (both slipping and on-time control)
    sc_f_slip = gen.generate_scenario("F", scenario_idx=0, split="dev", seed=20261122, filler_count=240)
    ev_f_slip = [s for s in sc_f_slip.sessions if s.get("is_evidence")]
    assert len(ev_f_slip) == STRESS_FORECAST_EVENT_COUNT == 24
    ev_f_tokens = count_tokens("\n\n".join(format_session_text(s) for s in ev_f_slip))
    assert ev_f_tokens > 1500, f"Stress Track F pure evidence ({ev_f_tokens}) must exceed 1,500-token budget"
    q_moves = next(q for q in sc_f_slip.questions if q.question_subkind == "move_count")
    q_days = next(q for q in sc_f_slip.questions if q.question_subkind == "slip_days")
    assert q_moves.gold_answer == 22
    assert q_days.gold_answer > 150

    # On-time control template (f_09) stays 0 slips / 0 slip days even with 24 sessions
    sc_f_ontime = gen.generate_scenario("F", scenario_idx=9, split="dev", seed=20261122, filler_count=240)
    assert next(q for q in sc_f_ontime.questions if q.question_subkind == "move_count").gold_answer == 0
    assert next(q for q in sc_f_ontime.questions if q.question_subkind == "slip_days").gold_answer == 0


def test_v2_2_upgrade2_multihop_track_d_zero_trigger_terminal_overlap():
    """Upgrade #2: All 12 stress Track D chains have 3 hops (5 evidence sessions) and 0 trigger/terminal overlap."""
    assert len(STRESS_MULTIHOP_DEPENDENCY_POOLS) == 12
    gen = ScenarioGenerator(distractor_mode="mixed", epistemic_stress=True)

    for idx, item in enumerate(STRESS_MULTIHOP_DEPENDENCY_POOLS):
        ov = _content_words(item["hop4_trigger"]) & _content_words(item["hop3_terminal"])
        assert len(ov) == 0, f"{item['id']}: trigger & terminal overlap {ov}"

        sc = gen.generate_scenario("D", scenario_idx=idx, split="dev", seed=20261122, filler_count=20)
        ev_ids = [s["session_id"] for s in sc.sessions if s.get("is_evidence")]
        assert ev_ids == ["s_hop1_premise", "s_hop2_bridge", "s_hop3_terminal", "s_unaffected", "s_hop4_trigger"]
        assert sc.metadata["hop_count"] == 3
        assert sc.metadata["bridge_sessions"] == ["s_hop2_bridge"]
        assert sc.metadata["terminal_sessions"] == ["s_hop3_terminal"]


def test_v2_2_upgrade3_track_x_strictly_zero_content_word_overlap():
    """Upgrade #3: All 12 stress Track X templates enforce EXACTLY 0 content-word overlap."""
    assert len(STRESS_ZERO_OVERLAP_CONSTRAINT_POOLS) == 12
    for item in STRESS_ZERO_OVERLAP_CONSTRAINT_POOLS:
        ov = _content_words(item["constraint_statement"]) & _content_words(item["query"])
        assert len(ov) == 0, f"{item['id']}: non-zero overlap {ov}"


def test_v2_2_upgrade4_track_i_portfolio_aggregation_exceeds_budget():
    """Upgrade #4: All 12 stress Track I portfolio scenarios have 16 sessions overflowing 1,500 tokens."""
    assert len(STRESS_PORTFOLIO_INTERVENTION_POOLS) == 12
    gen = ScenarioGenerator(distractor_mode="mixed", epistemic_stress=True)
    sc_i = gen.generate_scenario("I", scenario_idx=0, split="dev", seed=20261122, filler_count=240)
    ev_i = [s for s in sc_i.sessions if s.get("is_evidence")]
    assert len(ev_i) == STRESS_PORTFOLIO_SESSION_COUNT == 16
    ev_i_tokens = count_tokens("\n\n".join(format_session_text(s) for s in ev_i))
    assert ev_i_tokens > 1500, f"Stress Track I pure evidence ({ev_i_tokens}) must exceed 1,500-token budget"
    assert len(sc_i.questions[0].metadata["required_evidence_phrases"]) == 4


def test_v2_2_stress_chronology_and_manifest_matches_committed():
    """All 9 stress tracks maintain strict chronology and reproduce manifests/stress_manifest.json identically."""
    validate_pools()
    gen = ScenarioGenerator(distractor_mode="mixed", epistemic_stress=True)
    for track in VALID_TRACKS:
        sc = gen.generate_scenario(track, scenario_idx=0, split="dev", seed=20261122, filler_count=240)
        q_dt = parse_session_date(sc.question_date)
        dts = [parse_session_date(s["session_date"]) for s in sc.sessions]
        assert dts == sorted(dts), f"Stress track {track} not chronological"
        assert all(d < q_dt for d in dts), f"Stress track {track} session postdates question_date"
        total_toks = count_tokens("\n\n".join(format_session_text(s) for s in sc.sessions))
        assert total_toks >= 22000, f"Stress track {track} total tokens too small: {total_toks}"

    committed_path = REPO_ROOT / "manifests" / "stress_manifest.json"
    assert committed_path.is_file(), "Missing committed manifests/stress_manifest.json"
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    regenerated = generate_manifest(
        split=committed["split"],
        seed=committed["seed"],
        scenarios_per_track=committed["scenarios_per_track"],
        filler_count=committed["filler_count"],
        tracks=committed["tracks"],
        profile=committed["profile"],
        distractor_mode=committed["distractor_mode"],
    )
    assert committed == regenerated
    assert committed["profile"] == "stress"
    assert committed["filler_count"] == 240
    assert committed["total_scenarios"] == 108


@pytest.mark.asyncio
async def test_v2_2_upgrade5_zero_cost_offline_epistemic_eval_runner(tmp_path: Path):
    """Upgrade #5: Zero-cost offline epistemic-hygiene runner scores all metrics cleanly with $0 spend."""
    out_file = tmp_path / "offline_stress.json"
    report = await run_offline_epistemic_eval(
        profile="stress",
        tracks=["R", "D", "X", "F", "I", "E"],
        systems=["null_memory", "oracle", "recent_context", "vector_rag", "hybrid_rag"],
        scenarios_per_track=1,
        filler_count=30,
        use_fake_embedder=True,
        output_path=out_file,
    )
    assert out_file.is_file()
    assert report["total_llm_spend_usd"] == 0.0
    assert report["profile"] == "stress"
    assert report["epistemic_stress"] is True

    # Oracle achieves 100% session recall, 100% bridge recall, 100% overflow coverage, 0% Track E leak
    oracle = report["systems"]["oracle"]
    assert oracle["gold_session_recall_pct"] == 100.0
    assert oracle["bridge_hop_recall_pct"] == 100.0
    assert oracle["overflow_chain_coverage_pct"] == 100.0
    assert oracle["track_e_leak_rate_pct"] == 0.0
    assert oracle["track_e_control_retention_pct"] == 100.0
    assert oracle["as_of_violations"] == 0

    # Flat RAG physically cannot fit all 28/24/16 overflow evidence sessions inside 1,500 tokens!
    for rag in ("vector_rag", "hybrid_rag"):
        s = report["systems"][rag]
        assert s["as_of_violations"] == 0
        assert s["budget_overshoots"] == 0
        assert s["max_context_tokens"] <= 1500
        assert s["overflow_complete_chain_pct"] == 0.0
        assert s["overflow_chain_coverage_pct"] < 100.0

    table = format_offline_report(report)
    assert "EPIBENCH OFFLINE EPISTEMIC-HYGIENE REPORT" in table
    assert "vector_rag" in table and "hybrid_rag" in table


@pytest.mark.asyncio
async def test_v2_2_stress_offline_validity_gate(tmp_path: Path):
    """Offline validity gate executes cleanly on --profile stress."""
    rep = await run_validity_gate(
        tracks=["R", "D", "X", "F", "I", "P", "K", "E", "Q"],
        scenarios_per_track=1,
        filler_count=20,
        offline=True,
        profile="stress",
        output_dir=tmp_path / "val_stress",
    )
    assert rep["profile"] == "stress"
    assert len(rep["tracks"]) == 9


def test_v2_2_stress_offline_cli(tmp_path: Path):
    """`epibench eval-offline` CLI executes cleanly end-to-end on --profile stress."""
    rc = cli_main([
        "eval-offline",
        "--profile", "stress",
        "--tracks", "R,D,X",
        "--systems", "oracle,vector_rag",
        "--scenarios-per-track", "1",
        "--filler-count", "10",
        "--fake-embedder",
        "--output", str(tmp_path / "cli_out.json"),
    ])
    assert rc == 0
    assert (tmp_path / "cli_out.json").is_file()
