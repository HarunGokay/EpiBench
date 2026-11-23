"""Regression and invariant tests for EpiBench v2.1.0 hard-regime & same-domain distractors."""

from __future__ import annotations

import json
from pathlib import Path
import re
import pytest

from epibench.adapters.base import count_tokens, format_session_text
from epibench.generator._pools_xfi import HIDDEN_CONSTRAINT_POOLS
from epibench.generator.distractor_pool import (
    BUNDLED_SAME_DOMAIN_PATH,
    DistractorPool,
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
from epibench.runners.metering import SpendTracker
from epibench.runners.run_epibench import EpiBenchRunner

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_v2_1_same_domain_distractors_loaded_and_clean():
    """Bundled same-domain distractor pool has >=60 curated sessions and passes validate_pools."""
    assert BUNDLED_SAME_DOMAIN_PATH.is_file()
    same_pool = DistractorPool.get_instance(mode="same_domain")
    assert same_pool.size >= 60

    mixed_pool = DistractorPool.get_instance(mode="mixed")
    assert mixed_pool.size == same_pool.size + DistractorPool.get_instance(mode="neutral").size

    # validate_pools verifies BOTH neutral and same_domain pools have 0 entity-gold leaks
    counts = validate_pools()
    assert len(counts) == len(VALID_TRACKS)
    validate_pools(distractor_pool=same_pool)
    validate_pools(distractor_pool=mixed_pool)


def test_v2_1_mixed_sampling_ratio_and_custom_file(tmp_path: Path):
    """Mixed mode yields 50/50 same-domain + neutral fillers deterministically; custom file works."""
    mixed_pool = DistractorPool.get_instance(mode="mixed")
    s1 = mixed_pool.sample_distractors(120, seed=777)
    s2 = mixed_pool.sample_distractors(120, seed=777)
    assert s1 == s2
    same_cnt = sum(1 for x in s1 if str(x.get("session_id", "")).startswith("same_domain_"))
    neutral_cnt = sum(1 for x in s1 if str(x.get("session_id", "")).startswith("bundled_distractor_"))
    assert same_cnt == 60
    assert neutral_cnt == 60

    # Custom distractor file support
    custom_path = tmp_path / "custom_fillers.json"
    custom_path.write_text(
        json.dumps([
            {
                "session_id": "custom_001",
                "turns": [
                    {"role": "user", "content": "How do I tune a banjo?"},
                    {"role": "assistant", "content": "Standard open G tuning is gDGBD."},
                ],
            }
        ]),
        encoding="utf-8",
    )
    gen = ScenarioGenerator(distractor_file=custom_path)
    sc = gen.generate_scenario("K", scenario_idx=0, split="dev", seed=42, filler_count=3)
    assert sc.metadata["distractor_mode"] == "custom"
    assert sum(1 for s in sc.sessions if not s.get("is_evidence")) == 3


def test_v2_1_hard_profile_true_budget_overflow_and_dates():
    """Hard profile generates 120 mixed fillers with >10,000 history tokens vs 1,500 budget and valid dates."""
    assert GENERATOR_VERSION == "2.1.0"
    prof = resolve_profile(profile="hard")
    assert prof["filler_count"] == 120
    assert prof["distractor_mode"] == "mixed"
    assert prof["token_budget"] == 1500
    assert "hard" in BENCHMARK_PROFILES

    gen = ScenarioGenerator(distractor_mode=prof["distractor_mode"])
    token_totals = []
    for track in VALID_TRACKS:
        sc = gen.generate_scenario(
            track=track,
            scenario_idx=0,
            split="dev",
            seed=20261122,
            filler_count=prof["filler_count"],
        )
        q_dt = parse_session_date(sc.question_date)
        dts = [parse_session_date(s["session_date"]) for s in sc.sessions]
        assert dts == sorted(dts)
        assert all(d < q_dt for d in dts)
        full_text = "\n\n".join(format_session_text(s) for s in sc.sessions)
        tok = count_tokens(full_text)
        token_totals.append(tok)
        assert tok >= 10000, f"Track {track} hard profile history too small: {tok} tokens"

    assert sum(token_totals) / len(token_totals) >= 12000


def test_v2_1_hard_manifest_matches_committed():
    """Committed manifests/hard_manifest.json reproduces identically."""
    committed_path = REPO_ROOT / "manifests" / "hard_manifest.json"
    assert committed_path.is_file(), "Missing committed manifests/hard_manifest.json"
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
    assert committed["profile"] == "hard"
    assert committed["distractor_mode"] == "mixed"
    assert committed["filler_count"] == 120


def test_v2_1_track_x_low_lexical_overlap():
    """Hardened Track X templates keep mean content-word overlap <= 0.5 tokens."""
    stop = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "are",
        "was", "were", "be", "my", "our", "we", "i", "me", "you", "it", "that",
        "this", "with", "at", "by", "from", "as", "not", "no", "any", "all",
        "can", "should", "want", "wants", "have", "has", "had", "do", "does",
        "so", "if", "but", "into", "up", "out", "about", "per", "just", "now",
        "us", "her", "his", "their", "them", "would", "could", "will", "go",
        "ahead", "using", "because", "than", "before", "after",
    }
    overlaps = []
    for item in HIDDEN_CONSTRAINT_POOLS:
        c = {w for w in re.findall(r"[a-z0-9]+", item["constraint_statement"].lower()) if w not in stop and len(w) > 2}
        q = {w for w in re.findall(r"[a-z0-9]+", item["query"].lower()) if w not in stop and len(w) > 2}
        overlaps.append(len(c & q))
    assert sum(overlaps) / len(overlaps) <= 0.5


@pytest.mark.asyncio
async def test_v2_1_hard_profile_offline_runner_schema_valid(tmp_path: Path):
    """Offline EpiBenchRunner executes hard profile cleanly and passes JSON Schema validation."""
    tracker = SpendTracker(hard_cap_usd=1.0)
    runner = EpiBenchRunner(
        api_key="",
        spend_tracker=tracker,
        token_budget=1500,
        output_dir=tmp_path,
        use_fakes=True,
        distractor_mode="mixed",
        profile="hard",
    )
    res = await runner.run_benchmark(
        tracks=["R", "X"],
        systems=["null_memory", "oracle", "recent_context", "vector_rag"],
        scenarios_per_track=1,
        split="dev",
        seed=20261122,
        filler_count=120,
        resume=False,
        save_checkpoints=False,
    )
    assert res["generator_version"] == "2.1.0"
    assert res["profile"] == "hard"
    assert res["distractor_mode"] == "mixed"
    assert res["filler_count"] == 120
    assert res["performance_summary"]["by_system"]["recent_context"]["avg_context_tokens"] <= 1500
    assert res["performance_summary"]["by_system"]["vector_rag"]["avg_context_tokens"] <= 1500
