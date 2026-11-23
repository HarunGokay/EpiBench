"""Regression tests for Phase 2 generator and ground truth defects (G1-G5)."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from epibench.generator.manifest import generate_manifest
from epibench.generator.pools import (
    ALL_POOLS,
    MIN_TEMPLATES_PER_TRACK,
    REVISION_POOLS,
    compute_forecast_gold,
    validate_pools,
)
from epibench.generator.scenario_generator import (
    VALID_TRACKS,
    ScenarioGenerator,
    derive_seed,
    parse_session_date,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_G1_track_f_titan_computed_53_days():
    """G1: Track F Titan slip days is computed from ISO event dates as 53 (2024 leap year)."""
    gen = ScenarioGenerator()
    sc = gen.generate_scenario("F", scenario_idx=0, split="dev", seed=20261122, filler_count=6)
    assert sc.metadata["template_id"] == "f_00_titan_alpha"
    q_moves = next(q for q in sc.questions if q.question_subkind == "move_count")
    q_days = next(q for q in sc.questions if q.question_subkind == "slip_days")
    assert q_moves.gold_answer == 2
    assert q_days.gold_answer == 53

    # Also verify non-leap year 2023 computes 52 days for Feb 1 -> Mar 25
    non_leap = compute_forecast_gold([
        {"date": "2023-01-10", "event_kind": "commitment", "target_date": "2023-02-01"},
        {"date": "2023-02-20", "event_kind": "slip", "target_date": "2023-03-25"},
        {"date": "2023-03-25", "event_kind": "completion", "target_date": "2023-03-25"},
    ])
    assert non_leap["total_slip_days"] == 52
    assert non_leap["move_count"] == 1


def test_G2_track_r_pool6_distinct_values():
    """G2: Track R pool 6 has three strictly distinct dates (Dec 15, Jan 15, Feb 15)."""
    pool6 = REVISION_POOLS[6]
    assert pool6["values"] == ["December 15", "January 15", "February 15"]
    assert len(set(pool6["values"])) == 3

    gen = ScenarioGenerator()
    sc = gen.generate_scenario("R", scenario_idx=6, split="dev", seed=20261122, filler_count=6)
    by_kind = {q.question_subkind: q.gold_answer for q in sc.questions}
    assert by_kind["current"] == "February 15"
    assert by_kind["previous"] == "January 15"
    assert by_kind["as_of"] == "December 15"
    assert by_kind["count"] == 2


@pytest.mark.parametrize("filler_count", [0, 6, 24, 60, 200])
def test_G3_no_session_postdates_question_date(filler_count: int):
    """G3: No session (evidence or filler) ever postdates question_date across all 9 tracks."""
    gen = ScenarioGenerator()
    for track in VALID_TRACKS:
        sc = gen.generate_scenario(
            track=track,
            scenario_idx=0,
            split="dev",
            seed=20261122,
            filler_count=filler_count,
        )
        q_dt = parse_session_date(sc.question_date)
        session_dts = [parse_session_date(s["session_date"]) for s in sc.sessions]
        assert session_dts == sorted(session_dts), f"Sessions not chronological in {track}"
        for s, s_dt in zip(sc.sessions, session_dts):
            assert s_dt < q_dt, (
                f"Track {track} (filler_count={filler_count}): session {s['session_id']} "
                f"dated {s_dt} >= question_date {q_dt}"
            )


def test_G4_seed_derivation_no_collisions():
    """G4: Derived seeds never collide across tracks (including E vs P and K vs Q) or purposes."""
    seeds = set()
    for track in VALID_TRACKS:
        for idx in range(12):
            for purpose in ("fillers", "filler_dates"):
                s = derive_seed(20261122, track, idx, purpose)
                assert s not in seeds
                seeds.add(s)
    assert derive_seed(42, "E", 0, "fillers") != derive_seed(42, "P", 0, "fillers")
    assert derive_seed(42, "K", 0, "fillers") != derive_seed(42, "Q", 0, "fillers")


def test_G5_validate_pools_and_held_out_guard():
    """G5: validate_pools passes, >=12 curated templates/track, held_out requires private pool."""
    counts = validate_pools()
    for track in VALID_TRACKS:
        assert counts[track] >= MIN_TEMPLATES_PER_TRACK
    assert counts["R"] >= 16
    assert counts["D"] >= 14

    gen = ScenarioGenerator()
    with pytest.raises(ValueError, match="held_out"):
        gen.generate_scenario("R", scenario_idx=0, split="held_out", seed=42)

    # Verify every string_match question's gold (or alias) appears in its evidence text
    for track, pool in ALL_POOLS.items():
        for idx in range(len(pool)):
            sc = gen.generate_scenario(track, scenario_idx=idx, split="dev", seed=20261122, filler_count=2)
            ev_text = " ".join(
                (t.get("content") or "")
                for s in sc.sessions
                if s.get("is_evidence")
                for t in s.get("turns", [])
            ).lower()
            for q in sc.questions:
                if q.scoring_type == "string_match":
                    aliases = q.metadata.get("gold_aliases", [str(q.gold_answer)])
                    assert any(a.lower() in ev_text for a in aliases), (
                        f"{q.question_id}: none of {aliases} found in evidence text"
                    )


def test_manifest_reproducible_and_matches_committed():
    """Phase 2 Gate: scenario manifest is deterministic across runs and matches committed file."""
    m1 = generate_manifest(split="dev", seed=20261122, scenarios_per_track=12, filler_count=6)
    m2 = generate_manifest(split="dev", seed=20261122, scenarios_per_track=12, filler_count=6)
    assert m1 == m2
    assert m1["manifest_sha256"] == m2["manifest_sha256"]
    assert m1["total_scenarios"] == 12 * len(VALID_TRACKS)

    committed_path = REPO_ROOT / "manifests" / "dev_manifest.json"
    assert committed_path.exists(), "Committed manifests/dev_manifest.json is missing"
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    assert committed["manifest_sha256"] == m1["manifest_sha256"]
    assert committed == m1
