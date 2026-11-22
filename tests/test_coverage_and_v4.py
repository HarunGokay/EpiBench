"""Comprehensive function coverage tests across judging, metrics, generator, adapters, and V4."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random
import pytest

from epibench.adapters.base import (
    filter_sessions_as_of,
    format_session_text,
    is_on_or_before_as_of,
    parse_as_of_cutoff,
)
from epibench.adapters.hybrid_rag_adapter import BM25Index
from epibench.adapters.reference_adapters import NullMemoryAdapter
from epibench.generator.distractor_pool import DistractorPool
from epibench.generator.manifest import (
    canonical_json_bytes,
    generate_manifest,
    scenario_sha256,
    write_manifest,
)
from epibench.generator.scenario_generator import (
    allocate_filler_dates,
    format_date,
    parse_session_date,
)
from epibench.judging.programmatic import (
    extract_final_answer,
    parse_single_integer,
)
from epibench.metrics.eval_metrics import map_evaluation_to_dimension
from epibench.metrics.schema import load_results_schema

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_V4_invalid_results_archived_and_current_manifest_reproducible():
    """V4: Pre-v2 invalid results are archived with disclaimer; committed dev manifest reproduces identically."""
    archive_dir = REPO_ROOT / "results_archive_invalid"
    archive_readme = archive_dir / "README.md"
    assert archive_dir.is_dir()
    assert archive_readme.is_file()
    text = archive_readme.read_text(encoding="utf-8").lower()
    assert "do not cite" in text
    assert "buggy scorers" in text

    committed = json.loads((REPO_ROOT / "manifests" / "dev_manifest.json").read_text(encoding="utf-8"))
    regenerated = generate_manifest(
        split=committed["split"],
        seed=committed["seed"],
        scenarios_per_track=committed["scenarios_per_track"],
        filler_count=committed["filler_count"],
        tracks=committed["tracks"],
    )
    assert committed == regenerated


def test_programmatic_extract_and_integer_helpers():
    """Direct tests for extract_final_answer and parse_single_integer."""
    val, ok = extract_final_answer("Reasoning.\nFINAL ANSWER: Equinox.")
    assert ok is True and val == "Equinox"
    val_num, ok_num = extract_final_answer("Reasoning.\nFINAL ANSWER: 42")
    assert ok_num is True and val_num == "42"

    # Last marker wins
    val2, ok2 = extract_final_answer("FINAL ANSWER: wrong\nActually:\nFINAL ANSWER: right")
    assert ok2 is True and val2 == "right"

    assert extract_final_answer("no marker here") == (None, False)
    assert extract_final_answer("FINAL ANSWER:   ") == (None, False)

    assert parse_single_integer("53") == 53
    assert parse_single_integer("1,200") == 1200
    assert parse_single_integer("twenty-four") == 24
    assert parse_single_integer("one hundred twenty three") == 123
    assert parse_single_integer("two times") is None


def test_metrics_dimension_mapper_and_schema_loader():
    """Direct tests for map_evaluation_to_dimension and load_results_schema."""
    assert map_evaluation_to_dimension("K", "static_fact") == "factual_accuracy"
    assert map_evaluation_to_dimension("R", "current") == "temporal_consistency"
    assert map_evaluation_to_dimension("F", "slip_days") == "temporal_consistency"
    assert map_evaluation_to_dimension("D", "revisit_set") == "dependency_reasoning"
    assert map_evaluation_to_dimension("X", "hidden_constraint") == "constraint_compliance"
    assert map_evaluation_to_dimension("P", "provenance") == "provenance"
    assert map_evaluation_to_dimension("I", "causal_claim") == "calibration"
    assert map_evaluation_to_dimension("Q", "clarification") == "calibration"
    assert map_evaluation_to_dimension("E", "deletion_joint") == "privacy_and_deletion"
    assert map_evaluation_to_dimension("E", "deletion_control_retention") == "retention_control"
    assert map_evaluation_to_dimension("UNKNOWN", "other") is None

    schema = load_results_schema()
    assert schema["type"] == "object"
    assert "scorer_version" in schema["required"]


def test_generator_helpers_and_write_manifest(tmp_path: Path):
    """Direct tests for DistractorPool, format_date, allocate_filler_dates, canonical_json_bytes, write_manifest."""
    dp = DistractorPool.get_instance()
    assert dp.size >= 30
    sample = dp.sample_distractors(5, seed=123)
    assert len(sample) == 5

    dt = datetime(2024, 3, 15, 10, 30, 0, tzinfo=timezone.utc)
    assert format_date(dt) == "2024-03-15 10:30:00"
    assert parse_session_date("2024/03/15 10:30:00") == dt

    q_dt = datetime(2024, 4, 1, 10, 0, 0, tzinfo=timezone.utc)
    f_dates = allocate_filler_dates([dt], q_dt, 10, random.Random(7))
    assert len(f_dates) == 10
    assert all(d < q_dt for d in f_dates)

    raw = canonical_json_bytes({"b": 2, "a": 1})
    assert raw == b'{"a":1,"b":2}'
    assert len(scenario_sha256({"x": 1})) == 64

    manifest = generate_manifest(split="dev", seed=42, scenarios_per_track=1, filler_count=1, tracks=["R"])
    out = write_manifest(manifest, tmp_path / "m.json")
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8")) == manifest


@pytest.mark.asyncio
async def test_adapter_base_helpers_and_bm25_index():
    """Direct tests for adapter base date/format helpers, BM25Index, and ask_clarification."""
    cutoff = parse_as_of_cutoff("2024-01-25")
    assert cutoff is not None and cutoff.hour == 23 and cutoff.minute == 59
    assert is_on_or_before_as_of("2024-01-25 10:00:00", "2024-01-25") is True
    assert is_on_or_before_as_of("2024-01-26 00:00:00", "2024-01-25") is False
    assert is_on_or_before_as_of(None, "2024-01-25") is False

    sessions = [
        {"session_id": "s1", "session_date": "2024-01-10 10:00:00", "turns": [{"role": "user", "content": "Hello"}]},
        {"session_id": "s2", "session_date": "2024-02-10 10:00:00", "turns": [{"role": "user", "content": "Later"}]},
    ]
    kept = filter_sessions_as_of(sessions, "2024-01-25")
    assert [s["session_id"] for s in kept] == ["s1"]
    assert "[2024-01-10 10:00:00] User: Hello" in format_session_text(sessions[0])

    bm25 = BM25Index()
    bm25.index(["alpha beta gamma", "delta epsilon"])
    scores = bm25.score("alpha gamma")
    assert scores[0] > scores[1]

    null_adapter = NullMemoryAdapter()
    assert await null_adapter.ask_clarification("t", "q") == []
