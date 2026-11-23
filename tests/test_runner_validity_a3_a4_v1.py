"""Regression and gate tests for Phase 5 runner, judge, deletion replay, and validity gate (A3, A4, V1, M5)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pytest

from epibench.adapters.base import BaseMemoryAdapter, IngestResult, RetrievalResult
from epibench.adapters.reference_adapters import OracleAdapter
from epibench.generator.scenario_generator import ScenarioGenerator
from epibench.judging.rubric_judge import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_SECONDARY_AUDIT_RATE,
    RubricJudge,
    compute_cohens_kappa,
    parse_judge_json_strict,
)
from epibench.runners.fakes import FakeJudge, FakeReader
from epibench.runners.metering import SpendTracker
from epibench.runners.run_epibench import EpiBenchRunner
from epibench.runners.validate import run_validity_gate


class _NoDeleteAdapter(BaseMemoryAdapter):
    """Adapter that intentionally does not override delete() (uses default unsupported)."""

    def __init__(self):
        super().__init__("no_delete")
        self._sessions: Dict[str, List[Dict[str, Any]]] = {}

    async def ingest(self, tenant_id: str, sessions: List[Dict[str, Any]]) -> IngestResult:
        self._sessions.setdefault(tenant_id, []).extend(sessions)
        return IngestResult(tenant_id=tenant_id, session_count=len(sessions), latency_ms=0.1)

    async def retrieve(
        self, tenant_id: str, query: str, token_budget: int = 1500, as_of: Optional[str] = None
    ) -> RetrievalResult:
        lines = []
        sids = []
        for s in self._sessions.get(tenant_id, []):
            sids.append(s["session_id"])
            for t in s.get("turns", []):
                lines.append(f"[{s['session_date']}] {t['role'].capitalize()}: {t['content']}")
        ctx = "\n".join(lines)
        return RetrievalResult(context=ctx, token_count=len(ctx) // 4, retrieval_latency_ms=0.1, metadata={"source_session_ids": sids})


class _UnverifiedAndLeakyAdapter(BaseMemoryAdapter):
    """Adapter that either omits source_session_ids or leaks a future session."""

    def __init__(self, omit_ids: bool = False):
        super().__init__("unverified_or_leaky")
        self.omit_ids = omit_ids
        self._sessions: Dict[str, List[Dict[str, Any]]] = {}

    async def ingest(self, tenant_id: str, sessions: List[Dict[str, Any]]) -> IngestResult:
        self._sessions.setdefault(tenant_id, []).extend(sessions)
        return IngestResult(tenant_id=tenant_id, session_count=len(sessions), latency_ms=0.1)

    async def retrieve(
        self, tenant_id: str, query: str, token_budget: int = 1500, as_of: Optional[str] = None
    ) -> RetrievalResult:
        sessions = self._sessions.get(tenant_id, [])
        meta = {} if self.omit_ids else {"source_session_ids": [s["session_id"] for s in sessions]}
        return RetrievalResult(context="some context", token_count=2, retrieval_latency_ms=0.1, metadata=meta)


class _BrokenFormatReader:
    model_name = "broken-format-reader"

    async def answer_question(self, context: str, question: str, question_date: str = "", scoring_type: str = "llm_rubric") -> str:
        return "PostgreSQL 16 (without the required marker)"


class _BrokenJsonJudge(RubricJudge):
    def __init__(self):
        super().__init__(api_key="unused", model_name="broken-judge")
        self.call_count = 0

    async def _call_gemini(self, prompt: str, stage: str = "rubric_judge") -> str:
        self.call_count += 1
        return "I think this looks CORRECT probably (not valid json)"


@pytest.mark.asyncio
async def test_A3_runner_replays_deletion_and_flags_unsupported(tmp_path: Path):
    """A3: Runner triggers adapter.delete(target=...) on Track E and flags unsupported adapters."""
    no_del = _NoDeleteAdapter()
    oracle = OracleAdapter()
    runner = EpiBenchRunner(
        spend_tracker=SpendTracker(hard_cap_usd=1.0),
        output_dir=tmp_path,
        use_fakes=True,
        custom_adapters={"no_delete": no_del, "oracle": oracle},
    )
    gen = ScenarioGenerator()
    sc = gen.generate_scenario("E", scenario_idx=0, split="dev", seed=20261122, filler_count=4)
    res = await runner.evaluate_scenario(sc, {"no_delete": no_del, "oracle": oracle})

    # Oracle supports delete -> joint pass True, delete_unsupported False
    ora_res = res["system_results"]["oracle"]
    assert ora_res["delete_unsupported"] is False
    assert len(ora_res["delete_events"]) == 1
    assert ora_res["delete_events"][0]["deleted"] is True
    assert ora_res["track_e_joint"]["joint_pass"] is True

    # _NoDeleteAdapter does not support delete -> flagged delete_unsupported=True and leaks secret
    nd_res = res["system_results"]["no_delete"]
    assert nd_res["delete_unsupported"] is True
    assert nd_res["track_e_joint"]["leak_free"] is False
    assert nd_res["track_e_joint"]["joint_pass"] is False


@pytest.mark.asyncio
async def test_A4_reader_receives_true_question_date_not_as_of(tmp_path: Path):
    """A4: Reader receives true scenario.question_date even for as_of questions."""
    fake_reader = FakeReader()
    oracle = OracleAdapter()
    runner = EpiBenchRunner(
        spend_tracker=SpendTracker(hard_cap_usd=1.0),
        output_dir=tmp_path,
        reader=fake_reader,
        judge=FakeJudge(),
        use_fakes=True,
    )
    gen = ScenarioGenerator()
    sc = gen.generate_scenario("R", scenario_idx=0, split="dev", seed=20261122, filler_count=4)
    as_of_q = next(q for q in sc.questions if q.question_subkind == "as_of")
    assert as_of_q.metadata["as_of_date"] != sc.question_date

    await runner.evaluate_scenario(sc, {"oracle": oracle})
    assert len(fake_reader.calls) == len(sc.questions)
    for call in fake_reader.calls:
        assert call["question_date"] == sc.question_date
        assert call["question_date"] != as_of_q.metadata["as_of_date"]


@pytest.mark.asyncio
async def test_runner_as_of_audit_flags_unverified_and_violations(tmp_path: Path):
    """Runner flags missing source_session_ids as as_of_unverified and detects future leakage."""
    gen = ScenarioGenerator()
    sc = gen.generate_scenario("R", scenario_idx=0, split="dev", seed=20261122, filler_count=2)

    unverified = _UnverifiedAndLeakyAdapter(omit_ids=True)
    leaky = _UnverifiedAndLeakyAdapter(omit_ids=False)
    runner = EpiBenchRunner(
        spend_tracker=SpendTracker(hard_cap_usd=1.0),
        output_dir=tmp_path,
        use_fakes=True,
        strict_as_of_audit=False,
    )
    res = await runner.evaluate_scenario(sc, {"unverified": unverified, "leaky": leaky})
    assert res["system_results"]["unverified"]["as_of_unverified_count"] == len(sc.questions)
    assert res["system_results"]["leaky"]["as_of_violation_count"] >= 1

    strict_runner = EpiBenchRunner(
        spend_tracker=SpendTracker(hard_cap_usd=1.0),
        output_dir=tmp_path,
        use_fakes=True,
        strict_as_of_audit=True,
    )
    with pytest.raises(AssertionError, match="as_of leakage violation"):
        await strict_runner.evaluate_scenario(sc, {"leaky": leaky})


@pytest.mark.asyncio
async def test_runner_tracks_parse_failures_and_judge_errors(tmp_path: Path):
    """Missing FINAL ANSWER increments parse_failures; unparseable judge JSON retries once and increments judge_errors."""
    broken_judge = _BrokenJsonJudge()
    runner = EpiBenchRunner(
        spend_tracker=SpendTracker(hard_cap_usd=1.0),
        output_dir=tmp_path,
        reader=_BrokenFormatReader(),
        judge=broken_judge,
        secondary_audit_rate=0.0,
    )
    gen = ScenarioGenerator()

    # Programmatic parse failure test (Track R)
    sc_r = gen.generate_scenario("R", scenario_idx=0, split="dev", seed=42, filler_count=2)
    res_r = await runner.evaluate_scenario(sc_r, {"oracle": OracleAdapter()})
    assert res_r["system_results"]["oracle"]["parse_failures"] == len(sc_r.questions)
    assert res_r["system_results"]["oracle"]["correct_questions"] == 0

    # Strict judge JSON retry and error counting test (Track X)
    sc_x = gen.generate_scenario("X", scenario_idx=0, split="dev", seed=42, filler_count=2)
    res_x = await runner.evaluate_scenario(sc_x, {"oracle": OracleAdapter()})
    assert broken_judge.call_count == 2  # 1 initial + 1 retry
    assert res_x["system_results"]["oracle"]["judge_errors"] == 1
    q_res = res_x["system_results"]["oracle"]["questions"][0]
    assert q_res["judge_error"] is True
    assert q_res["evaluation_details"]["verdict"] == "ERROR"
    assert q_res["is_correct"] is False


@pytest.mark.asyncio
async def test_M5_secondary_audit_rate_distinct_models_and_kappa(tmp_path: Path, caplog):
    """M5: Default audit rate is 10%, judge defaults distinct from reader, warns when equal, computes kappa."""
    assert DEFAULT_SECONDARY_AUDIT_RATE == 0.10
    assert DEFAULT_JUDGE_MODEL != "gemini-3.5-flash-lite"

    with caplog.at_level(logging.WARNING, logger="epibench_judge"):
        _ = RubricJudge(api_key="x", model_name="gemini-3.5-flash-lite", reader_model_name="gemini-3.5-flash-lite")
    assert any("identical to reader model" in rec.message for rec in caplog.records)

    # Strict JSON parser unit checks
    ok_obj, err = parse_judge_json_strict('```json\n{"verdict": "CORRECT", "reasoning": "Good."}\n```')
    assert err is None and ok_obj == {"verdict": "CORRECT", "reasoning": "Good."}
    bad_obj, err2 = parse_judge_json_strict("Looks CORRECT to me")
    assert bad_obj is None and err2 is not None

    # Cohen's kappa exact checks
    k_perfect = compute_cohens_kappa([(True, True), (False, False), (True, True), (False, False)])
    assert k_perfect["agreement_rate"] == 100.0
    assert k_perfect["cohens_kappa"] == 1.0

    # 6/8 agreement (2 disagreements out of 8, 4 TT, 2 FF, 1 TF, 1 FT -> po=0.75, pe=0.53125 -> kappa=0.4667)
    k_75 = compute_cohens_kappa([
        (True, True), (True, True), (True, True), (True, True),
        (False, False), (False, False), (True, False), (False, True),
    ])
    assert k_75["agreement_rate"] == 75.0
    assert k_75["cohens_kappa"] == pytest.approx(0.4667, abs=0.001)

    # Runner end-to-end with secondary_audit_rate=1.0 populates secondary_audit summary
    runner = EpiBenchRunner(
        spend_tracker=SpendTracker(hard_cap_usd=1.0),
        output_dir=tmp_path,
        use_fakes=True,
        secondary_audit_rate=1.0,
    )
    out = await runner.run_benchmark(
        tracks=["X", "P"],
        systems=["null_memory", "oracle"],
        scenarios_per_track=2,
        filler_count=2,
        resume=False,
        save_checkpoints=False,
    )
    sec = out["performance_summary"]["secondary_audit"]
    assert sec["audit_count"] == 8
    assert sec["agreement_rate"] == 100.0
    assert sec["cohens_kappa"] == 1.0


@pytest.mark.asyncio
async def test_V1_validity_gate_offline_all_tracks_pass_and_track_e_joint(tmp_path: Path):
    """V1: Offline validity gate passes all 9 tracks with oracle=100%, null=0%, and Track E joint scoring."""
    report = await run_validity_gate(
        scenarios_per_track=12,
        filler_count=4,
        offline=True,
        output_dir=tmp_path,
    )
    assert report["live_verified"] is False
    assert "unverified live" in report["status_note"]
    assert report["all_tracks_passed"] is True
    assert report["experimental_tracks"] == []
    for track, info in report["tracks"].items():
        assert info["status"] == "PASS", f"Track {track} failed offline validity gate: {info}"
        assert info["oracle_score_pct"] >= 95.0
        assert info["null_memory_score_pct"] <= 15.0
    assert report["tracks"]["E"]["scoring_metric"] == "joint_scenario_pass_rate"
    assert report["tracks"]["E"]["null_memory_score_pct"] == 0.0
