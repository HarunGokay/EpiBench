"""Regression and gate tests for Phase 6 metrics, statistics, and results schema (M1-M5)."""

from __future__ import annotations

from pathlib import Path
import pytest

from epibench.metrics.eval_metrics import (
    DEFAULT_DIMENSION_WEIGHTS,
    DIMENSIONS,
    compute_dimensional_report,
)
from epibench.metrics.schema import validate_results_schema
from epibench.metrics.statistics import (
    exact_mcnemar_test,
    holm_bonferroni_correction,
    paired_scenario_bootstrap,
    wilson_ci,
)
from epibench.runners.fakes import FakeJudge, FakeReader
from epibench.runners.metering import SpendTracker
from epibench.runners.run_epibench import EpiBenchRunner


def test_M1_missing_dimensions_score_null_and_renormalize_composite():
    """M1: Un-evaluated dimensions score None (null), never 100.0; composite renormalizes weights."""
    only_r = [
        {"track": "R", "question_subkind": "current", "is_correct": True, "retrieval_latency_ms": 5.0, "context_tokens": 100},
        {"track": "R", "question_subkind": "as_of", "is_correct": False, "retrieval_latency_ms": 5.0, "context_tokens": 100},
    ]
    rep = compute_dimensional_report(only_r, "sys_r_only", total_spend_usd=0.01)
    assert rep.temporal_consistency == 50.0
    assert rep.factual_accuracy is None
    assert rep.dependency_reasoning is None
    assert rep.constraint_compliance is None
    assert rep.provenance is None
    assert rep.calibration is None
    assert rep.privacy_and_deletion is None
    assert rep.retention_control is None
    assert rep.included_dimensions == ["temporal_consistency"]
    assert len(rep.excluded_dimensions) == len(DIMENSIONS) - 1
    assert rep.composite_score == 50.0

    # Empty evaluations yield composite=None
    empty_rep = compute_dimensional_report([], "empty_sys")
    assert empty_rep.composite_score is None
    assert empty_rep.included_dimensions == []

    # Experimental tracks are excluded from composite
    exp_rep = compute_dimensional_report(only_r, "exp_sys", experimental_tracks=["R"])
    assert exp_rep.temporal_consistency is None
    assert exp_rep.composite_score is None


def test_M2_track_x_constraint_compliance_and_track_e_retention_separated():
    """M2: Track X maps to constraint_compliance; E-control maps to retention_control, never privacy."""
    evals = [
        {"track": "K", "question_subkind": "static_fact", "is_correct": True, "retrieval_latency_ms": 1.0, "context_tokens": 50},
        {"track": "X", "question_subkind": "hidden_constraint", "is_correct": False, "retrieval_latency_ms": 1.0, "context_tokens": 50},
        {"track": "E", "question_subkind": "deletion_joint", "is_correct": False},
        {"track": "E", "question_subkind": "deletion_control_retention", "is_correct": True, "retrieval_latency_ms": 1.0, "context_tokens": 50},
    ]
    rep = compute_dimensional_report(evals, "test_sys")
    assert rep.factual_accuracy == 100.0
    assert rep.constraint_compliance == 0.0
    assert rep.privacy_and_deletion == 0.0
    assert rep.retention_control == 100.0
    assert set(rep.included_dimensions) == {"factual_accuracy", "constraint_compliance", "privacy_and_deletion"}

    # Check exact renormalized composite
    w_k = DEFAULT_DIMENSION_WEIGHTS["factual_accuracy"]
    w_x = DEFAULT_DIMENSION_WEIGHTS["constraint_compliance"]
    w_e = DEFAULT_DIMENSION_WEIGHTS["privacy_and_deletion"]
    expected_comp = round(100.0 * (w_k / (w_k + w_x + w_e)), 2)
    assert rep.composite_score == expected_comp


class _MeteredFakeReader(FakeReader):
    def __init__(self, tracker: SpendTracker):
        super().__init__(model_name="metered-fake-reader")
        self._tracker = tracker

    async def answer_question(self, context: str, question: str, question_date: str = "", scoring_type: str = "llm_rubric") -> str:
        self._tracker.record(10000, 2000, stage="reader_llm")
        return await super().answer_question(context, question, question_date=question_date, scoring_type=scoring_type)


@pytest.mark.asyncio
async def test_M3_cost_per_successful_answer_real_spend(tmp_path: Path):
    """M3: Real per-system spend is passed into dimensional reports and cost_per_successful_answer_usd > 0."""
    rep = compute_dimensional_report(
        [{"track": "K", "question_subkind": "static_fact", "is_correct": True} for _ in range(5)],
        "cost_sys",
        total_spend_usd=0.25,
    )
    assert rep.cost_per_successful_answer_usd == pytest.approx(0.05)

    tracker = SpendTracker(hard_cap_usd=5.0)
    runner = EpiBenchRunner(
        spend_tracker=tracker,
        output_dir=tmp_path,
        reader=_MeteredFakeReader(tracker),
        judge=FakeJudge(),
        use_fakes=True,
        secondary_audit_rate=0.0,
    )
    out = await runner.run_benchmark(
        tracks=["K", "R"],
        systems=["oracle", "null_memory"],
        scenarios_per_track=2,
        filler_count=2,
        resume=False,
        save_checkpoints=False,
    )
    ora_sys = out["performance_summary"]["by_system"]["oracle"]
    ora_dim = out["performance_summary"]["dimensional_reports"]["oracle"]
    assert ora_sys["total_spend_usd"] > 0.0
    assert ora_sys["cost_per_successful_answer_usd"] is not None
    assert ora_sys["cost_per_successful_answer_usd"] > 0.0
    assert ora_dim["cost_per_successful_answer_usd"] == pytest.approx(ora_sys["cost_per_successful_answer_usd"], rel=1e-3)

    # Null memory got 0 correct answers -> cost_per_successful_answer_usd is None
    null_sys = out["performance_summary"]["by_system"]["null_memory"]
    assert null_sys["total_spend_usd"] > 0.0
    assert null_sys["cost_per_successful_answer_usd"] is None


@pytest.mark.asyncio
async def test_M4_comparisons_and_json_schema_validation(tmp_path: Path):
    """M4 + Phase 6 Gate: McNemar, paired bootstrap, Holm correction computed and output validates against schema."""
    # Known exact statistical values
    mc = exact_mcnemar_test(5, 0)
    assert mc["p_value"] == pytest.approx(0.0625, abs=1e-6)
    assert mc["odds_ratio"] is None
    assert mc["odds_ratio_infinite"] is True

    mc_sym = exact_mcnemar_test(4, 4)
    assert mc_sym["p_value"] == pytest.approx(1.0)
    assert mc_sym["odds_ratio"] == pytest.approx(1.0)

    holm = holm_bonferroni_correction([("p1", 0.01), ("p2", 0.04), ("p3", 0.20)])
    by_name = {h["name"]: h for h in holm}
    assert by_name["p1"]["adjusted_p"] == pytest.approx(0.03)
    assert by_name["p1"]["is_significant_05"] is True
    assert by_name["p2"]["adjusted_p"] == pytest.approx(0.08)
    assert by_name["p2"]["is_significant_05"] is False

    boot = paired_scenario_bootstrap([1.0] * 10, [0.0] * 10, n_resamples=2000, seed=42)
    assert boot["mean_diff"] == pytest.approx(1.0)
    assert boot["ci_lower"] == pytest.approx(1.0)
    assert boot["p_value"] == pytest.approx(0.0)

    ci = wilson_ci(8, 10)
    assert 0.45 < ci["lower"] < 0.8 < ci["upper"] < 0.98

    # End-to-end dry run with fakes -> non-empty comparisons + strict JSON schema validation
    runner = EpiBenchRunner(
        spend_tracker=SpendTracker(hard_cap_usd=2.0),
        output_dir=tmp_path,
        use_fakes=True,
        secondary_audit_rate=0.5,
    )
    out = await runner.run_benchmark(
        tracks=["R", "D", "X", "E"],
        systems=["oracle", "recent_context", "null_memory"],
        scenarios_per_track=3,
        filler_count=4,
        resume=False,
        save_checkpoints=False,
    )
    validate_results_schema(out)

    comps = out["performance_summary"]["comparisons"]
    assert len(comps) == 3  # 3 choose 2 pairs
    assert "oracle_vs_null_memory" in comps
    ora_vs_null = comps["oracle_vs_null_memory"]
    assert ora_vs_null["mcnemar"]["b"] > 0
    assert ora_vs_null["mcnemar"]["c"] == 0
    assert ora_vs_null["mcnemar"]["holm_significant_05"] is True
    assert ora_vs_null["paired_scenario_bootstrap"]["mean_diff"] > 0.9
    assert ora_vs_null["paired_scenario_bootstrap"]["holm_significant_05"] is True
