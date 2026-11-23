"""Regression and end-to-end gate tests for Phase 8 depth and efficiency runners (V3)."""

from __future__ import annotations

from pathlib import Path
import pytest

from epibench.runners.run_depth_benchmark import DEFAULT_DEPTH_SYSTEMS, run_depth_suite
from epibench.runners.run_efficiency_benchmark import PLANTED_FACTS, run_efficiency_suite


@pytest.mark.asyncio
async def test_V3_depth_runner_end_to_end_with_fakes_and_cis(tmp_path: Path):
    """V3 + Phase 8 Gate: Depth runner executes end-to-end with oracle/null/recent/unbounded and reports 95% CIs."""
    assert "null_memory" in DEFAULT_DEPTH_SYSTEMS
    assert "oracle" in DEFAULT_DEPTH_SYSTEMS
    assert "recent_context" in DEFAULT_DEPTH_SYSTEMS
    assert "full_context_unbounded" in DEFAULT_DEPTH_SYSTEMS

    res = await run_depth_suite(
        api_key="",
        budget_usd=1.0,
        depths=[6, 24],
        tracks=["R", "D", "X"],
        systems=["null_memory", "oracle", "recent_context", "full_context_unbounded"],
        scenarios_per_track=2,
        seed=20261122,
        output_dir=tmp_path / "depth",
        use_fakes=True,
    )

    assert (tmp_path / "depth" / "results_depth.json").exists()
    matrix = res["comparison_matrix"]
    for sys_name in ("null_memory", "oracle", "recent_context", "full_context_unbounded"):
        assert sys_name in matrix
        for d_str in ("6", "24"):
            assert d_str in matrix[sys_name]["question_accuracy_by_depth"]
            ci = matrix[sys_name]["question_ci_95_by_depth"][d_str]
            assert len(ci) == 2 and ci[0] <= ci[1]

    # Oracle hits 100%, Null hits 0% at both depths
    assert matrix["oracle"]["question_accuracy_by_depth"]["6"] == 100.0
    assert matrix["oracle"]["question_accuracy_by_depth"]["24"] == 100.0
    assert matrix["null_memory"]["question_accuracy_by_depth"]["6"] == 0.0
    assert matrix["null_memory"]["question_accuracy_by_depth"]["24"] == 0.0


@pytest.mark.asyncio
async def test_V3_efficiency_runner_measures_planted_fact_retention(tmp_path: Path):
    """V3 + Phase 8 Gate: Efficiency runner measures planted-fact recall, latency, and tokens across scales."""
    out_path = tmp_path / "eff.json"
    res = await run_efficiency_suite(
        scales=[20, 100],
        systems=["null_memory", "oracle", "recent_context", "vector_rag", "hybrid_rag"],
        seed=20261122,
        output_path=out_path,
        use_fake_embedder=True,
    )

    assert out_path.exists()
    for sc_str in ("20", "100"):
        by_sys = res["by_scale"][sc_str]
        assert by_sys["oracle"]["planted_facts_total"] == len(PLANTED_FACTS)
        assert by_sys["oracle"]["fact_retrieval_recall_pct"] == 100.0
        assert by_sys["null_memory"]["fact_retrieval_recall_pct"] == 0.0
        assert by_sys["hybrid_rag"]["fact_retrieval_recall_pct"] >= 80.0
        assert by_sys["vector_rag"]["avg_retrieval_latency_ms"] >= 0.0
        assert by_sys["recent_context"]["avg_context_tokens"] > 0
