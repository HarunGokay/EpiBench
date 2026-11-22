"""Offline Unit Tests for epibench: scenario generation, judging, and metrics."""

import pytest

from epibench.generator.scenario_generator import ScenarioGenerator
from epibench.judging.programmatic import (
    check_integer_match,
    check_set_overlap,
    check_stale_answer,
    check_string_match,
)
from epibench.metrics.statistics import (
    exact_mcnemar_test,
    paired_scenario_bootstrap,
    wilson_ci,
)


def test_scenario_generator_determinism():
    """Verify that identical seeds produce identical scenario data."""
    gen1 = ScenarioGenerator()
    gen2 = ScenarioGenerator()

    sc1 = gen1.generate_scenario(track="R", scenario_idx=0, split="dev", seed=20261122, filler_count=2)
    sc2 = gen2.generate_scenario(track="R", scenario_idx=0, split="dev", seed=20261122, filler_count=2)

    assert sc1.scenario_id == sc2.scenario_id
    assert sc1.track == "R"
    assert len(sc1.sessions) == len(sc2.sessions)
    assert len(sc1.questions) == len(sc2.questions)
    for q1, q2 in zip(sc1.questions, sc2.questions):
        assert q1.question_text == q2.question_text
        assert q1.gold_answer == q2.gold_answer


def test_scenario_tracks_generation():
    """Verify that all 9 tracks generate valid scenarios and questions."""
    gen = ScenarioGenerator()
    for track in ["R", "D", "X", "F", "I", "P", "K", "E", "Q"]:
        sc = gen.generate_scenario(track=track, scenario_idx=0, split="dev", seed=20261122, filler_count=1)
        assert sc.track == track
        assert len(sc.questions) >= 1
        for q in sc.questions:
            assert q.question_text
            assert q.gold_answer is not None


def test_programmatic_string_matcher():
    """Test strict structured string matching with normalization."""
    gold = "Kubernetes"
    hyp_pass = "The cluster was deployed on Kubernetes.\nFINAL ANSWER: Kubernetes"
    hyp_fail = "The cluster was deployed on Nomad.\nFINAL ANSWER: Nomad"
    hyp_no_marker = "The cluster was deployed on Kubernetes."

    assert check_string_match(hyp_pass, gold) is True
    assert check_string_match(hyp_fail, gold) is False
    assert check_string_match(hyp_no_marker, gold) is False


def test_programmatic_integer_matcher():
    """Test strict structured integer matching."""
    gold = 3
    hyp_pass = "The launch date slipped 3 times.\nFINAL ANSWER: 3"
    hyp_word_pass = "The launch date slipped three times.\nFINAL ANSWER: three"
    hyp_fail = "The launch date slipped 5 times.\nFINAL ANSWER: 5"

    assert check_integer_match(hyp_pass, gold) is True
    assert check_integer_match(hyp_word_pass, gold) is True
    assert check_integer_match(hyp_fail, gold) is False


def test_programmatic_set_overlap():
    """Test strict semicolon-separated set precision, recall, and F1 calculation."""
    expected = ["microservice architecture", "marketing campaign"]
    hyp_both = "Revisit both.\nFINAL ANSWER: microservice architecture; marketing campaign"
    hyp_one = "Only one affected.\nFINAL ANSWER: microservice architecture"
    hyp_none = "Nothing affected.\nFINAL ANSWER: NONE"

    res_both = check_set_overlap(hyp_both, expected)
    assert res_both["f1"] == 1.0
    assert res_both["is_correct"] is True

    res_one = check_set_overlap(hyp_one, expected)
    assert res_one["recall"] == 0.5
    assert res_one["is_correct"] is False

    res_none = check_set_overlap(hyp_none, expected)
    assert res_none["recall"] == 0.0
    assert res_none["is_correct"] is False


def test_statistical_mcnemar():
    """Test exact McNemar computation on discordant pairs."""
    # When b=5, c=0
    res = exact_mcnemar_test(5, 0)
    assert res["p_value"] == pytest.approx(0.0625, abs=0.001)

    # When symmetric b=5, c=5
    res_sym = exact_mcnemar_test(5, 5)
    assert res_sym["p_value"] == 1.0


def test_wilson_ci():
    """Test Wilson score interval calculation."""
    res = wilson_ci(10, 10, confidence=0.95)
    assert res["lower"] > 0.65
    assert res["upper"] == 1.0


def test_world_simulator_and_environment():
    """Verify procedural world simulation, events, and ground-truth probes."""
    from epibench.environment.env import EpiBenchEnvironment

    env = EpiBenchEnvironment(seed=42)
    obs = env.reset(num_events=10)
    assert obs.event_id
    assert len(obs.turns) >= 1

    steps_taken = 0
    while not env.is_done and steps_taken < 20:
        obs = env.step()
        steps_taken += 1

    assert steps_taken >= 9
    probes = env.generate_probes()
    assert len(probes) >= 4
    for p in probes:
        assert p.dimension in ("factual", "temporal", "dependency", "provenance", "calibration", "privacy")
        assert p.question
        assert p.gold_answer is not None


def test_six_dimensional_scoring():
    """Verify that six independent dimensions compute correctly."""
    from epibench.metrics.eval_metrics import compute_six_dimensional_report

    mock_evals = [
        {"track": "K", "question_subkind": "static_fact", "is_correct": True, "retrieval_latency_ms": 10.0, "context_tokens": 100},
        {"track": "R", "question_subkind": "as_of", "is_correct": True, "retrieval_latency_ms": 12.0, "context_tokens": 120},
        {"track": "D", "question_subkind": "revisit_set", "is_correct": False, "retrieval_latency_ms": 15.0, "context_tokens": 150},
        {"track": "P", "question_subkind": "provenance", "is_correct": True, "retrieval_latency_ms": 8.0, "context_tokens": 90},
        {"track": "Q", "question_subkind": "clarification", "is_correct": True, "retrieval_latency_ms": 9.0, "context_tokens": 80},
        {"track": "E", "question_subkind": "deletion_direct", "is_correct": True, "retrieval_latency_ms": 11.0, "context_tokens": 110},
    ]

    report = compute_six_dimensional_report(mock_evals, "test_system", total_spend_usd=0.05)
    assert report.factual_accuracy == 100.0
    assert report.temporal_consistency == 100.0
    assert report.dependency_reasoning == 0.0
    assert report.provenance == 100.0
    assert report.calibration == 100.0
    assert report.privacy_and_deletion == 100.0
    assert report.composite_score > 70.0
    assert report.avg_retrieval_latency_ms == pytest.approx(10.83, abs=0.1)

