"""Offline Unit Tests for EpiBench.

Tests scenario generation, distractor sampling, programmatic judging,
and statistical metric computations without requiring LLM API keys.
"""

import pytest

from EpiBench.generator.scenario_generator import ScenarioGenerator
from EpiBench.judging.programmatic import (
    check_integer_match,
    check_set_overlap,
    check_stale_answer,
    check_string_match,
)
from EpiBench.metrics.statistics import (
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
    """Verify that all Tier 1 tracks generate valid scenarios and questions."""
    gen = ScenarioGenerator()
    for track in ["R", "D", "X", "F", "I", "P", "K"]:
        sc = gen.generate_scenario(track=track, scenario_idx=0, split="dev", seed=20261122, filler_count=1)
        assert sc.track == track
        assert len(sc.questions) >= 1
        for q in sc.questions:
            assert q.question_text
            assert q.gold_answer is not None


def test_programmatic_string_matcher():
    """Test string matching with normalization."""
    gold = "Kubernetes"
    hyp_pass = "The cluster was deployed on Kubernetes successfully."
    hyp_fail = "The cluster was deployed on Nomad."

    assert check_string_match(hyp_pass, gold) is True
    assert check_string_match(hyp_fail, gold) is False


def test_programmatic_integer_matcher():
    """Test integer matching in text."""
    gold = 3
    hyp_pass = "The launch date slipped 3 times."
    hyp_word_pass = "The launch date slipped three times."
    hyp_fail = "The launch date slipped 5 times."

    assert check_integer_match(hyp_pass, gold) is True
    assert check_integer_match(hyp_word_pass, gold) is True
    assert check_integer_match(hyp_fail, gold) is False


def test_programmatic_set_overlap():
    """Test set precision, recall, and F1 calculation."""
    expected = ["microservice architecture", "marketing campaign"]
    hyp_both = "We need to revisit the microservice architecture and the marketing campaign."
    hyp_one = "Only the microservice architecture is affected."
    hyp_none = "Everything looks fine, no decisions affected."

    res_both = check_set_overlap(hyp_both, expected)
    assert res_both["f1"] == 1.0

    res_one = check_set_overlap(hyp_one, expected)
    assert res_one["recall"] == 0.5

    res_none = check_set_overlap(hyp_none, expected)
    assert res_none["recall"] == 0.0


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
