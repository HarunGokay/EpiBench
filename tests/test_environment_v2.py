"""Regression and 20-seed answerability gate tests for Phase 7 environment module (V2)."""

from __future__ import annotations

import pytest

from epibench.adapters.reference_adapters import OracleAdapter
from epibench.environment import EpiBenchEnvironment
from epibench.judging.programmatic import check_forbidden_phrases, normalize


@pytest.mark.parametrize("seed", list(range(100, 120)))
def test_V2_environment_probes_derivable_from_emitted_dialogue_20_seeds(seed: int):
    """V2 + Phase 7 Gate: For 20 distinct seeds, every probe's gold is strictly derivable from emitted dialogue."""
    env = EpiBenchEnvironment(seed=seed)
    obs = env.reset(num_events=15)
    assert obs.event_id
    steps = 0
    while not env.is_done and steps < 50:
        obs = env.step()
        steps += 1
    assert env.is_done

    dialogue = env.emitted_dialogue_text()
    norm_dialogue = normalize(dialogue)
    user_dialogue = "\n".join(
        t["content"] for e in env.events for t in e.dialogue_turns if t.get("role") == "user"
    )
    norm_user_dialogue = normalize(user_dialogue)
    event_types_seen = {e.event_type for e in env.events}

    probes = env.generate_probes()
    assert len(probes) >= 2

    for probe in probes:
        # Never ask about unmentioned tools like Figma
        assert "figma" not in probe.question.lower()

        if probe.scoring_type == "string_match":
            assert normalize(str(probe.gold_answer)) in norm_user_dialogue, (
                f"seed={seed} {probe.probe_id}: gold {probe.gold_answer!r} not in emitted user dialogue"
            )
            if probe.dimension == "temporal":
                stale = probe.metadata.get("stale_value")
                assert stale and normalize(str(stale)) in norm_user_dialogue

        elif probe.scoring_type == "set_overlap":
            assert "dependency_cascade" in event_types_seen
            for label in probe.gold_answer:
                assert normalize(label) in norm_user_dialogue, (
                    f"seed={seed} {probe.probe_id}: label {label!r} not in emitted user dialogue"
                )

        elif probe.dimension == "calibration":
            assert "ambiguous_inquiry" in event_types_seen
            for group in probe.metadata["required_candidates"]:
                assert any(normalize(a) in norm_user_dialogue for a in group)

        elif probe.dimension == "privacy":
            assert "deletion_command" in event_types_seen
            for forb in probe.metadata["forbidden_aliases"]:
                # Secret must have been emitted in dialogue earlier, and must NOT leak inside the question itself
                assert normalize(forb) in norm_dialogue
                assert forb not in probe.question


@pytest.mark.asyncio
async def test_V2_environment_adapter_replay_integration():
    """V2: EpiBenchEnvironment connects cleanly to BaseMemoryAdapter with deletion replay."""
    # Find a seed that triggers deletion_command within 15 events
    env = EpiBenchEnvironment(seed=100)
    env.reset(num_events=20)
    probes = env.generate_probes()

    oracle = OracleAdapter()
    replay_info = await env.replay_into_adapter(oracle, tenant_id="env_test")
    assert replay_info["ingested_sessions"] == len(env.events)

    # Budget fact is retrievable
    budget_probe = next(p for p in probes if p.probe_id == "probe_factual_budget")
    ret = await oracle.retrieve("env_test", budget_probe.question, token_budget=4000)
    assert budget_probe.gold_answer in ret.context

    # If deletion occurred, deleted API key must be absent from Oracle context after replay
    if any(p.dimension == "privacy" for p in probes):
        priv_probe = next(p for p in probes if p.dimension == "privacy")
        ret_p = await oracle.retrieve("env_test", priv_probe.question, token_budget=4000)
        leak = check_forbidden_phrases(ret_p.context, priv_probe.metadata["forbidden_aliases"])
        assert leak["contaminated"] is False
