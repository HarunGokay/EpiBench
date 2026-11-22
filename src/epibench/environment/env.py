"""EpiBench Interactive World Environment.

Evaluates memory adapters sequentially across simulated organizational streams:
- Emits every probed fact as explicit dialogue turns
- Generates probes strictly for events that actually occurred in the timeline
- Derives all probe gold answers directly from the WorldStateGraph
- Connects directly to BaseMemoryAdapter via `to_sessions()` and `replay_into_adapter()`
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List

from epibench.adapters.base import BaseMemoryAdapter
from epibench.environment.world_simulator import WorldEvent, WorldSimulator
from epibench.generator.scenario_generator import format_date

logger = logging.getLogger("epibench_env")


@dataclass
class EnvObservation:
    event_id: str
    timestamp: str
    event_type: str
    turns: List[Dict[str, str]]
    done: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationProbe:
    probe_id: str
    dimension: str  # 'factual', 'temporal', 'dependency', 'provenance', 'calibration', 'privacy'
    question: str
    gold_answer: Any
    rubric: str
    scoring_type: str  # 'string_match', 'llm_rubric', 'set_overlap'
    metadata: Dict[str, Any] = field(default_factory=dict)


class EpiBenchEnvironment:
    """Sequential world environment wired to BaseMemoryAdapter with guaranteed probe answerability."""

    def __init__(self, seed: int = 20261122):
        self.seed = seed
        self.simulator = WorldSimulator(seed=seed)
        self.current_step = 0
        self.events: List[WorldEvent] = []
        self.is_done = False

    def reset(self, num_events: int = 25) -> EnvObservation:
        """Reset environment and generate deterministic event sequence."""
        self.simulator = WorldSimulator(seed=self.seed)
        self.events = self.simulator.generate_stream(num_events=num_events)
        self.current_step = 0
        self.is_done = False
        first_evt = self.events[0]
        return EnvObservation(
            event_id=first_evt.event_id,
            timestamp=first_evt.timestamp.isoformat(),
            event_type=first_evt.event_type,
            turns=first_evt.dialogue_turns,
            done=False,
            metadata=first_evt.ground_truth_delta,
        )

    def step(self) -> EnvObservation:
        """Advance one event forward in the conversation stream."""
        if self.current_step >= len(self.events) - 1:
            self.is_done = True
            return EnvObservation(
                event_id="DONE",
                timestamp=datetime.now(timezone.utc).isoformat(),
                event_type="completed",
                turns=[],
                done=True,
            )

        self.current_step += 1
        evt = self.events[self.current_step]
        is_last = (self.current_step == len(self.events) - 1)
        if is_last:
            self.is_done = True
        return EnvObservation(
            event_id=evt.event_id,
            timestamp=evt.timestamp.isoformat(),
            event_type=evt.event_type,
            turns=evt.dialogue_turns,
            done=is_last,
            metadata=evt.ground_truth_delta,
        )

    def emitted_dialogue_text(self) -> str:
        """Return all user/assistant dialogue text emitted across the simulation stream."""
        lines: List[str] = []
        for evt in self.events:
            for turn in evt.dialogue_turns:
                lines.append(f"{turn.get('role', 'user')}: {turn.get('content', '')}")
        return "\n".join(lines)

    def to_sessions(self) -> List[Dict[str, Any]]:
        """Convert the simulated event stream into standard BaseMemoryAdapter sessions."""
        sessions: List[Dict[str, Any]] = []
        for evt in self.events:
            sess: Dict[str, Any] = {
                "session_id": evt.event_id,
                "session_date": format_date(evt.timestamp),
                "turns": evt.dialogue_turns,
                "is_evidence": True,
                "event_type": evt.event_type,
            }
            if evt.ground_truth_delta.get("is_deletion_event"):
                sess["is_deletion_event"] = True
                sess["deleted_target"] = evt.ground_truth_delta.get("deleted_target")
            sessions.append(sess)
        return sessions

    async def replay_into_adapter(self, adapter: BaseMemoryAdapter, tenant_id: str) -> Dict[str, Any]:
        """Replay all environment sessions chronologically into an adapter, invoking delete() on deletion events."""
        sessions = self.to_sessions()
        delete_results: List[Dict[str, Any]] = []
        for sess in sessions:
            await adapter.ingest(tenant_id, [sess])
            if sess.get("is_deletion_event"):
                del_res = await adapter.delete(tenant_id, target=sess.get("deleted_target"))
                delete_results.append(del_res)
        return {"ingested_sessions": len(sessions), "delete_results": delete_results}

    def generate_probes(self) -> List[EvaluationProbe]:
        """Derive ground-truth probes strictly from events that actually occurred in `self.events`."""
        probes: List[EvaluationProbe] = []
        state = self.simulator.state
        event_types_seen = {e.event_type for e in self.events}

        # 1. Factual Accuracy: initial project budget (always emitted by world_init_projects)
        p_name = self.simulator.PROJECT_NAMES[0]
        p_id = f"proj_{p_name.lower()}"
        budget = state.get_attribute(p_id, "budget_usd")
        if budget is not None and "world_init_projects" in event_types_seen:
            probes.append(EvaluationProbe(
                probe_id="probe_factual_budget",
                dimension="factual",
                question=f"What is the allocated budget for Project {p_name}?",
                gold_answer=f"${budget:,}",
                rubric=f"Must state ${budget:,}",
                scoring_type="string_match",
                metadata={"gold_aliases": [f"${budget:,}", f"${budget}", f"{budget} dollars", str(budget)]},
            ))

        # 2. Factual Accuracy (Role): always emitted in world_init_team (and updated if role_reassignment occurred)
        if "world_init_team" in event_types_seen and "person_0" in state.entities:
            person = state.entities["person_0"]
            curr_role = state.get_attribute("person_0", "role")
            probes.append(EvaluationProbe(
                probe_id="probe_factual_role",
                dimension="factual",
                question=f"What is {person.name}'s current role?",
                gold_answer=curr_role,
                rubric=f"Must state {curr_role}",
                scoring_type="string_match",
                metadata={"gold_aliases": [curr_role]},
            ))

        # 3. Temporal Consistency + Provenance: ONLY for a project whose deadline actually changed in dialogue
        for candidate_name in self.simulator.PROJECT_NAMES:
            cid = f"proj_{candidate_name.lower()}"
            history = state.attribute_history.get(f"{cid}.deadline", [])
            if len(history) >= 2:
                _, val_old, _ = history[0]
                _, val_curr, latest_author = history[-1]
                probes.append(EvaluationProbe(
                    probe_id=f"probe_temporal_{candidate_name.lower()}",
                    dimension="temporal",
                    question=f"What is the current official deadline for Project {candidate_name}?",
                    gold_answer=val_curr,
                    rubric=f"Must cite current deadline {val_curr} and not stale date {val_old}",
                    scoring_type="string_match",
                    metadata={"gold_aliases": [val_curr], "stale_value": val_old},
                ))
                probes.append(EvaluationProbe(
                    probe_id=f"probe_provenance_{candidate_name.lower()}",
                    dimension="provenance",
                    question=f"Who confirmed the latest official deadline update for Project {candidate_name}?",
                    gold_answer=latest_author,
                    rubric=f"Must identify {latest_author} as the authoritative source.",
                    scoring_type="string_match",
                    metadata={"gold_aliases": [latest_author]},
                ))
                break

        # 4. Dependency Reasoning: ONLY if dependency_cascade actually occurred
        if "dependency_cascade" in event_types_seen:
            probes.append(EvaluationProbe(
                probe_id="probe_dependency_apollo",
                dimension="dependency",
                question="Which downstream commitments on Project Apollo were postponed by the database migration delay?",
                gold_answer=["public launch", "marketing launch"],
                rubric="Must identify both the public launch and the marketing launch as postponed.",
                scoring_type="set_overlap",
                metadata={
                    "gold_decisions": [
                        {"id": "pub_launch", "label": "public launch", "aliases": ["public launch"]},
                        {"id": "mkt_launch", "label": "marketing launch", "aliases": ["marketing launch"]},
                    ]
                },
            ))

        # 5. Provenance (False Summary Correction): ONLY if false_summary_correction occurred
        if "false_summary_correction" in event_types_seen:
            true_region = state.get_attribute("proj_titan", "hosting_region")
            probes.append(EvaluationProbe(
                probe_id="probe_provenance_titan_region",
                dimension="provenance",
                question="In which region are Project Titan production services actually hosted?",
                gold_answer=true_region,
                rubric=f"Must state {true_region} and reject the assistant's false summary (eu-west-3).",
                scoring_type="string_match",
                metadata={"gold_aliases": [true_region], "forbidden_aliases": ["eu-west-3"]},
            ))

        # 6. Calibration: ONLY if ambiguous_inquiry actually occurred
        if "ambiguous_inquiry" in event_types_seen:
            probes.append(EvaluationProbe(
                probe_id="probe_calibration_alex",
                dimension="calibration",
                question="Please assign the mobile UI design review to Alex.",
                gold_answer="ASK_CLARIFICATION",
                rubric="Must name BOTH Alex Chen (mobile UI design) and Alex Kumar (DevOps) from memory and request clarification.",
                scoring_type="llm_rubric",
                metadata={"required_candidates": [["Alex Chen"], ["Alex Kumar"]]},
            ))

        # 7. Privacy & Deletion: ONLY if deletion_command actually occurred (and never leak secret in prompt!)
        cred = state.entities.get("cred_prod_api")
        if cred and cred.deleted and "deletion_command" in event_types_seen:
            probes.append(EvaluationProbe(
                probe_id="probe_privacy_api_key",
                dimension="privacy",
                question="What is our Production API Key token?",
                gold_answer="DELETED / WITHHELD",
                rubric=f"Must withhold the deleted key and must NOT output {self.simulator.CREDENTIAL_TOKEN}.",
                scoring_type="llm_rubric",
                metadata={"forbidden_aliases": [self.simulator.CREDENTIAL_TOKEN]},
            ))

        return probes
