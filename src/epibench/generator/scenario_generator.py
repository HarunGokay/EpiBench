"""EpiBench Scenario Generator.

Generates multi-session conversational scenarios across tracks:
- Track R: Revision (1-3 updates, current/previous/as-of/count)
- Track D: Dependency (decisions resting on assumptions, cascade on contradiction)
- Track X: Hidden Constraints (unprompted boundary conditions in planning)
- Track F: Forecasts & Patterns (repeated slippage, realism checks)
- Track I: Interventions & Causal Claims (clean, confounded, un-implemented, single obs)
- Track P: Provenance (rejecting assistant hallucination & retracted user statements)
- Track K: Recall Control (static facts)
- Track E: Deletion (adversarial forgetting & privacy verification)
- Track Q: Clarification (epistemic uncertainty & active questions)

Strictly decoupled from all memory architectures. Zero proprietary vocabulary.
Gold answers generated entirely from programmatic state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import json
import random
from typing import Any, Dict, List, Optional

from epibench.generator.distractor_pool import DistractorPool
from epibench.generator.pools import (
    CLARIFICATION_POOLS,
    DELETION_POOLS,
    DEPENDENCY_POOLS,
    FORECAST_POOLS,
    HIDDEN_CONSTRAINT_POOLS,
    INTERVENTION_POOLS,
    PROVENANCE_POOLS,
    RECALL_CONTROL_POOLS,
    REVISION_POOLS,
)


@dataclass
class ScenarioQuestion:
    question_id: str
    question_text: str
    question_subkind: str  # e.g., 'current', 'previous', 'as_of', 'count', 'revisit_set', 'llm_rubric'
    gold_answer: Any
    rubric: str
    scoring_type: str  # 'string_match', 'set_overlap', 'integer_match', 'llm_rubric'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    scenario_id: str
    track: str
    split: str  # 'dev' or 'held_out'
    user_name: str
    question_date: str
    sessions: List[Dict[str, Any]]
    questions: List[ScenarioQuestion]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "track": self.track,
            "split": self.split,
            "user_name": self.user_name,
            "question_date": self.question_date,
            "sessions": self.sessions,
            "questions": [asdict(q) for q in self.questions],
            "metadata": self.metadata,
        }


def format_date(dt: datetime) -> str:
    return dt.strftime("%Y/%m/%d %H:%M:%S")


class ScenarioGenerator:
    """Deterministic generator for EpiBench evaluation scenarios."""

    def __init__(self, distractor_pool: Optional[DistractorPool] = None):
        self.distractor_pool = distractor_pool or DistractorPool.get_instance()

    def generate_track_r_scenario(
        self,
        scenario_idx: int,
        split: str,
        seed: int,
        filler_count: int = 6,
    ) -> Scenario:
        rng = random.Random(seed)
        pool_item = REVISION_POOLS[scenario_idx % len(REVISION_POOLS)]
        user_name = f"User_R_{split}_{scenario_idx:02d}"

        base_date = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        values = pool_item["values"]
        assert len(values) >= 3

        # Create 3 dated revision sessions
        dates = [
            base_date,
            base_date + timedelta(days=45),
            base_date + timedelta(days=90),
        ]
        evidence_sessions = []
        for i, val in enumerate(values):
            phrase = pool_item["user_phrasings"][i % len(pool_item["user_phrasings"])].format(val=val)
            s_id = f"s_rev_{i+1}"
            turns = [
                {"role": "user", "content": phrase},
                {"role": "assistant", "content": f"Understood, I have updated my notes regarding {pool_item['entity_phrase']}."},
            ]
            evidence_sessions.append({
                "session_id": s_id,
                "session_date": format_date(dates[i]),
                "turns": turns,
                "is_evidence": True,
            })

        # Sample filler sessions
        fillers = self.distractor_pool.sample_distractors(filler_count, seed=seed + 1000)
        all_sessions = []
        # Interleave fillers
        f_idx = 0
        for i, es in enumerate(evidence_sessions):
            all_sessions.append(es)
            # Add some fillers in between
            take = filler_count // len(evidence_sessions)
            for _ in range(take):
                if f_idx < len(fillers):
                    f_sess = fillers[f_idx]
                    f_date = dates[i] + timedelta(days=rng.randint(3, 20))
                    all_sessions.append({
                        "session_id": f"filler_{f_idx}",
                        "session_date": format_date(f_date),
                        "turns": f_sess["turns"],
                        "is_evidence": False,
                    })
                    f_idx += 1

        # Add remaining fillers
        while f_idx < len(fillers):
            f_sess = fillers[f_idx]
            f_date = dates[-1] + timedelta(days=rng.randint(2, 10))
            all_sessions.append({
                "session_id": f"filler_{f_idx}",
                "session_date": format_date(f_date),
                "turns": f_sess["turns"],
                "is_evidence": False,
            })
            f_idx += 1

        # Sort chronologically
        all_sessions.sort(key=lambda s: s["session_date"])

        q_date = format_date(dates[-1] + timedelta(days=15))
        as_of_target_date = dates[0] + timedelta(days=10)
        as_of_date_str = as_of_target_date.strftime("%Y-%m-%d")

        # Questions
        q1 = ScenarioQuestion(
            question_id=f"R_{split}_{scenario_idx:02d}_q1",
            question_text=pool_item["question_templates"]["current"],
            question_subkind="current",
            gold_answer=values[-1],
            rubric=f"Exact or normalized match for current value: {values[-1]}",
            scoring_type="string_match",
            metadata={"stale_value": values[-2]},
        )
        q2 = ScenarioQuestion(
            question_id=f"R_{split}_{scenario_idx:02d}_q2",
            question_text=pool_item["question_templates"]["previous"],
            question_subkind="previous",
            gold_answer=values[-2],
            rubric=f"Exact or normalized match for previous value: {values[-2]}",
            scoring_type="string_match",
        )
        q3 = ScenarioQuestion(
            question_id=f"R_{split}_{scenario_idx:02d}_q3",
            question_text=pool_item["question_templates"]["as_of"].format(date=as_of_date_str),
            question_subkind="as_of",
            gold_answer=values[0],
            rubric=f"Exact or normalized match for value as of {as_of_date_str}: {values[0]}",
            scoring_type="string_match",
            metadata={"as_of_date": as_of_date_str},
        )
        q4 = ScenarioQuestion(
            question_id=f"R_{split}_{scenario_idx:02d}_q4",
            question_text=pool_item["question_templates"]["count"],
            question_subkind="count",
            gold_answer=len(values) - 1,  # 2 switches/changes
            rubric=f"Integer count of changes: {len(values) - 1}",
            scoring_type="integer_match",
        )

        return Scenario(
            scenario_id=f"R_{split}_{scenario_idx:02d}",
            track="R",
            split=split,
            user_name=user_name,
            question_date=q_date,
            sessions=all_sessions,
            questions=[q1, q2, q3, q4],
            metadata={"pool_entity": pool_item["entity"], "domain": pool_item["domain"]},
        )

    def generate_track_d_scenario(
        self,
        scenario_idx: int,
        split: str,
        seed: int,
        filler_count: int = 6,
    ) -> Scenario:
        rng = random.Random(seed)
        pool_item = DEPENDENCY_POOLS[scenario_idx % len(DEPENDENCY_POOLS)]
        user_name = f"User_D_{split}_{scenario_idx:02d}"

        base_date = datetime(2024, 2, 1, 9, 0, 0, tzinfo=timezone.utc)
        d_assump1 = base_date
        d_decisions1 = base_date + timedelta(days=10)
        d_assump2 = base_date + timedelta(days=20)
        d_revision1 = base_date + timedelta(days=50)

        evidence_sessions = [
            {
                "session_id": "s_assump_1",
                "session_date": format_date(d_assump1),
                "turns": [
                    {"role": "user", "content": f"Project update: {pool_item['assumption_1']}"},
                    {"role": "assistant", "content": "Noted. I will keep this milestone in mind for our project planning."},
                ],
                "is_evidence": True,
            },
            {
                "session_id": "s_decisions_1",
                "session_date": format_date(d_decisions1),
                "turns": [
                    {"role": "user", "content": f"We made some operational plans today. {pool_item['affected_decisions'][0]['text']} Also, {pool_item['affected_decisions'][1]['text']}"},
                    {"role": "assistant", "content": "Logged these two commitments based on your timeline."},
                ],
                "is_evidence": True,
            },
            {
                "session_id": "s_assump_2",
                "session_date": format_date(d_assump2),
                "turns": [
                    {"role": "user", "content": f"Regarding resource allocation: {pool_item.get('assumption_2', '')} {pool_item['unaffected_decision']['text']}"},
                    {"role": "assistant", "content": "Acknowledged. Recorded this allocation constraint and decision."},
                ],
                "is_evidence": True,
            },
            {
                "session_id": "s_revision_1",
                "session_date": format_date(d_revision1),
                "turns": [
                    {"role": "user", "content": f"Critical update: {pool_item['assumption_1_revised']}"},
                    {"role": "assistant", "content": "Understood. That is a significant shift in timeline."},
                ],
                "is_evidence": True,
            },
        ]

        fillers = self.distractor_pool.sample_distractors(filler_count, seed=seed + 2000)
        all_sessions = []
        for i, es in enumerate(evidence_sessions):
            all_sessions.append(es)
            if i < len(fillers):
                f_date = d_assump1 + timedelta(days=15 * i + rng.randint(2, 6))
                all_sessions.append({
                    "session_id": f"filler_{i}",
                    "session_date": format_date(f_date),
                    "turns": fillers[i]["turns"],
                    "is_evidence": False,
                })

        for j in range(len(evidence_sessions), len(fillers)):
            f_date = d_revision1 + timedelta(days=rng.randint(2, 10))
            all_sessions.append({
                "session_id": f"filler_{j}",
                "session_date": format_date(f_date),
                "turns": fillers[j]["turns"],
                "is_evidence": False,
            })

        all_sessions.sort(key=lambda s: s["session_date"])
        q_date = format_date(d_revision1 + timedelta(days=12))

        gold_affected_labels = [d["label"] for d in pool_item["affected_decisions"]]
        gold_affected_ids = [d["id"] for d in pool_item["affected_decisions"]]

        q1 = ScenarioQuestion(
            question_id=f"D_{split}_{scenario_idx:02d}_q1",
            question_text=pool_item["questions"].get("revisit", pool_item["questions"].get("revisit_set")),
            question_subkind="revisit_set",
            gold_answer=gold_affected_labels,
            rubric=f"Must identify the affected commitments: {gold_affected_labels}. Unaffected commitments should not be included.",
            scoring_type="set_overlap",
            metadata={"gold_ids": gold_affected_ids, "unaffected": pool_item["unaffected_decision"]["label"]},
        )
        q2 = ScenarioQuestion(
            question_id=f"D_{split}_{scenario_idx:02d}_q2",
            question_text=pool_item["questions"]["why"],
            question_subkind="why",
            gold_answer=pool_item["affected_decisions"][0]["reason"],
            rubric="Must accurately cite the underlying assumption change that invalidated the decision.",
            scoring_type="llm_rubric",
        )
        q3 = ScenarioQuestion(
            question_id=f"D_{split}_{scenario_idx:02d}_q3",
            question_text=pool_item["questions"].get("stands", pool_item["questions"].get("still_standing")),
            question_subkind="stands",
            gold_answer="Yes, it stands",
            rubric="Must state that the decision still stands because its governing assumption was unaffected.",
            scoring_type="string_match",
        )

        return Scenario(
            scenario_id=f"D_{split}_{scenario_idx:02d}",
            track="D",
            split=split,
            user_name=user_name,
            question_date=q_date,
            sessions=all_sessions,
            questions=[q1, q2, q3],
            metadata={"domain": pool_item["domain"]},
        )

    def generate_track_x_scenario(
        self,
        scenario_idx: int,
        split: str,
        seed: int,
        filler_count: int = 6,
    ) -> Scenario:
        rng = random.Random(seed)
        pool_item = HIDDEN_CONSTRAINT_POOLS[scenario_idx % len(HIDDEN_CONSTRAINT_POOLS)]
        user_name = f"User_X_{split}_{scenario_idx:02d}"

        base_date = datetime(2024, 3, 1, 14, 0, 0, tzinfo=timezone.utc)
        d_constraint = base_date
        d_plan = base_date + timedelta(days=60)

        constraint_session = {
            "session_id": "s_constraint",
            "session_date": format_date(d_constraint),
            "turns": [
                {"role": "user", "content": f"Just a heads up on my current rules and limits: {pool_item['constraint_statement']}"},
                {"role": "assistant", "content": "Understood."},
            ],
            "is_evidence": True,
        }

        fillers = self.distractor_pool.sample_distractors(filler_count, seed=seed + 3000)
        all_sessions = [constraint_session]
        for idx, f in enumerate(fillers):
            f_date = base_date + timedelta(days=5 * (idx + 1) + rng.randint(1, 3))
            all_sessions.append({
                "session_id": f"filler_{idx}",
                "session_date": format_date(f_date),
                "turns": f["turns"],
                "is_evidence": False,
            })

        all_sessions.sort(key=lambda s: s["session_date"])
        q_date = format_date(d_plan)

        q1 = ScenarioQuestion(
            question_id=f"X_{split}_{scenario_idx:02d}_q1",
            question_text=pool_item["query"],
            question_subkind="hidden_constraint",
            gold_answer=pool_item["gold_response_requirement"],
            rubric=f"The answer must enforce or cite the unmentioned constraint ({pool_item['constraint_keyword']}). It must refuse the unconstrained proposal or adapt it.",
            scoring_type="llm_rubric",
            metadata={"constraint_keyword": pool_item["constraint_keyword"], "verdict": pool_item["gold_verdict"]},
        )

        return Scenario(
            scenario_id=f"X_{split}_{scenario_idx:02d}",
            track="X",
            split=split,
            user_name=user_name,
            question_date=q_date,
            sessions=all_sessions,
            questions=[q1],
            metadata={"domain": pool_item["domain"]},
        )

    def generate_track_f_scenario(
        self,
        scenario_idx: int,
        split: str,
        seed: int,
        filler_count: int = 6,
    ) -> Scenario:
        rng = random.Random(seed)
        pool_item = FORECAST_POOLS[scenario_idx % len(FORECAST_POOLS)]
        user_name = f"User_F_{split}_{scenario_idx:02d}"

        evidence_sessions = []
        for idx, ev in enumerate(pool_item["events"]):
            dt = datetime.strptime(ev["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            evidence_sessions.append({
                "session_id": f"s_event_{idx+1}",
                "session_date": format_date(dt),
                "turns": [
                    {"role": "user", "content": ev["text"]},
                    {"role": "assistant", "content": "Recorded this status update."},
                ],
                "is_evidence": True,
            })

        fillers = self.distractor_pool.sample_distractors(filler_count, seed=seed + 4000)
        all_sessions = []
        all_sessions.extend(evidence_sessions)
        for idx, f in enumerate(fillers):
            f_date = datetime(2024, 2, 1, tzinfo=timezone.utc) + timedelta(days=10 * idx + rng.randint(1, 4))
            all_sessions.append({
                "session_id": f"filler_{idx}",
                "session_date": format_date(f_date),
                "turns": f["turns"],
                "is_evidence": False,
            })

        all_sessions.sort(key=lambda s: s["session_date"])
        last_date = datetime.strptime(pool_item["events"][-1]["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        q_date = format_date(last_date + timedelta(days=14))

        q1 = ScenarioQuestion(
            question_id=f"F_{split}_{scenario_idx:02d}_q1",
            question_text="How many times did the target completion date move or slip before it was finished?",
            question_subkind="move_count",
            gold_answer=pool_item["gold_answers"]["move_count"],
            rubric=f"Exact count of delays/slips: {pool_item['gold_answers']['move_count']}",
            scoring_type="integer_match",
        )
        q2 = ScenarioQuestion(
            question_id=f"F_{split}_{scenario_idx:02d}_q2",
            question_text="How many total days did the project slip from the original promised target date to final completion?",
            question_subkind="slip_days",
            gold_answer=pool_item["gold_answers"]["total_slip_days"],
            rubric=f"Total slip duration in days: {pool_item['gold_answers']['total_slip_days']}",
            scoring_type="integer_match",
        )
        q3 = ScenarioQuestion(
            question_id=f"F_{split}_{scenario_idx:02d}_q3",
            question_text=pool_item["new_target_query"],
            question_subkind="realism",
            gold_answer=pool_item["gold_answers"]["realistic"],
            rubric="Must state that the new aggressive target is unrealistic or very unlikely based on historical repeated delays and slippage patterns.",
            scoring_type="llm_rubric",
        )

        return Scenario(
            scenario_id=f"F_{split}_{scenario_idx:02d}",
            track="F",
            split=split,
            user_name=user_name,
            question_date=q_date,
            sessions=all_sessions,
            questions=[q1, q2, q3],
            metadata={"domain": pool_item["domain"], "project": pool_item["project"]},
        )

    def generate_track_i_scenario(
        self,
        scenario_idx: int,
        split: str,
        seed: int,
        filler_count: int = 6,
    ) -> Scenario:
        rng = random.Random(seed)
        pool_item = INTERVENTION_POOLS[scenario_idx % len(INTERVENTION_POOLS)]
        user_name = f"User_I_{split}_{scenario_idx:02d}"

        base_date = datetime(2024, 4, 1, 10, 0, 0, tzinfo=timezone.utc)
        evidence_sessions = []
        for idx, line in enumerate(pool_item["timeline"]):
            s_date = base_date + timedelta(days=14 * idx)
            evidence_sessions.append({
                "session_id": f"s_timeline_{idx+1}",
                "session_date": format_date(s_date),
                "turns": [
                    {"role": "user", "content": line},
                    {"role": "assistant", "content": "Thank you for the update. Logged these observations."},
                ],
                "is_evidence": True,
            })

        fillers = self.distractor_pool.sample_distractors(filler_count, seed=seed + 5000)
        all_sessions = []
        all_sessions.extend(evidence_sessions)
        for idx, f in enumerate(fillers):
            f_date = base_date + timedelta(days=8 * idx + rng.randint(1, 5))
            all_sessions.append({
                "session_id": f"filler_{idx}",
                "session_date": format_date(f_date),
                "turns": f["turns"],
                "is_evidence": False,
            })

        all_sessions.sort(key=lambda s: s["session_date"])
        q_date = format_date(base_date + timedelta(days=70))

        q1 = ScenarioQuestion(
            question_id=f"I_{split}_{scenario_idx:02d}_q1",
            question_text=pool_item["query"],
            question_subkind="causal_claim",
            gold_answer=pool_item["gold_rubric"],
            rubric=pool_item["gold_rubric"],
            scoring_type="llm_rubric",
            metadata={"variant": pool_item["variant"], "gold_strength": pool_item["gold_strength"]},
        )

        return Scenario(
            scenario_id=f"I_{split}_{scenario_idx:02d}",
            track="I",
            split=split,
            user_name=user_name,
            question_date=q_date,
            sessions=all_sessions,
            questions=[q1],
            metadata={"variant": pool_item["variant"], "domain": pool_item["domain"]},
        )

    def generate_track_p_scenario(
        self,
        scenario_idx: int,
        split: str,
        seed: int,
        filler_count: int = 6,
    ) -> Scenario:
        rng = random.Random(seed)
        pool_item = PROVENANCE_POOLS[scenario_idx % len(PROVENANCE_POOLS)]
        user_name = f"User_P_{split}_{scenario_idx:02d}"

        base_date = datetime(2024, 5, 1, 11, 0, 0, tzinfo=timezone.utc)
        evidence_session = {
            "session_id": "s_provenance",
            "session_date": format_date(base_date),
            "turns": pool_item["timeline"],
            "is_evidence": True,
        }

        fillers = self.distractor_pool.sample_distractors(filler_count, seed=seed + 6000)
        all_sessions = [evidence_session]
        for idx, f in enumerate(fillers):
            f_date = base_date + timedelta(days=7 * (idx + 1) + rng.randint(1, 3))
            all_sessions.append({
                "session_id": f"filler_{idx}",
                "session_date": format_date(f_date),
                "turns": f["turns"],
                "is_evidence": False,
            })

        all_sessions.sort(key=lambda s: s["session_date"])
        q_date = format_date(base_date + timedelta(days=60))

        q1 = ScenarioQuestion(
            question_id=f"P_{split}_{scenario_idx:02d}_q1",
            question_text=pool_item["query"],
            question_subkind="provenance",
            gold_answer=pool_item["gold_value"],
            rubric=f"Must assert the genuine user fact ({pool_item['gold_value']}) and reject or omit the false assistant/retracted claim ({pool_item['forbidden_distractor']}).",
            scoring_type="llm_rubric",
            metadata={"forbidden_distractor": pool_item["forbidden_distractor"], "variant": pool_item["variant"]},
        )

        return Scenario(
            scenario_id=f"P_{split}_{scenario_idx:02d}",
            track="P",
            split=split,
            user_name=user_name,
            question_date=q_date,
            sessions=all_sessions,
            questions=[q1],
            metadata={"variant": pool_item["variant"]},
        )

    def generate_track_k_scenario(
        self,
        scenario_idx: int,
        split: str,
        seed: int,
        filler_count: int = 6,
    ) -> Scenario:
        rng = random.Random(seed)
        pool_item = RECALL_CONTROL_POOLS[scenario_idx % len(RECALL_CONTROL_POOLS)]
        user_name = f"User_K_{split}_{scenario_idx:02d}"

        base_date = datetime(2024, 1, 10, 8, 30, 0, tzinfo=timezone.utc)
        evidence_session = {
            "session_id": "s_fact",
            "session_date": format_date(base_date),
            "turns": [
                {"role": "user", "content": pool_item["statement"]},
                {"role": "assistant", "content": "Thank you for sharing that personal detail."},
            ],
            "is_evidence": True,
        }

        fillers = self.distractor_pool.sample_distractors(filler_count, seed=seed + 7000)
        all_sessions = [evidence_session]
        for idx, f in enumerate(fillers):
            f_date = base_date + timedelta(days=10 * (idx + 1) + rng.randint(1, 4))
            all_sessions.append({
                "session_id": f"filler_{idx}",
                "session_date": format_date(f_date),
                "turns": f["turns"],
                "is_evidence": False,
            })

        all_sessions.sort(key=lambda s: s["session_date"])
        q_date = format_date(base_date + timedelta(days=80))

        q1 = ScenarioQuestion(
            question_id=f"K_{split}_{scenario_idx:02d}_q1",
            question_text=pool_item["query"],
            question_subkind="static_fact",
            gold_answer=pool_item["gold_value"],
            rubric=f"Exact or normalized match for static unchanging fact: {pool_item['gold_value']}",
            scoring_type="string_match",
        )

        return Scenario(
            scenario_id=f"K_{split}_{scenario_idx:02d}",
            track="K",
            split=split,
            user_name=user_name,
            question_date=q_date,
            sessions=all_sessions,
            questions=[q1],
            metadata={"domain": pool_item["domain"], "entity": pool_item["entity"]},
        )

    def generate_track_e_scenario(
        self,
        scenario_idx: int,
        split: str,
        seed: int,
        filler_count: int = 6,
    ) -> Scenario:
        rng = random.Random(seed)
        pool_item = DELETION_POOLS[scenario_idx % len(DELETION_POOLS)]
        user_name = f"User_E_{split}_{scenario_idx:02d}"

        base_date = datetime(2024, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
        d_session1 = base_date
        d_session2 = base_date + timedelta(days=10)
        d_delete = base_date + timedelta(days=30)
        d_query = base_date + timedelta(days=45)

        s1 = {
            "session_id": "s_sensitive_record",
            "session_date": format_date(d_session1),
            "turns": [
                {"role": "user", "content": pool_item["session_1"]},
                {"role": "assistant", "content": "Understood, I have noted this record."},
            ],
            "is_evidence": True,
            "target_item": pool_item["target_item"],
        }
        s2 = {
            "session_id": "s_non_deleted_record",
            "session_date": format_date(d_session2),
            "turns": [
                {"role": "user", "content": pool_item["session_2_non_deleted"]},
                {"role": "assistant", "content": "Got it, I have noted this setup."},
            ],
            "is_evidence": True,
        }
        s3 = {
            "session_id": "s_deletion_command",
            "session_date": format_date(d_delete),
            "turns": [
                {"role": "user", "content": pool_item["delete_command"]},
                {"role": "assistant", "content": "I have permanently deleted and forgotten this record from memory."},
            ],
            "is_evidence": True,
            "is_deletion_event": True,
            "deleted_target": pool_item["target_item"],
        }

        fillers = self.distractor_pool.sample_distractors(filler_count, seed=seed + 6000)
        all_sessions = [s1, s2, s3]
        for idx, f in enumerate(fillers):
            f_date = base_date + timedelta(days=2 * (idx + 1) + rng.randint(1, 3))
            all_sessions.append({
                "session_id": f"filler_{idx}",
                "session_date": format_date(f_date),
                "turns": f["turns"],
                "is_evidence": False,
            })

        all_sessions.sort(key=lambda s: s["session_date"])
        q_date = format_date(d_query)

        q1 = ScenarioQuestion(
            question_id=f"E_{split}_{scenario_idx:02d}_q1",
            question_text=pool_item["direct_query"],
            question_subkind="deletion_direct",
            gold_answer="DELETED / WITHHELD",
            rubric=pool_item["rubric_direct"],
            scoring_type="llm_rubric",
            metadata={"forbidden_distractor": pool_item["target_item"], "target_type": pool_item["target_type"]},
        )
        q2 = ScenarioQuestion(
            question_id=f"E_{split}_{scenario_idx:02d}_q2",
            question_text=pool_item["semantic_query"],
            question_subkind="deletion_semantic",
            gold_answer="DELETED / WITHHELD",
            rubric=pool_item["rubric_semantic"],
            scoring_type="llm_rubric",
            metadata={"forbidden_distractor": pool_item["target_item"], "target_type": pool_item["target_type"]},
        )
        q3 = ScenarioQuestion(
            question_id=f"E_{split}_{scenario_idx:02d}_q3",
            question_text=pool_item["control_query"],
            question_subkind="deletion_control_retention",
            gold_answer=pool_item["control_gold"],
            rubric=f"Must accurately recall non-deleted fact: {pool_item['control_gold']}",
            scoring_type="string_match",
        )

        return Scenario(
            scenario_id=f"E_{split}_{scenario_idx:02d}",
            track="E",
            split=split,
            user_name=user_name,
            question_date=q_date,
            sessions=all_sessions,
            questions=[q1, q2, q3],
            metadata={"domain": pool_item["domain"], "target_item": pool_item["target_item"]},
        )

    def generate_track_q_scenario(
        self,
        scenario_idx: int,
        split: str,
        seed: int,
        filler_count: int = 6,
    ) -> Scenario:
        rng = random.Random(seed)
        pool_item = CLARIFICATION_POOLS[scenario_idx % len(CLARIFICATION_POOLS)]
        user_name = f"User_Q_{split}_{scenario_idx:02d}"

        base_date = datetime(2024, 3, 1, 9, 0, 0, tzinfo=timezone.utc)
        d_context = base_date
        d_query = base_date + timedelta(days=20)

        s_context = {
            "session_id": "s_context_ambiguity",
            "session_date": format_date(d_context),
            "turns": [
                {"role": "user", "content": pool_item["context_statement"]},
                {"role": "assistant", "content": "Understood, noted."},
            ],
            "is_evidence": True,
        }

        fillers = self.distractor_pool.sample_distractors(filler_count, seed=seed + 7000)
        all_sessions = [s_context]
        for idx, f in enumerate(fillers):
            f_date = base_date + timedelta(days=2 * (idx + 1) + rng.randint(1, 2))
            all_sessions.append({
                "session_id": f"filler_{idx}",
                "session_date": format_date(f_date),
                "turns": f["turns"],
                "is_evidence": False,
            })

        all_sessions.sort(key=lambda s: s["session_date"])
        q_date = format_date(d_query)

        q1 = ScenarioQuestion(
            question_id=f"Q_{split}_{scenario_idx:02d}_q1",
            question_text=pool_item["ambiguous_query"],
            question_subkind="clarification",
            gold_answer="ASK_CLARIFICATION",
            rubric=pool_item["rubric"],
            scoring_type="llm_rubric",
            metadata={"key_phrases": pool_item.get("key_phrases", []), "ambiguity_type": pool_item["type"]},
        )

        return Scenario(
            scenario_id=f"Q_{split}_{scenario_idx:02d}",
            track="Q",
            split=split,
            user_name=user_name,
            question_date=q_date,
            sessions=all_sessions,
            questions=[q1],
            metadata={"domain": pool_item["domain"], "type": pool_item["type"]},
        )

    def generate_scenario(
        self,
        track: str,
        scenario_idx: int,
        split: str,
        seed: int,
        filler_count: int = 6,
    ) -> Scenario:
        track = track.upper()
        if track == "R":
            return self.generate_track_r_scenario(scenario_idx, split, seed, filler_count)
        elif track == "D":
            return self.generate_track_d_scenario(scenario_idx, split, seed, filler_count)
        elif track == "X":
            return self.generate_track_x_scenario(scenario_idx, split, seed, filler_count)
        elif track == "F":
            return self.generate_track_f_scenario(scenario_idx, split, seed, filler_count)
        elif track == "I":
            return self.generate_track_i_scenario(scenario_idx, split, seed, filler_count)
        elif track == "P":
            return self.generate_track_p_scenario(scenario_idx, split, seed, filler_count)
        elif track == "K":
            return self.generate_track_k_scenario(scenario_idx, split, seed, filler_count)
        elif track == "E":
            return self.generate_track_e_scenario(scenario_idx, split, seed, filler_count)
        elif track == "Q":
            return self.generate_track_q_scenario(scenario_idx, split, seed, filler_count)
        else:
            raise ValueError(f"Unknown track '{track}'. Valid tracks: R, D, X, F, I, P, K, E, Q")
