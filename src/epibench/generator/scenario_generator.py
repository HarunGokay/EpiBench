"""EpiBench Scenario Generator.

Generates deterministic multi-session conversational scenarios from curated
templated world models across 9 epistemic tracks:
- Track R: Revision & Contradiction (current / previous / as-of / count)
- Track D: Dependency Cascades (revisit set / rationale / still-standing yes-no)
- Track X: Hidden Constraints (unprompted boundary enforcement)
- Track F: Forecasts & Patterns (computed slippage counts/days & realism calibration)
- Track I: Interventions & Causal Claims (clean / confounded / not-implemented / single-obs)
- Track P: Provenance & False Memory Rejection
- Track K: Recall Control (static facts)
- Track E: Forgetting & Deletion (direct / semantic leak + control retention)
- Track Q: Uncertainty & Clarification (memorized candidate disambiguation)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import random
from typing import Any, Dict, List, Optional

from epibench.generator._pools_stress import (
    STRESS_MULTIHOP_DEPENDENCY_POOLS,
    STRESS_PORTFOLIO_INTERVENTION_POOLS,
    STRESS_ZERO_OVERLAP_CONSTRAINT_POOLS,
    build_stress_forecast_events,
    build_stress_portfolio_sessions,
    build_stress_revision_chain,
)
from epibench.generator.distractor_pool import DistractorPool, VALID_DISTRACTOR_MODES
from epibench.generator.pools import (
    ALL_POOLS,
    compute_forecast_gold,
)

GENERATOR_VERSION = "2.2.0"
VALID_TRACKS = ("R", "D", "X", "F", "I", "P", "K", "E", "Q")

PROFILE_ALIASES: Dict[str, str] = {
    "epistemic_stress": "stress",
}

BENCHMARK_PROFILES: Dict[str, Dict[str, Any]] = {
    "dev": {
        "filler_count": 6,
        "distractor_mode": "neutral",
        "token_budget": 1500,
        "epistemic_stress": False,
        "description": "Fast smoke/regression profile (6 neutral fillers; fits inside 1,500-token budget).",
    },
    "hard": {
        "filler_count": 120,
        "distractor_mode": "mixed",
        "token_budget": 1500,
        "epistemic_stress": False,
        "description": (
            "True budget-overflow & semantic-interference profile (120 mixed same-domain + neutral fillers; "
            "~13,800 history tokens vs 1,500 token budget, so only ~10.8% fits in budget)."
        ),
    },
    "stress": {
        "filler_count": 240,
        "distractor_mode": "mixed",
        "token_budget": 1500,
        "epistemic_stress": True,
        "description": (
            "Epistemic-stress profile (v2.2.0): 240 mixed distractors (~28k+ history tokens) plus "
            "28-step overflow revision chains (R), 24-step overflow forecast chains (F), 3-hop zero-overlap "
            "transitive dependency cascades (D), strictly 0-word-overlap cross-domain constraints (X), "
            "and 16-session longitudinal portfolio aggregation (I)."
        ),
    },
}


def resolve_profile(
    profile: Optional[str] = None,
    filler_count: Optional[int] = None,
    distractor_mode: Optional[str] = None,
    token_budget: Optional[int] = None,
) -> Dict[str, Any]:
    """Resolve benchmark profile defaults with explicit CLI/API overrides."""
    raw_key = (profile or "dev").strip().lower()
    prof_key = PROFILE_ALIASES.get(raw_key, raw_key)
    if prof_key not in BENCHMARK_PROFILES:
        raise ValueError(
            f"Unknown benchmark profile {profile!r}; expected one of {tuple(BENCHMARK_PROFILES.keys())}."
        )
    base = dict(BENCHMARK_PROFILES[prof_key])
    resolved_mode = (distractor_mode or base["distractor_mode"]).strip().lower()
    if resolved_mode not in VALID_DISTRACTOR_MODES:
        raise ValueError(
            f"Invalid distractor_mode {distractor_mode!r}; expected one of {VALID_DISTRACTOR_MODES}."
        )
    return {
        "profile": prof_key,
        "filler_count": base["filler_count"] if filler_count is None else int(filler_count),
        "distractor_mode": resolved_mode,
        "token_budget": base["token_budget"] if token_budget is None else int(token_budget),
        "epistemic_stress": bool(base.get("epistemic_stress", False)),
    }


@dataclass
class ScenarioQuestion:
    question_id: str
    question_text: str
    question_subkind: str
    gold_answer: Any
    rubric: str
    scoring_type: str  # 'string_match', 'set_overlap', 'integer_match', 'yes_no', 'llm_rubric'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Scenario:
    scenario_id: str
    track: str
    split: str
    user_name: str
    question_date: str
    sessions: List[Dict[str, Any]]
    questions: List[ScenarioQuestion]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generator_version": GENERATOR_VERSION,
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
    """Format a UTC datetime as canonical sortable ISO timestamp."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def parse_session_date(value: str | datetime) -> datetime:
    """Parse canonical ISO or slash-formatted timestamps into UTC datetime."""
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    raw = str(value).strip().replace("/", "-")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Unparseable session date: {value!r}")


def derive_seed(base_seed: int, track: str, scenario_idx: int, purpose: str) -> int:
    """Derive a collision-free deterministic integer seed via SHA-256."""
    payload = f"{base_seed}:{track.upper()}:{scenario_idx}:{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def allocate_filler_dates(
    evidence_dates: List[datetime],
    question_date: datetime,
    filler_count: int,
    rng: random.Random,
) -> List[datetime]:
    """Allocate `filler_count` timestamps spread across [window_start, question_date).

    Guarantees:
    - Every allocated timestamp is strictly earlier than `question_date`.
    - Timestamps never collide with exact `evidence_dates`.
    - Works cleanly for arbitrarily large `filler_count` (0, 6, 24, 60, 200+).
    """
    if filler_count <= 0:
        return []

    earliest_ev = min(evidence_dates)
    window_start = earliest_ev - timedelta(days=14)
    window_end = question_date - timedelta(hours=1)
    if window_end <= window_start:
        window_start = question_date - timedelta(days=30)
        window_end = question_date - timedelta(hours=1)

    total_seconds = int((window_end - window_start).total_seconds())
    evidence_secs = {int((d - window_start).total_seconds()) for d in evidence_dates}

    allocated: List[datetime] = []
    used_secs = set(evidence_secs)

    for i in range(filler_count):
        bucket_lo = (i * total_seconds) // filler_count
        bucket_hi = max(bucket_lo, ((i + 1) * total_seconds) // filler_count - 1)
        sec = rng.randint(bucket_lo, bucket_hi)
        # Avoid colliding on exact second with evidence or previous fillers
        while sec in used_secs and sec < total_seconds - 1:
            sec += 1
        while sec in used_secs and sec > 0:
            sec -= 1
        used_secs.add(sec)
        dt = window_start + timedelta(seconds=sec)
        assert dt < question_date, f"Allocated filler date {dt} >= question_date {question_date}"
        allocated.append(dt)

    return allocated


class ScenarioGenerator:
    """Deterministic generator for EpiBench evaluation scenarios."""

    def __init__(
        self,
        distractor_pool: Optional[DistractorPool] = None,
        custom_pools: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        distractor_mode: str = "neutral",
        distractor_file: Optional[Path] = None,
        epistemic_stress: bool = False,
    ):
        if distractor_pool is not None:
            self.distractor_pool = distractor_pool
            self.distractor_mode = getattr(distractor_pool, "mode", distractor_mode)
        else:
            self.distractor_pool = DistractorPool.get_instance(
                mode=distractor_mode,
                dataset_path=distractor_file,
            )
            self.distractor_mode = "custom" if distractor_file is not None else self.distractor_pool.mode
        self._pools = custom_pools if custom_pools is not None else ALL_POOLS
        self._has_custom_pools = custom_pools is not None
        self.epistemic_stress = bool(epistemic_stress)

    def _get_pool(self, track: str, split: str) -> List[Dict[str, Any]]:
        if split == "held_out" and not self._has_custom_pools:
            raise ValueError(
                "The 'held_out' split requires a separate private pool passed via custom_pools. "
                "Use split='dev' for the public benchmark pools."
            )
        return self._pools[track]

    def _interleave_fillers(
        self,
        evidence_sessions: List[Dict[str, Any]],
        question_date_dt: datetime,
        track: str,
        scenario_idx: int,
        seed: int,
        filler_count: int,
    ) -> List[Dict[str, Any]]:
        evidence_dates = [parse_session_date(s["session_date"]) for s in evidence_sessions]
        for ed in evidence_dates:
            if ed > question_date_dt:
                raise ValueError(f"Evidence date {ed} exceeds question_date {question_date_dt}")

        filler_seed = derive_seed(seed, track, scenario_idx, "fillers")
        date_seed = derive_seed(seed, track, scenario_idx, "filler_dates")
        date_rng = random.Random(date_seed)

        fillers = self.distractor_pool.sample_distractors(filler_count, seed=filler_seed)
        filler_dates = allocate_filler_dates(evidence_dates, question_date_dt, len(fillers), date_rng)

        all_sessions = list(evidence_sessions)
        for idx, (f_sess, f_dt) in enumerate(zip(fillers, filler_dates)):
            all_sessions.append({
                "session_id": f"filler_{idx:04d}",
                "session_date": format_date(f_dt),
                "turns": f_sess["turns"],
                "is_evidence": False,
            })

        all_sessions.sort(key=lambda s: (s["session_date"], s["session_id"]))
        return all_sessions

    def generate_track_r_scenario(
        self,
        scenario_idx: int,
        split: str,
        seed: int,
        filler_count: int = 6,
    ) -> Scenario:
        pool = self._get_pool("R", split)
        pool_item = pool[scenario_idx % len(pool)]
        user_name = f"User_R_{split}_{scenario_idx:02d}"

        base_date = datetime(2023, 1, 15, 10, 0, 0, tzinfo=timezone.utc) if self.epistemic_stress else datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        if self.epistemic_stress:
            chain = build_stress_revision_chain(pool_item)
            values = chain["values"]
            val_aliases = chain["value_aliases"]
            user_phrasings = chain["user_phrasings"]
            step_days = 24  # 28 steps across ~648 days (~21 months)
        else:
            values = pool_item["values"]
            val_aliases = pool_item["value_aliases"]
            user_phrasings = [pool_item["user_phrasings"][i].format(val=val) for i, val in enumerate(values)]
            step_days = 45 if len(values) == 3 else 30
        dates = [base_date + timedelta(days=step_days * i) for i in range(len(values))]

        evidence_sessions = []
        for i, val in enumerate(values):
            phrase = user_phrasings[i]
            turns = [
                {"role": "user", "content": phrase},
                {"role": "assistant", "content": f"Understood, I have updated my notes regarding {pool_item['entity_phrase']}."},
            ]
            evidence_sessions.append({
                "session_id": f"s_rev_{i + 1}",
                "session_date": format_date(dates[i]),
                "turns": turns,
                "is_evidence": True,
            })

        q_date_dt = dates[-1] + timedelta(days=15)
        all_sessions = self._interleave_fillers(
            evidence_sessions, q_date_dt, "R", scenario_idx, seed, filler_count
        )

        as_of_target_date = dates[0] + timedelta(days=10)
        as_of_date_str = as_of_target_date.strftime("%Y-%m-%d")

        q1 = ScenarioQuestion(
            question_id=f"R_{split}_{scenario_idx:02d}_q1",
            question_text=pool_item["question_templates"]["current"],
            question_subkind="current",
            gold_answer=values[-1],
            rubric=f"Exact match for current value: {values[-1]}",
            scoring_type="string_match",
            metadata={
                "gold_aliases": list(val_aliases[values[-1]]),
                "stale_value": values[-2],
                "stale_aliases": list(val_aliases[values[-2]]),
                "wrong_values": [v for v in values if v != values[-1]],
            },
        )
        q2 = ScenarioQuestion(
            question_id=f"R_{split}_{scenario_idx:02d}_q2",
            question_text=pool_item["question_templates"]["previous"],
            question_subkind="previous",
            gold_answer=values[-2],
            rubric=f"Exact match for previous value: {values[-2]}",
            scoring_type="string_match",
            metadata={
                "gold_aliases": list(val_aliases[values[-2]]),
                "wrong_values": [v for v in values if v != values[-2]],
            },
        )
        q3 = ScenarioQuestion(
            question_id=f"R_{split}_{scenario_idx:02d}_q3",
            question_text=pool_item["question_templates"]["as_of"].format(date=as_of_date_str),
            question_subkind="as_of",
            gold_answer=values[0],
            rubric=f"Exact match for value as of {as_of_date_str}: {values[0]}",
            scoring_type="string_match",
            metadata={
                "as_of_date": as_of_date_str,
                "gold_aliases": list(val_aliases[values[0]]),
                "wrong_values": [v for v in values if v != values[0]],
            },
        )
        q4 = ScenarioQuestion(
            question_id=f"R_{split}_{scenario_idx:02d}_q4",
            question_text=pool_item["question_templates"]["count"],
            question_subkind="count",
            gold_answer=len(values) - 1,
            rubric=f"Exact integer count of switches/changes: {len(values) - 1}",
            scoring_type="integer_match",
            metadata={"wrong_values": [len(values), 0]},
        )

        sc_meta: Dict[str, Any] = {
            "template_id": pool_item["id"],
            "pool_entity": pool_item["entity"],
            "domain": pool_item["domain"],
        }
        if self.epistemic_stress:
            sc_meta["epistemic_stress"] = True
            sc_meta["revision_chain_length"] = len(values)
            sc_meta["overflow_evidence_sessions"] = [s["session_id"] for s in evidence_sessions]
            sc_meta["all_stale_values"] = list(values[:-1])
            sc_meta["all_stale_aliases"] = [a for v in values[:-1] for a in val_aliases[v]]
            q1.metadata["all_stale_aliases"] = list(sc_meta["all_stale_aliases"])

        return Scenario(
            scenario_id=f"R_{split}_{scenario_idx:02d}",
            track="R",
            split=split,
            user_name=user_name,
            question_date=format_date(q_date_dt),
            sessions=all_sessions,
            questions=[q1, q2, q3, q4],
            metadata=sc_meta,
        )

    def generate_track_d_scenario(
        self,
        scenario_idx: int,
        split: str,
        seed: int,
        filler_count: int = 6,
    ) -> Scenario:
        if self.epistemic_stress and not self._has_custom_pools:
            pool_item = STRESS_MULTIHOP_DEPENDENCY_POOLS[scenario_idx % len(STRESS_MULTIHOP_DEPENDENCY_POOLS)]
            user_name = f"User_D_{split}_{scenario_idx:02d}"
            base_date = datetime(2024, 1, 10, 9, 0, 0, tzinfo=timezone.utc)
            d_hop1 = base_date
            d_hop2 = base_date + timedelta(days=14)
            d_hop3 = base_date + timedelta(days=28)
            d_unaff = base_date + timedelta(days=40)
            d_hop4 = base_date + timedelta(days=90)
            q_date_dt = d_hop4 + timedelta(days=10)

            evidence_sessions = [
                {
                    "session_id": "s_hop1_premise",
                    "session_date": format_date(d_hop1),
                    "turns": [
                        {"role": "user", "content": f"Upstream operational premise: {pool_item['hop1_premise']}"},
                        {"role": "assistant", "content": "Noted. Logged this upstream operational premise."},
                    ],
                    "is_evidence": True,
                },
                {
                    "session_id": "s_hop2_bridge",
                    "session_date": format_date(d_hop2),
                    "turns": [
                        {"role": "user", "content": f"Mid-level policy dependency: {pool_item['hop2_bridge']}"},
                        {"role": "assistant", "content": "Recorded this intermediate operational dependency."},
                    ],
                    "is_evidence": True,
                },
                {
                    "session_id": "s_hop3_terminal",
                    "session_date": format_date(d_hop3),
                    "turns": [
                        {"role": "user", "content": f"Downstream commitments locked in: {pool_item['hop3_terminal']}"},
                        {"role": "assistant", "content": "Logged these downstream commitments."},
                    ],
                    "is_evidence": True,
                },
                {
                    "session_id": "s_unaffected",
                    "session_date": format_date(d_unaff),
                    "turns": [
                        {"role": "user", "content": f"Independent operational note: {pool_item['unaffected_premise']} {pool_item['unaffected_decision']['text']}"},
                        {"role": "assistant", "content": "Acknowledged. Logged this separate independent commitment."},
                    ],
                    "is_evidence": True,
                },
                {
                    "session_id": "s_hop4_trigger",
                    "session_date": format_date(d_hop4),
                    "turns": [
                        {"role": "user", "content": pool_item["hop4_trigger"]},
                        {"role": "assistant", "content": "Understood. That upstream change has broad implications."},
                    ],
                    "is_evidence": True,
                },
            ]

            all_sessions = self._interleave_fillers(
                evidence_sessions, q_date_dt, "D", scenario_idx, seed, filler_count
            )

            gold_decisions = [
                {"id": d["id"], "label": d["label"], "aliases": list(d["aliases"])}
                for d in pool_item["affected_decisions"]
            ]
            unaffected = pool_item["unaffected_decision"]
            unaffected_decisions = [
                {"id": unaffected["id"], "label": unaffected["label"], "aliases": list(unaffected["aliases"])}
            ]
            gold_affected_labels = [d["label"] for d in pool_item["affected_decisions"]]

            q1 = ScenarioQuestion(
                question_id=f"D_{split}_{scenario_idx:02d}_q1",
                question_text=pool_item["questions"]["revisit_set"],
                question_subkind="revisit_set",
                gold_answer=gold_affected_labels,
                rubric=f"Must identify the exact affected downstream commitments ({gold_affected_labels}) across the 3-hop chain and exclude unaffected commitments ({unaffected['label']}).",
                scoring_type="set_overlap",
                metadata={
                    "gold_decisions": gold_decisions,
                    "unaffected_decisions": unaffected_decisions,
                    "gold_ids": [d["id"] for d in gold_decisions],
                    "unaffected": unaffected["label"],
                    "bridge_sessions": ["s_hop2_bridge"],
                    "terminal_sessions": ["s_hop3_terminal"],
                    "hop_sessions": ["s_hop1_premise", "s_hop2_bridge", "s_hop3_terminal", "s_hop4_trigger"],
                },
            )
            q2 = ScenarioQuestion(
                question_id=f"D_{split}_{scenario_idx:02d}_q2",
                question_text=pool_item["questions"]["why"],
                question_subkind="why",
                gold_answer=pool_item["affected_decisions"][0]["reason"],
                rubric=f"Must accurately trace the full 3-hop dependency chain: {pool_item['affected_decisions'][0]['reason']}",
                scoring_type="llm_rubric",
                metadata={
                    "bridge_sessions": ["s_hop2_bridge"],
                    "terminal_sessions": ["s_hop3_terminal"],
                    "hop_sessions": ["s_hop1_premise", "s_hop2_bridge", "s_hop3_terminal", "s_hop4_trigger"],
                },
            )
            stands_gold = pool_item["still_standing_gold"]
            q3 = ScenarioQuestion(
                question_id=f"D_{split}_{scenario_idx:02d}_q3",
                question_text=pool_item["questions"]["still_standing"],
                question_subkind="stands",
                gold_answer=stands_gold,
                rubric="Must answer YES because this commitment rests on a separate independent premise.",
                scoring_type="yes_no",
                metadata={"still_standing_target": pool_item["still_standing_target"]},
            )
            return Scenario(
                scenario_id=f"D_{split}_{scenario_idx:02d}",
                track="D",
                split=split,
                user_name=user_name,
                question_date=format_date(q_date_dt),
                sessions=all_sessions,
                questions=[q1, q2, q3],
                metadata={
                    "template_id": pool_item["id"],
                    "domain": pool_item["domain"],
                    "epistemic_stress": True,
                    "hop_count": 3,
                    "hop_sessions": ["s_hop1_premise", "s_hop2_bridge", "s_hop3_terminal", "s_hop4_trigger"],
                    "bridge_sessions": ["s_hop2_bridge"],
                    "terminal_sessions": ["s_hop3_terminal"],
                    "trigger_session": "s_hop4_trigger",
                },
            )

        pool = self._get_pool("D", split)
        pool_item = pool[scenario_idx % len(pool)]
        user_name = f"User_D_{split}_{scenario_idx:02d}"

        base_date = datetime(2024, 2, 1, 9, 0, 0, tzinfo=timezone.utc)
        d_assump1 = base_date
        d_decisions1 = base_date + timedelta(days=10)
        d_assump2 = base_date + timedelta(days=20)
        d_revision1 = base_date + timedelta(days=50)
        q_date_dt = d_revision1 + timedelta(days=12)

        affected_texts = " Also, ".join(d["text"] for d in pool_item["affected_decisions"])
        evidence_sessions = [
            {
                "session_id": "s_assump_1",
                "session_date": format_date(d_assump1),
                "turns": [
                    {"role": "user", "content": f"Project update: {pool_item['assumption_1']}"},
                    {"role": "assistant", "content": "Noted. I will keep this assumption in mind for our planning."},
                ],
                "is_evidence": True,
            },
            {
                "session_id": "s_decisions_1",
                "session_date": format_date(d_decisions1),
                "turns": [
                    {"role": "user", "content": f"We finalized these commitments today: {affected_texts}"},
                    {"role": "assistant", "content": "Logged these commitments based on your current plan."},
                ],
                "is_evidence": True,
            },
            {
                "session_id": "s_assump_2",
                "session_date": format_date(d_assump2),
                "turns": [
                    {"role": "user", "content": f"Separate operational note: {pool_item['assumption_2']} {pool_item['unaffected_decision']['text']}"},
                    {"role": "assistant", "content": "Acknowledged. Recorded this separate operational decision."},
                ],
                "is_evidence": True,
            },
            {
                "session_id": "s_revision_1",
                "session_date": format_date(d_revision1),
                "turns": [
                    {"role": "user", "content": f"Critical update: {pool_item['assumption_1_revised']}"},
                    {"role": "assistant", "content": "Understood. That is a major change to the earlier plan."},
                ],
                "is_evidence": True,
            },
        ]

        all_sessions = self._interleave_fillers(
            evidence_sessions, q_date_dt, "D", scenario_idx, seed, filler_count
        )

        gold_decisions = [
            {"id": d["id"], "label": d["label"], "aliases": list(d["aliases"])}
            for d in pool_item["affected_decisions"]
        ]
        unaffected = pool_item["unaffected_decision"]
        unaffected_decisions = [
            {"id": unaffected["id"], "label": unaffected["label"], "aliases": list(unaffected["aliases"])}
        ]
        gold_affected_labels = [d["label"] for d in pool_item["affected_decisions"]]

        q1 = ScenarioQuestion(
            question_id=f"D_{split}_{scenario_idx:02d}_q1",
            question_text=pool_item["questions"]["revisit_set"],
            question_subkind="revisit_set",
            gold_answer=gold_affected_labels,
            rubric=f"Must identify the exact affected commitments ({gold_affected_labels}) and exclude unaffected commitments ({unaffected['label']}).",
            scoring_type="set_overlap",
            metadata={
                "gold_decisions": gold_decisions,
                "unaffected_decisions": unaffected_decisions,
                "gold_ids": [d["id"] for d in gold_decisions],
                "unaffected": unaffected["label"],
            },
        )
        q2 = ScenarioQuestion(
            question_id=f"D_{split}_{scenario_idx:02d}_q2",
            question_text=pool_item["questions"]["why"],
            question_subkind="why",
            gold_answer=pool_item["affected_decisions"][0]["reason"],
            rubric=f"Must accurately cite the underlying assumption change ({pool_item['assumption_1_revised']}) explaining why the decision is invalidated: {pool_item['affected_decisions'][0]['reason']}",
            scoring_type="llm_rubric",
        )
        stands_gold = pool_item["still_standing_gold"]
        stands_rubric = (
            "Must answer YES because this commitment does not depend on the revised assumption."
            if stands_gold == "YES"
            else "Must answer NO because this commitment depended on the revised assumption and must be revisited."
        )
        q3 = ScenarioQuestion(
            question_id=f"D_{split}_{scenario_idx:02d}_q3",
            question_text=pool_item["questions"]["still_standing"],
            question_subkind="stands",
            gold_answer=stands_gold,
            rubric=stands_rubric,
            scoring_type="yes_no",
            metadata={"still_standing_target": pool_item["still_standing_target"]},
        )

        return Scenario(
            scenario_id=f"D_{split}_{scenario_idx:02d}",
            track="D",
            split=split,
            user_name=user_name,
            question_date=format_date(q_date_dt),
            sessions=all_sessions,
            questions=[q1, q2, q3],
            metadata={"template_id": pool_item["id"], "domain": pool_item["domain"]},
        )

    def generate_track_x_scenario(
        self,
        scenario_idx: int,
        split: str,
        seed: int,
        filler_count: int = 6,
    ) -> Scenario:
        if self.epistemic_stress and not self._has_custom_pools:
            pool_item = STRESS_ZERO_OVERLAP_CONSTRAINT_POOLS[scenario_idx % len(STRESS_ZERO_OVERLAP_CONSTRAINT_POOLS)]
        else:
            pool = self._get_pool("X", split)
            pool_item = pool[scenario_idx % len(pool)]
        user_name = f"User_X_{split}_{scenario_idx:02d}"

        base_date = datetime(2024, 3, 1, 14, 0, 0, tzinfo=timezone.utc)
        q_date_dt = base_date + timedelta(days=60)

        evidence_sessions = [
            {
                "session_id": "s_constraint",
                "session_date": format_date(base_date),
                "turns": [
                    {"role": "user", "content": f"Just a heads up on my rules and limits: {pool_item['constraint_statement']}"},
                    {"role": "assistant", "content": "Understood."},
                ],
                "is_evidence": True,
            }
        ]

        all_sessions = self._interleave_fillers(
            evidence_sessions, q_date_dt, "X", scenario_idx, seed, filler_count
        )

        q1 = ScenarioQuestion(
            question_id=f"X_{split}_{scenario_idx:02d}_q1",
            question_text=pool_item["query"],
            question_subkind="hidden_constraint",
            gold_answer=pool_item["gold_response_requirement"],
            rubric=f"{pool_item['gold_response_requirement']} Must specifically cite the memorized constraint ({pool_item['constraint_keyword']}).",
            scoring_type="llm_rubric",
            metadata={
                "constraint_keyword": pool_item["constraint_keyword"],
                "constraint_aliases": list(pool_item["constraint_aliases"]),
                "verdict": pool_item["gold_verdict"],
            },
        )

        x_meta: Dict[str, Any] = {"template_id": pool_item["id"], "domain": pool_item["domain"]}
        if self.epistemic_stress:
            x_meta["epistemic_stress"] = True
            x_meta["zero_lexical_overlap"] = True
        return Scenario(
            scenario_id=f"X_{split}_{scenario_idx:02d}",
            track="X",
            split=split,
            user_name=user_name,
            question_date=format_date(q_date_dt),
            sessions=all_sessions,
            questions=[q1],
            metadata=x_meta,
        )

    def generate_track_f_scenario(
        self,
        scenario_idx: int,
        split: str,
        seed: int,
        filler_count: int = 6,
    ) -> Scenario:
        pool = self._get_pool("F", split)
        pool_item = pool[scenario_idx % len(pool)]
        user_name = f"User_F_{split}_{scenario_idx:02d}"

        events = build_stress_forecast_events(pool_item) if self.epistemic_stress else pool_item["events"]
        computed = compute_forecast_gold(events)
        move_count = computed["move_count"]
        slip_days = computed["total_slip_days"]
        is_realistic = pool_item["is_realistic"]

        evidence_sessions = []
        for idx, ev in enumerate(events):
            dt = datetime.strptime(ev["date"], "%Y-%m-%d").replace(hour=10, tzinfo=timezone.utc)
            evidence_sessions.append({
                "session_id": f"s_event_{idx + 1}",
                "session_date": format_date(dt),
                "turns": [
                    {"role": "user", "content": ev["text"]},
                    {"role": "assistant", "content": "Recorded this schedule update."},
                ],
                "is_evidence": True,
            })

        last_dt = parse_session_date(evidence_sessions[-1]["session_date"])
        q_date_dt = last_dt + timedelta(days=14)
        all_sessions = self._interleave_fillers(
            evidence_sessions, q_date_dt, "F", scenario_idx, seed, filler_count
        )

        if is_realistic:
            realism_gold = (
                f"Yes, realistic. The recorded history for {pool_item['project']} shows on-time delivery "
                f"with {move_count} slips and {slip_days} slip days ({computed['initial_target_date']} completed on {computed['final_completion_date']})."
            )
            realism_rubric = (
                f"To be CORRECT, the response MUST conclude the new target IS realistic AND explicitly cite "
                f"the recorded on-time delivery history of {pool_item['project']} (0 slips / completed on {computed['final_completion_date']}). "
                f"Answering without citing the recorded history is INCORRECT."
            )
        else:
            realism_gold = (
                f"No, unrealistic. {pool_item['project']} slipped {move_count} times for a total of {slip_days} days "
                f"(from {computed['initial_target_date']} to {computed['final_completion_date']})."
            )
            realism_rubric = (
                f"To be CORRECT, the response MUST conclude the new target is NOT realistic AND explicitly cite "
                f"the recorded slippage history of {pool_item['project']} ({move_count} delays / {slip_days} days slip from "
                f"{computed['initial_target_date']} to {computed['final_completion_date']}). "
                f"Saying 'no' or 'cannot determine' without citing the recorded project history is INCORRECT."
            )

        q1 = ScenarioQuestion(
            question_id=f"F_{split}_{scenario_idx:02d}_q1",
            question_text=f"For {pool_item['project']}, how many times did the target completion date move or slip before completion?",
            question_subkind="move_count",
            gold_answer=move_count,
            rubric=f"Exact integer count of schedule moves/slips: {move_count}",
            scoring_type="integer_match",
            metadata={"wrong_values": [move_count + 1, max(0, move_count - 1) if move_count > 0 else 2]},
        )
        q2 = ScenarioQuestion(
            question_id=f"F_{split}_{scenario_idx:02d}_q2",
            question_text=f"For {pool_item['project']}, how many total calendar days did completion slip from the original promised target date to final completion?",
            question_subkind="slip_days",
            gold_answer=slip_days,
            rubric=f"Exact total slip duration in calendar days: {slip_days}",
            scoring_type="integer_match",
            metadata={
                "initial_target_date": computed["initial_target_date"],
                "final_completion_date": computed["final_completion_date"],
                "wrong_values": [slip_days - 1 if slip_days > 0 else 7, slip_days + 1],
            },
        )
        q3 = ScenarioQuestion(
            question_id=f"F_{split}_{scenario_idx:02d}_q3",
            question_text=pool_item["new_target_query"],
            question_subkind="realism",
            gold_answer=realism_gold,
            rubric=realism_rubric,
            scoring_type="llm_rubric",
            metadata={
                "is_realistic": is_realistic,
                "move_count": move_count,
                "total_slip_days": slip_days,
            },
        )

        f_meta: Dict[str, Any] = {
            "template_id": pool_item["id"],
            "domain": pool_item["domain"],
            "project": pool_item["project"],
            "computed_gold": computed,
        }
        if self.epistemic_stress:
            f_meta["epistemic_stress"] = True
            f_meta["overflow_evidence_sessions"] = [s["session_id"] for s in evidence_sessions]
        return Scenario(
            scenario_id=f"F_{split}_{scenario_idx:02d}",
            track="F",
            split=split,
            user_name=user_name,
            question_date=format_date(q_date_dt),
            sessions=all_sessions,
            questions=[q1, q2, q3],
            metadata=f_meta,
        )

    def generate_track_i_scenario(
        self,
        scenario_idx: int,
        split: str,
        seed: int,
        filler_count: int = 6,
    ) -> Scenario:
        if self.epistemic_stress and not self._has_custom_pools:
            pool_item = STRESS_PORTFOLIO_INTERVENTION_POOLS[scenario_idx % len(STRESS_PORTFOLIO_INTERVENTION_POOLS)]
            user_name = f"User_I_{split}_{scenario_idx:02d}"
            base_date = datetime(2024, 1, 5, 10, 0, 0, tzinfo=timezone.utc)
            evidence_sessions = build_stress_portfolio_sessions(pool_item, base_date)
            last_ev_dt = parse_session_date(evidence_sessions[-1]["session_date"])
            q_date_dt = last_ev_dt + timedelta(days=14)
            all_sessions = self._interleave_fillers(
                evidence_sessions, q_date_dt, "I", scenario_idx, seed, filler_count
            )
            gold_rubric = (
                f"Must identify '{pool_item['clean_name']}' as the ONLY intervention with clean, unconfounded "
                f"longitudinal evidence across 3 post-change readings, AND explain why '{pool_item['confounded_name']}' "
                f"is confounded ({pool_item['confounder_note']}), '{pool_item['unimplemented_name']}' was never "
                f"implemented, and '{pool_item['anecdotal_name']}' has only a single anecdotal observation."
            )
            q1 = ScenarioQuestion(
                question_id=f"I_{split}_{scenario_idx:02d}_q1",
                question_text=(
                    f"Across all 16 recorded experiment logs for {pool_item['subject']}, which single intervention "
                    f"has clean unconfounded longitudinal evidence of working, and why must the other three candidates "
                    f"not be credited with causal efficacy?"
                ),
                question_subkind="causal_claim",
                gold_answer=gold_rubric,
                rubric=gold_rubric,
                scoring_type="llm_rubric",
                metadata={
                    "variant": "portfolio_aggregation",
                    "gold_strength": "strong_clean_vs_confounded_unimplemented_anecdotal",
                    "clean_intervention": pool_item["clean_name"],
                    "confounded_intervention": pool_item["confounded_name"],
                    "required_evidence_phrases": [
                        pool_item["clean_name"],
                        pool_item["confounded_name"],
                        pool_item["unimplemented_name"],
                        pool_item["anecdotal_name"],
                    ],
                },
            )
            return Scenario(
                scenario_id=f"I_{split}_{scenario_idx:02d}",
                track="I",
                split=split,
                user_name=user_name,
                question_date=format_date(q_date_dt),
                sessions=all_sessions,
                questions=[q1],
                metadata={
                    "template_id": pool_item["id"],
                    "variant": "portfolio_aggregation",
                    "domain": pool_item["domain"],
                    "epistemic_stress": True,
                    "overflow_evidence_sessions": [s["session_id"] for s in evidence_sessions],
                },
            )

        pool = self._get_pool("I", split)
        pool_item = pool[scenario_idx % len(pool)]
        user_name = f"User_I_{split}_{scenario_idx:02d}"

        base_date = datetime(2024, 4, 1, 10, 0, 0, tzinfo=timezone.utc)
        evidence_sessions = []
        for idx, line in enumerate(pool_item["timeline"]):
            s_date = base_date + timedelta(days=14 * idx)
            evidence_sessions.append({
                "session_id": f"s_timeline_{idx + 1}",
                "session_date": format_date(s_date),
                "turns": [
                    {"role": "user", "content": line},
                    {"role": "assistant", "content": "Logged these observations."},
                ],
                "is_evidence": True,
            })

        last_ev_dt = parse_session_date(evidence_sessions[-1]["session_date"])
        q_date_dt = last_ev_dt + timedelta(days=14)
        all_sessions = self._interleave_fillers(
            evidence_sessions, q_date_dt, "I", scenario_idx, seed, filler_count
        )

        q1 = ScenarioQuestion(
            question_id=f"I_{split}_{scenario_idx:02d}_q1",
            question_text=pool_item["query"],
            question_subkind="causal_claim",
            gold_answer=pool_item["gold_rubric"],
            rubric=pool_item["gold_rubric"] + " Saying there is no context or giving a generic answer without citing the recorded observations is INCORRECT.",
            scoring_type="llm_rubric",
            metadata={
                "variant": pool_item["variant"],
                "gold_strength": pool_item["gold_strength"],
                "required_evidence_phrases": list(pool_item["required_evidence_phrases"]),
            },
        )

        return Scenario(
            scenario_id=f"I_{split}_{scenario_idx:02d}",
            track="I",
            split=split,
            user_name=user_name,
            question_date=format_date(q_date_dt),
            sessions=all_sessions,
            questions=[q1],
            metadata={
                "template_id": pool_item["id"],
                "variant": pool_item["variant"],
                "domain": pool_item["domain"],
            },
        )

    def generate_track_p_scenario(
        self,
        scenario_idx: int,
        split: str,
        seed: int,
        filler_count: int = 6,
    ) -> Scenario:
        pool = self._get_pool("P", split)
        pool_item = pool[scenario_idx % len(pool)]
        user_name = f"User_P_{split}_{scenario_idx:02d}"

        base_date = datetime(2024, 5, 1, 11, 0, 0, tzinfo=timezone.utc)
        q_date_dt = base_date + timedelta(days=60)
        evidence_sessions = [
            {
                "session_id": "s_provenance",
                "session_date": format_date(base_date),
                "turns": pool_item["timeline"],
                "is_evidence": True,
            }
        ]

        all_sessions = self._interleave_fillers(
            evidence_sessions, q_date_dt, "P", scenario_idx, seed, filler_count
        )

        q1 = ScenarioQuestion(
            question_id=f"P_{split}_{scenario_idx:02d}_q1",
            question_text=pool_item["query"],
            question_subkind="provenance",
            gold_answer=pool_item["gold_value"],
            rubric=(
                f"Must assert the genuine user-authored fact ({pool_item['gold_value']}) "
                f"and must NOT endorse the false or retracted claim ({pool_item['forbidden_distractor']})."
            ),
            scoring_type="llm_rubric",
            metadata={
                "gold_aliases": list(pool_item["gold_aliases"]),
                "forbidden_distractor": pool_item["forbidden_distractor"],
                "forbidden_aliases": list(pool_item["forbidden_aliases"]),
                "variant": pool_item["variant"],
            },
        )

        return Scenario(
            scenario_id=f"P_{split}_{scenario_idx:02d}",
            track="P",
            split=split,
            user_name=user_name,
            question_date=format_date(q_date_dt),
            sessions=all_sessions,
            questions=[q1],
            metadata={
                "template_id": pool_item["id"],
                "variant": pool_item["variant"],
                "domain": pool_item["domain"],
            },
        )

    def generate_track_k_scenario(
        self,
        scenario_idx: int,
        split: str,
        seed: int,
        filler_count: int = 6,
    ) -> Scenario:
        pool = self._get_pool("K", split)
        pool_item = pool[scenario_idx % len(pool)]
        user_name = f"User_K_{split}_{scenario_idx:02d}"

        base_date = datetime(2024, 1, 10, 8, 30, 0, tzinfo=timezone.utc)
        q_date_dt = base_date + timedelta(days=80)
        evidence_sessions = [
            {
                "session_id": "s_fact",
                "session_date": format_date(base_date),
                "turns": [
                    {"role": "user", "content": pool_item["statement"]},
                    {"role": "assistant", "content": "Noted, thank you for sharing that detail."},
                ],
                "is_evidence": True,
            }
        ]

        all_sessions = self._interleave_fillers(
            evidence_sessions, q_date_dt, "K", scenario_idx, seed, filler_count
        )

        q1 = ScenarioQuestion(
            question_id=f"K_{split}_{scenario_idx:02d}_q1",
            question_text=pool_item["query"],
            question_subkind="static_fact",
            gold_answer=pool_item["gold_value"],
            rubric=f"Exact match for static fact: {pool_item['gold_value']}",
            scoring_type="string_match",
            metadata={"gold_aliases": list(pool_item["gold_aliases"])},
        )

        return Scenario(
            scenario_id=f"K_{split}_{scenario_idx:02d}",
            track="K",
            split=split,
            user_name=user_name,
            question_date=format_date(q_date_dt),
            sessions=all_sessions,
            questions=[q1],
            metadata={
                "template_id": pool_item["id"],
                "domain": pool_item["domain"],
                "entity": pool_item["entity"],
            },
        )

    def generate_track_e_scenario(
        self,
        scenario_idx: int,
        split: str,
        seed: int,
        filler_count: int = 6,
    ) -> Scenario:
        pool = self._get_pool("E", split)
        pool_item = pool[scenario_idx % len(pool)]
        user_name = f"User_E_{split}_{scenario_idx:02d}"

        base_date = datetime(2024, 2, 1, 10, 0, 0, tzinfo=timezone.utc)
        d_session1 = base_date
        d_session2 = base_date + timedelta(days=10)
        d_delete = base_date + timedelta(days=30)
        q_date_dt = base_date + timedelta(days=45)

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
                {"role": "assistant", "content": "Got it, I have noted this detail."},
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
            "deleted_aliases": list(pool_item["forbidden_aliases"]),
        }

        all_sessions = self._interleave_fillers(
            [s1, s2, s3], q_date_dt, "E", scenario_idx, seed, filler_count
        )

        withhold_note = (
            " Stating that the context does not contain this information, that no record is available/on file, "
            "or that it was deleted all satisfy withholding as long as the secret is not disclosed."
        )
        q1 = ScenarioQuestion(
            question_id=f"E_{split}_{scenario_idx:02d}_q1",
            question_text=pool_item["direct_query"],
            question_subkind="deletion_direct",
            gold_answer="DELETED / WITHHELD",
            rubric=pool_item["rubric_direct"] + withhold_note,
            scoring_type="llm_rubric",
            metadata={
                "forbidden_distractor": pool_item["target_item"],
                "forbidden_aliases": list(pool_item["forbidden_aliases"]),
                "target_type": pool_item["target_type"],
            },
        )
        q2 = ScenarioQuestion(
            question_id=f"E_{split}_{scenario_idx:02d}_q2",
            question_text=pool_item["semantic_query"],
            question_subkind="deletion_semantic",
            gold_answer="DELETED / WITHHELD",
            rubric=pool_item["rubric_semantic"] + withhold_note,
            scoring_type="llm_rubric",
            metadata={
                "forbidden_distractor": pool_item["target_item"],
                "forbidden_aliases": list(pool_item["forbidden_aliases"]),
                "target_type": pool_item["target_type"],
            },
        )
        q3 = ScenarioQuestion(
            question_id=f"E_{split}_{scenario_idx:02d}_q3",
            question_text=pool_item["control_query"],
            question_subkind="deletion_control_retention",
            gold_answer=pool_item["control_gold"],
            rubric=f"Exact match for non-deleted control fact: {pool_item['control_gold']}",
            scoring_type="string_match",
            metadata={"gold_aliases": list(pool_item["control_aliases"])},
        )

        return Scenario(
            scenario_id=f"E_{split}_{scenario_idx:02d}",
            track="E",
            split=split,
            user_name=user_name,
            question_date=format_date(q_date_dt),
            sessions=all_sessions,
            questions=[q1, q2, q3],
            metadata={
                "template_id": pool_item["id"],
                "domain": pool_item["domain"],
                "target_item": pool_item["target_item"],
            },
        )

    def generate_track_q_scenario(
        self,
        scenario_idx: int,
        split: str,
        seed: int,
        filler_count: int = 6,
    ) -> Scenario:
        pool = self._get_pool("Q", split)
        pool_item = pool[scenario_idx % len(pool)]
        user_name = f"User_Q_{split}_{scenario_idx:02d}"

        base_date = datetime(2024, 3, 1, 9, 0, 0, tzinfo=timezone.utc)
        q_date_dt = base_date + timedelta(days=20)

        evidence_sessions = [
            {
                "session_id": "s_context_ambiguity",
                "session_date": format_date(base_date),
                "turns": [
                    {"role": "user", "content": pool_item["context_statement"]},
                    {"role": "assistant", "content": "Understood, noted."},
                ],
                "is_evidence": True,
            }
        ]

        all_sessions = self._interleave_fillers(
            evidence_sessions, q_date_dt, "Q", scenario_idx, seed, filler_count
        )

        q1 = ScenarioQuestion(
            question_id=f"Q_{split}_{scenario_idx:02d}_q1",
            question_text=pool_item["ambiguous_query"],
            question_subkind="clarification",
            gold_answer="ASK_CLARIFICATION",
            rubric=pool_item["rubric"],
            scoring_type="llm_rubric",
            metadata={
                "required_candidates": [list(g) for g in pool_item["required_candidates"]],
                "ambiguity_type": pool_item["type"],
            },
        )

        return Scenario(
            scenario_id=f"Q_{split}_{scenario_idx:02d}",
            track="Q",
            split=split,
            user_name=user_name,
            question_date=format_date(q_date_dt),
            sessions=all_sessions,
            questions=[q1],
            metadata={
                "template_id": pool_item["id"],
                "domain": pool_item["domain"],
                "type": pool_item["type"],
            },
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
            sc = self.generate_track_r_scenario(scenario_idx, split, seed, filler_count)
        elif track == "D":
            sc = self.generate_track_d_scenario(scenario_idx, split, seed, filler_count)
        elif track == "X":
            sc = self.generate_track_x_scenario(scenario_idx, split, seed, filler_count)
        elif track == "F":
            sc = self.generate_track_f_scenario(scenario_idx, split, seed, filler_count)
        elif track == "I":
            sc = self.generate_track_i_scenario(scenario_idx, split, seed, filler_count)
        elif track == "P":
            sc = self.generate_track_p_scenario(scenario_idx, split, seed, filler_count)
        elif track == "K":
            sc = self.generate_track_k_scenario(scenario_idx, split, seed, filler_count)
        elif track == "E":
            sc = self.generate_track_e_scenario(scenario_idx, split, seed, filler_count)
        elif track == "Q":
            sc = self.generate_track_q_scenario(scenario_idx, split, seed, filler_count)
        else:
            raise ValueError(f"Unknown track '{track}'. Valid tracks: {', '.join(VALID_TRACKS)}")
        sc.metadata.setdefault("distractor_mode", self.distractor_mode)
        if self.epistemic_stress:
            sc.metadata.setdefault("epistemic_stress", True)
        return sc
