"""EpiBench Seven-Dimensional Epistemic Evaluation Metrics (SCORER_VERSION 2.0.0).

Computes seven independent epistemic dimension scores, renormalized composite
score, separate Track E control-retention diagnostic, and efficiency metrics:
1. factual_accuracy: Track K (static recall control)
2. temporal_consistency: Tracks R & F (revision, point-in-time, slippage)
3. dependency_reasoning: Track D (cascade set, rationale, still-standing)
4. constraint_compliance: Track X (unprompted hidden constraints)
5. provenance: Track P (rejecting hallucinations and retractions)
6. calibration: Tracks I & Q (causal calibration and active clarification)
7. privacy_and_deletion: Track E joint outcome (control retained AND leak-free)

Diagnostic metric (excluded from privacy_and_deletion and composite):
- retention_control: Track E `deletion_control_retention` questions

Rules:
- Dimensions with 0 evaluated items score `None` (JSON `null`) and are excluded
  from the composite; weights are renormalized over `included_dimensions`.
- Tracks in `experimental_tracks` are excluded from the composite.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

DIMENSIONS: tuple[str, ...] = (
    "factual_accuracy",
    "temporal_consistency",
    "dependency_reasoning",
    "constraint_compliance",
    "provenance",
    "calibration",
    "privacy_and_deletion",
)

# Default weights summing to 1.0 across the 7 epistemic dimensions
DEFAULT_DIMENSION_WEIGHTS: Dict[str, float] = {
    "factual_accuracy": 0.15,
    "temporal_consistency": 0.20,
    "dependency_reasoning": 0.15,
    "constraint_compliance": 0.15,
    "provenance": 0.125,
    "calibration": 0.125,
    "privacy_and_deletion": 0.10,
}


@dataclass
class DimensionalScore:
    dimension_name: str
    score_pct: Optional[float]
    correct_count: int
    total_count: int
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemEvaluationReport:
    system_name: str
    factual_accuracy: Optional[float]
    temporal_consistency: Optional[float]
    dependency_reasoning: Optional[float]
    constraint_compliance: Optional[float]
    provenance: Optional[float]
    calibration: Optional[float]
    privacy_and_deletion: Optional[float]
    retention_control: Optional[float]
    composite_score: Optional[float]
    included_dimensions: List[str]
    excluded_dimensions: List[str]
    avg_tokens_per_query: float
    avg_retrieval_latency_ms: float
    total_spend_usd: float
    cost_per_successful_answer_usd: Optional[float]
    dimension_breakdown: Dict[str, DimensionalScore] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "system_name": self.system_name,
            "factual_accuracy": self.factual_accuracy,
            "temporal_consistency": self.temporal_consistency,
            "dependency_reasoning": self.dependency_reasoning,
            "constraint_compliance": self.constraint_compliance,
            "provenance": self.provenance,
            "calibration": self.calibration,
            "privacy_and_deletion": self.privacy_and_deletion,
            "retention_control": self.retention_control,
            "composite_score": self.composite_score,
            "included_dimensions": list(self.included_dimensions),
            "excluded_dimensions": list(self.excluded_dimensions),
            "avg_tokens_per_query": self.avg_tokens_per_query,
            "avg_retrieval_latency_ms": self.avg_retrieval_latency_ms,
            "total_spend_usd": self.total_spend_usd,
            "cost_per_successful_answer_usd": self.cost_per_successful_answer_usd,
            "dimension_breakdown": {k: asdict(v) for k, v in self.dimension_breakdown.items()},
        }


def map_evaluation_to_dimension(track: str, q_subkind: str) -> Optional[str]:
    """Map an evaluation item's (track, question_subkind) to its exact dimension."""
    tr = track.upper().strip()
    sub = q_subkind.strip().lower()

    # Track E control retention belongs ONLY to the separate retention_control metric
    if sub == "deletion_control_retention":
        return "retention_control"
    if sub in ("deletion_joint", "deletion_direct", "deletion_semantic"):
        return "privacy_and_deletion"
    if tr == "E":
        return "privacy_and_deletion"
    if tr == "X" or sub == "hidden_constraint":
        return "constraint_compliance"
    if tr == "K" or sub in ("static_fact", "control"):
        return "factual_accuracy"
    if tr in ("R", "F") or sub in ("as_of", "previous", "current", "count", "move_count", "slip_days", "realism"):
        return "temporal_consistency"
    if tr == "D" or sub in ("revisit_set", "why", "stands", "cascade"):
        return "dependency_reasoning"
    if tr == "P" or sub in ("provenance", "authority"):
        return "provenance"
    if tr in ("I", "Q") or sub in ("causal_claim", "clarification"):
        return "calibration"
    return None


def compute_dimensional_report(
    evaluations: List[Dict[str, Any]],
    system_name: str,
    weights: Optional[Dict[str, float]] = None,
    total_spend_usd: float = 0.0,
    experimental_tracks: Optional[Sequence[str]] = None,
) -> SystemEvaluationReport:
    """Compute seven dimension scores, retention_control, and renormalized composite."""
    w = weights or DEFAULT_DIMENSION_WEIGHTS
    exp_set: Set[str] = {t.upper() for t in (experimental_tracks or [])}

    dim_counts: Dict[str, Dict[str, int]] = {
        d: {"correct": 0, "total": 0} for d in DIMENSIONS
    }
    retention_counts = {"correct": 0, "total": 0}

    latencies: List[float] = []
    tokens: List[int] = []
    total_correct_questions = 0

    for ev in evaluations:
        track = str(ev.get("track", "")).upper()
        q_subkind = str(ev.get("question_subkind", ""))
        passed = bool(ev.get("is_correct", False))

        if q_subkind != "deletion_joint":
            latencies.append(float(ev.get("retrieval_latency_ms", 0.0)))
            tokens.append(int(ev.get("context_tokens", 0)))
            if passed:
                total_correct_questions += 1

        if track in exp_set:
            continue

        target_dim = map_evaluation_to_dimension(track, q_subkind)
        if target_dim is None:
            continue

        if target_dim == "retention_control":
            retention_counts["total"] += 1
            if passed:
                retention_counts["correct"] += 1
        else:
            dim_counts[target_dim]["total"] += 1
            if passed:
                dim_counts[target_dim]["correct"] += 1

    breakdown: Dict[str, DimensionalScore] = {}
    pcts: Dict[str, Optional[float]] = {}
    included_dims: List[str] = []
    excluded_dims: List[str] = []

    for d_name in DIMENSIONS:
        cnts = dim_counts[d_name]
        if cnts["total"] > 0:
            score: Optional[float] = round((cnts["correct"] / cnts["total"]) * 100.0, 2)
            included_dims.append(d_name)
        else:
            score = None
            excluded_dims.append(d_name)
        pcts[d_name] = score
        breakdown[d_name] = DimensionalScore(
            dimension_name=d_name,
            score_pct=score,
            correct_count=cnts["correct"],
            total_count=cnts["total"],
        )

    ret_score: Optional[float] = (
        round((retention_counts["correct"] / retention_counts["total"]) * 100.0, 2)
        if retention_counts["total"] > 0
        else None
    )
    breakdown["retention_control"] = DimensionalScore(
        dimension_name="retention_control",
        score_pct=ret_score,
        correct_count=retention_counts["correct"],
        total_count=retention_counts["total"],
    )

    # Renormalize weights strictly over included (non-null) dimensions
    if included_dims:
        active_weight_sum = sum(w.get(d, 1.0 / len(DIMENSIONS)) for d in included_dims)
        composite: Optional[float] = round(
            sum(
                (pcts[d] or 0.0) * (w.get(d, 1.0 / len(DIMENSIONS)) / active_weight_sum)
                for d in included_dims
            ),
            2,
        )
    else:
        composite = None

    avg_lat = sum(latencies) / len(latencies) if latencies else 0.0
    avg_tok = sum(tokens) / len(tokens) if tokens else 0.0
    cost_per_answer: Optional[float] = (
        round(total_spend_usd / total_correct_questions, 6)
        if total_correct_questions > 0
        else None
    )

    return SystemEvaluationReport(
        system_name=system_name,
        factual_accuracy=pcts["factual_accuracy"],
        temporal_consistency=pcts["temporal_consistency"],
        dependency_reasoning=pcts["dependency_reasoning"],
        constraint_compliance=pcts["constraint_compliance"],
        provenance=pcts["provenance"],
        calibration=pcts["calibration"],
        privacy_and_deletion=pcts["privacy_and_deletion"],
        retention_control=ret_score,
        composite_score=composite,
        included_dimensions=included_dims,
        excluded_dimensions=excluded_dims,
        avg_tokens_per_query=round(avg_tok, 1),
        avg_retrieval_latency_ms=round(avg_lat, 2),
        total_spend_usd=round(float(total_spend_usd), 6),
        cost_per_successful_answer_usd=cost_per_answer,
        dimension_breakdown=breakdown,
    )


# Backwards-compatible alias
compute_six_dimensional_report = compute_dimensional_report
