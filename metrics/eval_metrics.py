"""EpiBench Six-Dimensional Evaluation Metrics.

Computes six independent epistemic dimension scores, composite quality score,
and efficiency/resource metrics:
1. Factual Accuracy: Exact recall of established user facts.
2. Temporal Consistency: Point-in-time validity, supersession, and timeline order.
3. Dependency Reasoning: Downstream cascade identification upon assumption revisions.
4. Provenance: Source attribution and rejection of assistant hallucinations.
5. Calibration: Epistemic humility, asking clarification, avoiding ungrounded claims.
6. Privacy & Deletion: Verifiable forgetting and absence of deleted information.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DimensionalScore:
    dimension_name: str
    score_pct: float
    correct_count: int
    total_count: int
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemEvaluationReport:
    system_name: str
    factual_accuracy: float
    temporal_consistency: float
    dependency_reasoning: float
    provenance: float
    calibration: float
    privacy_and_deletion: float
    composite_score: float
    avg_tokens_per_query: float
    avg_retrieval_latency_ms: float
    cost_per_successful_answer_usd: float
    dimension_breakdown: Dict[str, DimensionalScore] = field(default_factory=dict)


# Default balanced weights for the composite world-model understanding score
DEFAULT_DIMENSION_WEIGHTS: Dict[str, float] = {
    "factual_accuracy": 0.20,
    "temporal_consistency": 0.20,
    "dependency_reasoning": 0.20,
    "provenance": 0.15,
    "calibration": 0.15,
    "privacy_and_deletion": 0.10,
}


def compute_six_dimensional_report(
    evaluations: List[Dict[str, Any]],
    system_name: str,
    weights: Optional[Dict[str, float]] = None,
    total_spend_usd: float = 0.0,
) -> SystemEvaluationReport:
    """Compute six independent dimension scores and composite score from scenario questions."""
    w = weights or DEFAULT_DIMENSION_WEIGHTS
    dim_counts = {
        "factual_accuracy": {"correct": 0, "total": 0},
        "temporal_consistency": {"correct": 0, "total": 0},
        "dependency_reasoning": {"correct": 0, "total": 0},
        "provenance": {"correct": 0, "total": 0},
        "calibration": {"correct": 0, "total": 0},
        "privacy_and_deletion": {"correct": 0, "total": 0},
    }

    latencies = []
    tokens = []

    for ev in evaluations:
        track = ev.get("track", "").upper()
        q_subkind = ev.get("question_subkind", "")
        passed = bool(ev.get("is_correct", False))
        lat = ev.get("retrieval_latency_ms", 0.0)
        tok = ev.get("context_tokens", 0)

        latencies.append(lat)
        tokens.append(tok)

        # Dimension mapping
        if track == "K" or q_subkind in ("static_fact", "control"):
            target_dim = "factual_accuracy"
        elif track in ("R", "F") or q_subkind in ("as_of", "previous", "current", "superseded", "slip"):
            target_dim = "temporal_consistency"
        elif track == "D" or q_subkind in ("revisit_set", "why", "cascade"):
            target_dim = "dependency_reasoning"
        elif track == "P" or q_subkind in ("provenance", "authority"):
            target_dim = "provenance"
        elif track in ("I", "Q") or q_subkind in ("causal_claim", "clarification"):
            target_dim = "calibration"
        elif track == "E" or q_subkind in ("deletion_direct", "deletion_semantic"):
            target_dim = "privacy_and_deletion"
        else:
            target_dim = "factual_accuracy"

        dim_counts[target_dim]["total"] += 1
        if passed:
            dim_counts[target_dim]["correct"] += 1

    breakdown = {}
    pcts = {}
    for d_name, cnts in dim_counts.items():
        score = (cnts["correct"] / max(1, cnts["total"])) * 100.0 if cnts["total"] > 0 else 100.0
        pcts[d_name] = round(score, 2)
        breakdown[d_name] = DimensionalScore(
            dimension_name=d_name,
            score_pct=round(score, 2),
            correct_count=cnts["correct"],
            total_count=cnts["total"],
        )

    # Composite weighted score
    composite = sum(pcts[d] * w.get(d, 1.0 / len(w)) for d in dim_counts.keys())

    total_correct = sum(c["correct"] for c in dim_counts.values())
    avg_lat = sum(latencies) / max(1, len(latencies))
    avg_tok = sum(tokens) / max(1, len(tokens))
    cost_per_answer = (total_spend_usd / max(1, total_correct)) if total_correct > 0 else 0.0

    return SystemEvaluationReport(
        system_name=system_name,
        factual_accuracy=pcts["factual_accuracy"],
        temporal_consistency=pcts["temporal_consistency"],
        dependency_reasoning=pcts["dependency_reasoning"],
        provenance=pcts["provenance"],
        calibration=pcts["calibration"],
        privacy_and_deletion=pcts["privacy_and_deletion"],
        composite_score=round(composite, 2),
        avg_tokens_per_query=round(avg_tok, 1),
        avg_retrieval_latency_ms=round(avg_lat, 2),
        cost_per_successful_answer_usd=round(cost_per_answer, 5),
        dimension_breakdown=breakdown,
    )
