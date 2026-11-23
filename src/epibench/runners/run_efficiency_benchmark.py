"""EpiBench Memory Efficiency & Scale Benchmark Runner (SCORER_VERSION 2.0.0).

Evaluates memory architectures across configurable conversation turn scales:
- Plants deterministic factual needles at evenly spaced positions across the
  conversation timeline (beginning, quartiles, end).
- Measures at every scale:
  1. Planted-fact retrieval recall (%) with 95% Wilson CI
  2. Context token count per query
  3. Retrieval latency (ms)
  4. Ingestion wall time (s)
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from epibench.adapters.base import BaseMemoryAdapter
from epibench.adapters.full_context_adapter import (
    FullContextUnboundedAdapter,
    RecentContextAdapter,
)
from epibench.adapters.hybrid_rag_adapter import HybridRAGAdapter
from epibench.adapters.reference_adapters import NullMemoryAdapter, OracleAdapter
from epibench.adapters.vector_rag_adapter import VectorRAGAdapter
from epibench.generator.scenario_generator import GENERATOR_VERSION, format_date
from epibench.judging.programmatic import SCORER_VERSION, normalize
from epibench.metrics.statistics import wilson_ci
from epibench.runners.fakes import FakeEmbedder
from epibench.utils import get_git_commit_sha

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("epibench_efficiency")

PLANTED_FACTS: List[Dict[str, str]] = [
    {
        "fact_id": "needle_db",
        "statement": "Our primary production analytics warehouse runs on ClickHouse 24.3.",
        "query": "What engine runs our primary production analytics warehouse?",
        "gold": "ClickHouse 24.3",
    },
    {
        "fact_id": "needle_budget",
        "statement": "The finance board approved a Q4 infrastructure tooling cap of $87,500.",
        "query": "What is the approved Q4 infrastructure tooling budget cap?",
        "gold": "$87,500",
    },
    {
        "fact_id": "needle_lead",
        "statement": "Dr. Naomi Alvarez was appointed chair of the security review council.",
        "query": "Who was appointed chair of the security review council?",
        "gold": "Dr. Naomi Alvarez",
    },
    {
        "fact_id": "needle_datacenter",
        "statement": "Our European disaster-recovery failover vault is hosted in Zurich-CH2.",
        "query": "Where is our European disaster-recovery failover vault hosted?",
        "gold": "Zurich-CH2",
    },
    {
        "fact_id": "needle_sla",
        "statement": "Enterprise tier customers receive a guaranteed 99.98% uptime SLA.",
        "query": "What uptime SLA is guaranteed for enterprise tier customers?",
        "gold": "99.98%",
    },
]


def generate_synthetic_history(
    turn_count: int,
    seed: int = 20261122,
    plant_facts: bool = True,
) -> List[Dict[str, Any]]:
    """Generate synthetic chronological sessions with optional evenly spaced planted facts."""
    rng = random.Random(seed)
    topics = ["budget planning", "cloud migrations", "team staffing", "design review", "api contracts"]
    base_dt = datetime(2024, 1, 1, 8, 0, 0, tzinfo=timezone.utc)

    n_total_sessions = max(1, (turn_count + 1) // 2)
    planted = list(PLANTED_FACTS) if plant_facts else []
    n_planted = min(len(planted), n_total_sessions)
    n_filler = n_total_sessions - n_planted
    sessions: List[Dict[str, Any]] = []

    for s_idx in range(n_filler):
        topic = rng.choice(topics)
        dt = base_dt + timedelta(hours=4 * s_idx)
        sessions.append({
            "session_id": f"scale_sess_{s_idx:05d}",
            "session_date": format_date(dt),
            "turns": [
                {"role": "user", "content": f"Status check on {topic}: reviewing milestone {s_idx}."},
                {"role": "assistant", "content": f"Acknowledged. Milestone {s_idx} for {topic} is on schedule."},
            ],
            "is_evidence": False,
        })

    if n_planted > 0:
        for f_idx, fact in enumerate(planted[:n_planted]):
            pos = int((f_idx / max(1, n_planted - 1)) * max(0, n_filler - 1)) if n_filler > 1 else f_idx
            dt = base_dt + timedelta(hours=4 * pos, minutes=30 + f_idx)
            sessions.append({
                "session_id": f"planted_{fact['fact_id']}",
                "session_date": format_date(dt),
                "turns": [
                    {"role": "user", "content": fact["statement"]},
                    {"role": "assistant", "content": "Noted and stored in project memory."},
                ],
                "is_evidence": True,
                "fact_id": fact["fact_id"],
            })

    sessions.sort(key=lambda s: (s["session_date"], s["session_id"]))
    return sessions


async def benchmark_system_at_scale(
    adapter: BaseMemoryAdapter,
    tenant_id: str,
    sessions: List[Dict[str, Any]],
    planted_facts: Sequence[Dict[str, str]] = PLANTED_FACTS,
    token_budget: int = 1500,
) -> Dict[str, Any]:
    """Ingest history and measure latency, token usage, and planted-fact retrieval recall."""
    t0_ingest = time.perf_counter()
    await adapter.ingest(tenant_id, sessions)
    ingest_time_s = time.perf_counter() - t0_ingest

    retrieval_latencies: List[float] = []
    tokens_used: List[int] = []
    facts_retrieved = 0

    for fact in planted_facts:
        t0_ret = time.perf_counter()
        ret = await adapter.retrieve(tenant_id=tenant_id, query=fact["query"], token_budget=token_budget)
        retrieval_latencies.append((time.perf_counter() - t0_ret) * 1000.0)
        tokens_used.append(ret.token_count)
        if normalize(fact["gold"]) in normalize(ret.context):
            facts_retrieved += 1

    total_facts = len(planted_facts)
    recall_pct = (facts_retrieved / max(1, total_facts)) * 100.0
    ci = wilson_ci(facts_retrieved, total_facts)
    avg_ret_ms = sum(retrieval_latencies) / max(1, len(retrieval_latencies))
    avg_tok = sum(tokens_used) / max(1, len(tokens_used))

    return {
        "turn_count": sum(len(s.get("turns", [])) for s in sessions),
        "session_count": len(sessions),
        "planted_facts_retrieved": facts_retrieved,
        "planted_facts_total": total_facts,
        "fact_retrieval_recall_pct": round(recall_pct, 2),
        "fact_retrieval_ci_95": [round(ci["lower"] * 100.0, 2), round(ci["upper"] * 100.0, 2)],
        "ingest_time_s": round(ingest_time_s, 4),
        "avg_retrieval_latency_ms": round(avg_ret_ms, 2),
        "avg_context_tokens": round(avg_tok, 1),
    }


def _make_efficiency_adapter(sys_name: str, use_fake_embedder: bool = False) -> BaseMemoryAdapter:
    embedder = FakeEmbedder() if use_fake_embedder else None
    key = sys_name.lower().strip()
    if key == "vector_rag":
        return VectorRAGAdapter(embedder=embedder)
    if key == "hybrid_rag":
        return HybridRAGAdapter(embedder=embedder)
    if key in ("recent_context", "full_context"):
        return RecentContextAdapter(system_name=key)
    if key == "full_context_unbounded":
        return FullContextUnboundedAdapter()
    if key == "oracle":
        return OracleAdapter()
    if key == "null_memory":
        return NullMemoryAdapter()
    raise ValueError(f"Unsupported efficiency adapter: {sys_name!r}")


async def run_efficiency_suite(
    scales: Sequence[int] = (100, 1000, 5000),
    systems: Sequence[str] = ("null_memory", "oracle", "recent_context", "vector_rag", "hybrid_rag"),
    seed: int = 20261122,
    token_budget: int = 1500,
    output_path: Optional[Path] = Path("results/efficiency_results.json"),
    use_fake_embedder: bool = False,
) -> Dict[str, Any]:
    """Run scaling efficiency and planted-fact retention benchmark across turn counts."""
    results: Dict[str, Any] = {
        "benchmark": "EpiBench Memory Efficiency & Scale Benchmark",
        "scorer_version": SCORER_VERSION,
        "generator_version": GENERATOR_VERSION,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": get_git_commit_sha(),
        "seed": seed,
        "token_budget": token_budget,
        "use_fake_embedder": use_fake_embedder,
        "scales": list(scales),
        "systems": list(systems),
        "by_scale": {},
    }

    for scale in scales:
        history = generate_synthetic_history(scale, seed=seed + scale, plant_facts=True)
        results["by_scale"][str(scale)] = {}
        for sys_name in systems:
            adapter = _make_efficiency_adapter(sys_name, use_fake_embedder=use_fake_embedder)
            tenant_id = f"scale_{scale}_{sys_name}"
            res = await benchmark_system_at_scale(
                adapter=adapter,
                tenant_id=tenant_id,
                sessions=history,
                planted_facts=PLANTED_FACTS,
                token_budget=token_budget,
            )
            results["by_scale"][str(scale)][sys_name] = res
            await adapter.cleanup(tenant_id)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="EpiBench Memory Efficiency & Scale Benchmark")
    parser.add_argument("--scales", type=str, default="100,1000,5000", help="Comma-separated turn counts")
    parser.add_argument(
        "--systems",
        type=str,
        default="null_memory,oracle,recent_context,vector_rag,hybrid_rag",
    )
    parser.add_argument("--token-budget", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--output", type=str, default="results/efficiency_results.json")
    parser.add_argument("--offline", action="store_true", help="Use fast deterministic FakeEmbedder.")
    args = parser.parse_args()

    scale_list = [int(s.strip()) for s in args.scales.split(",") if s.strip()]
    system_names = [s.strip().lower() for s in args.systems.split(",") if s.strip()]

    results = asyncio.run(
        run_efficiency_suite(
            scales=scale_list,
            systems=system_names,
            seed=args.seed,
            token_budget=args.token_budget,
            output_path=Path(args.output),
            use_fake_embedder=args.offline,
        )
    )

    print("\n=== Memory Efficiency & Scale Summary ===")
    print(f"{'Scale':<10}{'System':<22}{'Recall %':<12}{'Ingest (s)':<12}{'Ret (ms)':<12}{'Avg Tokens':<12}")
    for sc, sys_dict in results["by_scale"].items():
        for sname, r in sys_dict.items():
            print(
                f"{sc:<10}{sname:<22}{r['fact_retrieval_recall_pct']:<12.1f}"
                f"{r['ingest_time_s']:<12.4f}{r['avg_retrieval_latency_ms']:<12.2f}{r['avg_context_tokens']:<12.1f}"
            )


if __name__ == "__main__":
    main()
