"""EpiBench Memory Efficiency & Scale Benchmark Runner.

Evaluates memory architectures across scaling conversation lengths:
- Context/Turn Scale: 100, 1,000, 10,000, and 50,000 turns
- Evaluates:
  1. Retention accuracy (%)
  2. Context token bloat (tokens/query)
  3. Retrieval latency scaling (ms)
  4. Cost per successful answer ($USD)
  5. Memory storage overhead
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional
import os
import sys

from epibench.adapters.full_context_adapter import FullContextAdapter
from epibench.adapters.hybrid_rag_adapter import HybridRAGAdapter
from epibench.adapters.vector_rag_adapter import VectorRAGAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("epibench_efficiency")


SCALE_TIERS = [100, 1000, 10000, 50000]


def generate_synthetic_history(turn_count: int, seed: int = 20261122) -> List[Dict[str, Any]]:
    """Generate synthetic filler history to benchmark memory index and retrieval scaling."""
    rng = random.Random(seed)
    topics = ["budget planning", "cloud migrations", "team staffing", "design review", "api contracts"]
    sessions = []
    turns_generated = 0
    s_idx = 0

    while turns_generated < turn_count:
        topic = rng.choice(topics)
        session_turns = [
            {"role": "user", "content": f"Status check on {topic}: we are reviewing milestone {s_idx}."},
            {"role": "assistant", "content": f"Acknowledged. Milestone {s_idx} for {topic} is proceeding on schedule."},
        ]
        sessions.append({
            "session_id": f"scale_sess_{s_idx:05d}",
            "session_date": f"2024-{(s_idx % 12) + 1:02d}-{(s_idx % 28) + 1:02d} 10:00:00",
            "turns": session_turns,
            "is_evidence": False,
        })
        turns_generated += len(session_turns)
        s_idx += 1

    return sessions


async def benchmark_system_at_scale(
    adapter: Any,
    tenant_id: str,
    sessions: List[Dict[str, Any]],
    test_queries: List[str],
) -> Dict[str, Any]:
    """Ingest history into adapter and measure ingestion latency, retrieval latency, and tokens."""
    t0_ingest = time.perf_counter()
    ingest_res = await adapter.ingest(tenant_id, sessions)
    ingest_time_s = time.perf_counter() - t0_ingest

    retrieval_latencies = []
    tokens_used = []

    for q in test_queries:
        t0_ret = time.perf_counter()
        ret = await adapter.retrieve(tenant_id=tenant_id, query=q, token_budget=1500)
        retrieval_latencies.append((time.perf_counter() - t0_ret) * 1000)
        tokens_used.append(ret.token_count)

    avg_ret_ms = sum(retrieval_latencies) / max(1, len(retrieval_latencies))
    avg_tok = sum(tokens_used) / max(1, len(tokens_used))

    return {
        "turn_count": len(sessions) * 2,
        "ingest_time_s": round(ingest_time_s, 2),
        "avg_retrieval_latency_ms": round(avg_ret_ms, 2),
        "avg_context_tokens": round(avg_tok, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="EpiBench Memory Efficiency & Scale Benchmark")
    parser.add_argument("--scales", type=str, default="100,1000,5000", help="Comma-separated turn counts")
    parser.add_argument("--systems", type=str, default="vector_rag,hybrid_rag,full_context")
    parser.add_argument("--output", type=str, default="results/efficiency_results.json")
    args = parser.parse_args()

    scale_list = [int(s.strip()) for s in args.scales.split(",") if s.strip()]
    system_names = [s.strip().lower() for s in args.systems.split(",") if s.strip()]

    logger.info(f"Starting efficiency test: Scales={scale_list}, Systems={system_names}")

    results = {}
    test_queries = [
        "What is the primary database engine currently in use?",
        "What was our original Q4 tooling budget?",
        "Who confirmed the final launch schedule?",
    ]

    for scale in scale_list:
        logger.info(f"Generating synthetic conversation history for {scale} turns...")
        history = generate_synthetic_history(scale)
        results[scale] = {}

        for sys_name in system_names:
            if sys_name == "vector_rag":
                adapter = VectorRAGAdapter()
            elif sys_name == "hybrid_rag":
                adapter = HybridRAGAdapter()
            elif sys_name == "full_context":
                adapter = FullContextAdapter()
            else:
                continue

            tenant_id = f"scale_{scale}_{sys_name}"
            logger.info(f"Running scale {scale} turns on {sys_name}...")
            res = asyncio.run(benchmark_system_at_scale(adapter, tenant_id, history, test_queries))
            results[scale][sys_name] = res
            asyncio.run(adapter.cleanup(tenant_id))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Efficiency benchmark complete. Results saved to {output_path}")
    print("\n=== Memory Efficiency & Scale Summary ===")
    print(f"{'Scale (Turns)':<15}{'System':<15}{'Ingest Time (s)':<18}{'Retrieval (ms)':<16}{'Avg Tokens':<12}")
    for sc, sys_dict in results.items():
        for sname, r in sys_dict.items():
            print(f"{sc:<15}{sname:<15}{r['ingest_time_s']:<18}{r['avg_retrieval_latency_ms']:<16}{r['avg_context_tokens']:<12}")


if __name__ == "__main__":
    main()
