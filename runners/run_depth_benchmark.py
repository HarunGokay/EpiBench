"""EpiBench Depth Benchmark Runner.

Measures memory architecture performance under increasing haystack depth:
- Filler levels: 6, 24, and 60 conversational filler sessions
- Tracks: R (Revision), D (Dependency), X (Hidden Constraints)
- Systems: Mem0, Hybrid RAG, Naive Vector RAG
- Unit of Analysis: Scenario

Records accuracy, retrieval latency, and context token inflation across scale.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath("."))

from EpiBench.generator.scenario_generator import ScenarioGenerator
from EpiBench.runners.metering import BUDGET_HARD_CAP_USD, SpendTracker
from EpiBench.runners.run_epibench import EpiBenchRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("epibench_depth")


async def run_depth_suite(
    api_key: str,
    depths: List[int] = [6, 24, 60],
    tracks: List[str] = ["R", "D", "X"],
    systems: List[str] = ["vector_rag", "hybrid_rag", "mem0"],
    scenarios_per_track: int = 2,
    seed: int = 20261122,
    output_dir: Path = Path("EpiBench/results/depth"),
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    spend_tracker = SpendTracker(hard_cap_usd=BUDGET_HARD_CAP_USD)

    depth_master_results: Dict[str, Any] = {
        "benchmark": "EpiBench Depth Stress Test",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": "3fcd4ef",
        "depths": depths,
        "tracks": tracks,
        "systems": systems,
        "scenarios_per_track": scenarios_per_track,
        "results_by_depth": {},
        "comparison_matrix": {},
    }

    t0_suite = time.perf_counter()

    for depth in depths:
        logger.info(f"\n=======================================================")
        logger.info(f"STARTING DEPTH STRESS TEST: FILLER COUNT = {depth}")
        logger.info(f"=======================================================")

        depth_dir = output_dir / f"depth_{depth}"
        depth_dir.mkdir(parents=True, exist_ok=True)

        runner = EpiBenchRunner(
            api_key=api_key,
            spend_tracker=spend_tracker,
            token_budget=1500,
            output_dir=depth_dir,
        )

        res = await runner.run_benchmark(
            tracks=tracks,
            systems=systems,
            scenarios_per_track=scenarios_per_track,
            split="dev",
            seed=seed + depth * 10,
            filler_count=depth,
        )

        depth_master_results["results_by_depth"][str(depth)] = {
            "performance_summary": res["performance_summary"],
            "spend_summary": spend_tracker.summary(),
        }

    # Build Comparative Matrix across Depths
    matrix: Dict[str, Dict[str, Any]] = {}
    for sys_name in systems:
        matrix[sys_name] = {
            "question_accuracy_by_depth": {},
            "scenario_pass_rate_by_depth": {},
            "avg_latency_by_depth": {},
            "avg_tokens_by_depth": {},
        }
        for depth in depths:
            d_str = str(depth)
            d_summary = depth_master_results["results_by_depth"][d_str]["performance_summary"]["by_system"].get(sys_name, {})
            matrix[sys_name]["question_accuracy_by_depth"][d_str] = d_summary.get("question_accuracy", 0.0)
            matrix[sys_name]["scenario_pass_rate_by_depth"][d_str] = d_summary.get("scenario_pass_rate", 0.0)
            matrix[sys_name]["avg_latency_by_depth"][d_str] = d_summary.get("avg_retrieval_latency_ms", 0.0)
            matrix[sys_name]["avg_tokens_by_depth"][d_str] = d_summary.get("avg_context_tokens", 0.0)

    depth_master_results["comparison_matrix"] = matrix
    depth_master_results["total_spend_summary"] = spend_tracker.summary()
    depth_master_results["total_wall_time_s"] = round(time.perf_counter() - t0_suite, 2)

    master_path = output_dir / "results_depth.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(depth_master_results, f, indent=2)

    logger.info(f"\nDepth stress test successfully completed! Results written to {master_path}")
    return depth_master_results


async def main():
    parser = argparse.ArgumentParser(description="Run EpiBench Depth Stress Test")
    parser.add_argument("--depths", type=str, default="6,24,60")
    parser.add_argument("--tracks", type=str, default="R,D,X")
    parser.add_argument("--systems", type=str, default="vector_rag,hybrid_rag,mem0")
    parser.add_argument("--scenarios-per-track", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--output-dir", type=str, default="EpiBench/results/depth")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required.")

    depth_list = [int(d.strip()) for d in args.depths.split(",") if d.strip()]
    track_list = [t.strip().upper() for t in args.tracks.split(",") if t.strip()]
    system_list = [s.strip().lower() for s in args.systems.split(",") if s.strip()]

    await run_depth_suite(
        api_key=api_key,
        depths=depth_list,
        tracks=track_list,
        systems=system_list,
        scenarios_per_track=args.scenarios_per_track,
        seed=args.seed,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    asyncio.run(main())
