"""EpiBench Depth Stress Benchmark Runner (SCORER_VERSION 2.0.0).

Measures memory architecture performance under increasing haystack depth:
- Default filler depths: 6, 24, and 60 conversational filler sessions
- Default sample size: 30 scenarios per track per depth (configurable)
- Default systems: null_memory, oracle, recent_context, full_context_unbounded,
  vector_rag, hybrid_rag
- Reports question accuracy & scenario pass rates with 95% Wilson confidence
  intervals, pairwise comparisons, retrieval latency, and context tokens.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Sequence

from epibench.adapters.base import BaseMemoryAdapter
from epibench.generator.scenario_generator import GENERATOR_VERSION
from epibench.judging.programmatic import SCORER_VERSION
from epibench.judging.rubric_judge import DEFAULT_JUDGE_MODEL
from epibench.runners.metering import SpendTracker
from epibench.runners.reader import READER_PROMPT_VERSION
from epibench.runners.run_epibench import EpiBenchRunner
from epibench.utils import get_git_commit_sha

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("epibench_depth")

DEFAULT_DEPTH_SYSTEMS = (
    "null_memory",
    "oracle",
    "recent_context",
    "full_context_unbounded",
    "vector_rag",
    "hybrid_rag",
)


async def run_depth_suite(
    api_key: str,
    budget_usd: float,
    depths: Sequence[int] = (6, 24, 60),
    tracks: Sequence[str] = ("R", "D", "X"),
    systems: Sequence[str] = DEFAULT_DEPTH_SYSTEMS,
    scenarios_per_track: int = 30,
    seed: int = 20261122,
    token_budget: int = 1500,
    output_dir: Path = Path("results/depth"),
    reader_model: str = "gemini-3.5-flash-lite",
    judge_model: str = DEFAULT_JUDGE_MODEL,
    use_fakes: bool = False,
    custom_adapters: Optional[Dict[str, BaseMemoryAdapter]] = None,
    concurrency: int = 1,
    distractor_mode: str = "neutral",
    distractor_file: Optional[Path] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spend_tracker = SpendTracker(hard_cap_usd=budget_usd)

    depth_list = list(depths)
    track_list = [t.upper() for t in tracks]
    system_list = [s.lower() for s in systems]

    depth_master_results: Dict[str, Any] = {
        "benchmark": "EpiBench Depth Stress Test",
        "scorer_version": SCORER_VERSION,
        "generator_version": GENERATOR_VERSION,
        "reader_prompt_version": READER_PROMPT_VERSION,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": get_git_commit_sha(),
        "use_fakes": use_fakes,
        "distractor_mode": "custom" if distractor_file is not None else distractor_mode,
        "seed": seed,
        "token_budget": token_budget,
        "depths": depth_list,
        "tracks": track_list,
        "systems": system_list,
        "scenarios_per_track": scenarios_per_track,
        "results_by_depth": {},
        "comparison_matrix": {},
    }

    t0_suite = time.perf_counter()

    for depth in depth_list:
        logger.info("STARTING DEPTH STRESS TEST: FILLER COUNT = %d (mode=%s)", depth, distractor_mode)
        depth_dir = output_dir / f"depth_{depth}"
        depth_dir.mkdir(parents=True, exist_ok=True)

        runner = EpiBenchRunner(
            api_key=api_key,
            spend_tracker=spend_tracker,
            token_budget=token_budget,
            output_dir=depth_dir,
            reader_model=reader_model,
            judge_model=judge_model,
            use_fakes=use_fakes,
            custom_adapters=custom_adapters,
            distractor_mode=distractor_mode,
            distractor_file=distractor_file,
        )

        res = await runner.run_benchmark(
            tracks=track_list,
            systems=system_list,
            scenarios_per_track=scenarios_per_track,
            split="dev",
            seed=seed + depth * 10,
            filler_count=depth,
            resume=not use_fakes,
            save_checkpoints=not use_fakes,
            concurrency=concurrency,
        )

        depth_master_results["results_by_depth"][str(depth)] = {
            "manifest_sha256": res["manifest_sha256"],
            "performance_summary": res["performance_summary"],
            "spend_summary": spend_tracker.summary(),
        }

    # Build comparative matrix across depths with 95% Wilson CIs
    matrix: Dict[str, Dict[str, Any]] = {}
    for sys_name in system_list:
        matrix[sys_name] = {
            "question_accuracy_by_depth": {},
            "question_ci_95_by_depth": {},
            "scenario_pass_rate_by_depth": {},
            "scenario_ci_95_by_depth": {},
            "avg_latency_by_depth": {},
            "avg_tokens_by_depth": {},
        }
        for depth in depth_list:
            d_str = str(depth)
            d_summary = depth_master_results["results_by_depth"][d_str]["performance_summary"]["by_system"].get(sys_name, {})
            matrix[sys_name]["question_accuracy_by_depth"][d_str] = d_summary.get("question_accuracy", 0.0)
            matrix[sys_name]["question_ci_95_by_depth"][d_str] = d_summary.get("question_ci_95", [0.0, 0.0])
            matrix[sys_name]["scenario_pass_rate_by_depth"][d_str] = d_summary.get("scenario_pass_rate", 0.0)
            matrix[sys_name]["scenario_ci_95_by_depth"][d_str] = d_summary.get("scenario_ci_95", [0.0, 0.0])
            matrix[sys_name]["avg_latency_by_depth"][d_str] = d_summary.get("avg_retrieval_latency_ms", 0.0)
            matrix[sys_name]["avg_tokens_by_depth"][d_str] = d_summary.get("avg_context_tokens", 0.0)

    depth_master_results["comparison_matrix"] = matrix
    depth_master_results["total_spend_summary"] = spend_tracker.summary()
    depth_master_results["total_wall_time_s"] = round(time.perf_counter() - t0_suite, 2)

    master_path = output_dir / "results_depth.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(depth_master_results, f, indent=2)

    logger.info("Depth stress test completed. Results written to %s", master_path)
    return depth_master_results


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run EpiBench Depth Stress Test")
    parser.add_argument("--depths", type=str, default="6,24,60")
    parser.add_argument("--tracks", type=str, default="R,D,X")
    parser.add_argument("--systems", type=str, default=",".join(DEFAULT_DEPTH_SYSTEMS))
    parser.add_argument("--scenarios-per-track", type=int, default=30)
    parser.add_argument("--token-budget", type=int, default=1500)
    parser.add_argument("--budget-usd", type=float, required=True, help="Hard spend ceiling in USD (required).")
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--output-dir", type=str, default="results/depth")
    parser.add_argument("--distractor-mode", choices=["neutral", "same_domain", "mixed"], default="neutral")
    parser.add_argument("--distractor-file", type=str, default=None)
    parser.add_argument("--offline", action="store_true", help="Use deterministic offline fakes.")
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not args.offline and not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required unless --offline is specified.")

    depth_list = [int(d.strip()) for d in args.depths.split(",") if d.strip()]
    track_list = [t.strip().upper() for t in args.tracks.split(",") if t.strip()]
    system_list = [s.strip().lower() for s in args.systems.split(",") if s.strip()]

    await run_depth_suite(
        api_key=api_key,
        budget_usd=args.budget_usd,
        depths=depth_list,
        tracks=track_list,
        systems=system_list,
        scenarios_per_track=args.scenarios_per_track,
        seed=args.seed,
        token_budget=args.token_budget,
        output_dir=Path(args.output_dir),
        use_fakes=args.offline,
        concurrency=args.concurrency,
        distractor_mode=args.distractor_mode,
        distractor_file=Path(args.distractor_file) if args.distractor_file else None,
    )


if __name__ == "__main__":
    asyncio.run(main())
