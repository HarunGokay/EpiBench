"""EpiBench Mandatory Validity Gate (`epibench validate`).

Runs `null_memory` (lower reference bound) and `oracle` (upper reference bound)
through the full benchmark pipeline on every track.

Mandatory acceptance criteria per track:
- oracle effective score >= 95.0%
- null_memory effective score <= 15.0%
(Track E uses joint scenario pass rate: control_retained AND leak_free.)

Any track violating these thresholds is marked `experimental` and excluded
from composite scores with an explicit documented reason.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from epibench.generator.pools import validate_pools
from epibench.generator.scenario_generator import (
    GENERATOR_VERSION,
    VALID_TRACKS,
    resolve_profile,
)
from epibench.judging.programmatic import SCORER_VERSION
from epibench.judging.rubric_judge import DEFAULT_JUDGE_MODEL
from epibench.runners.metering import SpendTracker
from epibench.runners.reader import READER_PROMPT_VERSION
from epibench.runners.run_epibench import EpiBenchRunner
from epibench.utils import get_git_commit_sha

ORACLE_MIN_THRESHOLD_PCT = 95.0
NULL_MAX_THRESHOLD_PCT = 15.0


async def run_validity_gate(
    tracks: Optional[Sequence[str]] = None,
    scenarios_per_track: int = 12,
    filler_count: Optional[int] = None,
    seed: int = 20261122,
    budget_usd: float = 2.0,
    api_key: str = "",
    offline: bool = False,
    output_dir: Path = Path("results/validity"),
    reader_model: str = "gemini-3.5-flash-lite",
    judge_model: str = DEFAULT_JUDGE_MODEL,
    profile: str = "dev",
    distractor_mode: Optional[str] = None,
    distractor_file: Optional[Path] = None,
    report_filename: Optional[str] = None,
    concurrency: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute the mandatory null_memory vs oracle validity gate across all tracks."""
    validate_pools()
    resolved = resolve_profile(
        profile=profile,
        filler_count=filler_count,
        distractor_mode=distractor_mode,
    )
    eff_fillers = resolved["filler_count"]
    eff_mode = "custom" if distractor_file is not None else resolved["distractor_mode"]
    track_list = [t.upper() for t in (tracks or VALID_TRACKS)]

    if not offline and not api_key:
        raise ValueError("GEMINI_API_KEY is required for live validation (or pass offline=True).")

    spend_tracker = SpendTracker(hard_cap_usd=budget_usd)
    runner = EpiBenchRunner(
        api_key=api_key,
        spend_tracker=spend_tracker,
        token_budget=resolved["token_budget"],
        output_dir=output_dir,
        reader_model=reader_model,
        judge_model=judge_model,
        secondary_audit_rate=0.0,
        use_fakes=offline,
        strict_as_of_audit=True,
        distractor_mode=resolved["distractor_mode"],
        distractor_file=distractor_file,
        profile=resolved["profile"],
    )

    raw_name = (
        f"validity_{resolved['profile']}_raw.json"
        if resolved["profile"] != "dev"
        else "validity_raw.json"
    )
    run_res = await runner.run_benchmark(
        tracks=track_list,
        systems=["null_memory", "oracle"],
        scenarios_per_track=scenarios_per_track,
        split="dev",
        seed=seed,
        filler_count=eff_fillers,
        resume=not offline,
        save_checkpoints=not offline,
        output_filename=raw_name,
        concurrency=(1 if offline else 6) if concurrency is None else concurrency,
    )

    by_track = run_res["performance_summary"]["by_track"]
    track_verdicts: Dict[str, Dict[str, Any]] = {}
    experimental_tracks: List[str] = []
    passing_tracks: List[str] = []

    for track in track_list:
        null_stats = by_track[track]["null_memory"]
        oracle_stats = by_track[track]["oracle"]
        null_score = float(null_stats["effective_score"])
        oracle_score = float(oracle_stats["effective_score"])

        reasons: List[str] = []
        if oracle_score < ORACLE_MIN_THRESHOLD_PCT:
            reasons.append(f"oracle={oracle_score:.2f}% < {ORACLE_MIN_THRESHOLD_PCT:.1f}%")
        if null_score > NULL_MAX_THRESHOLD_PCT:
            reasons.append(f"null_memory={null_score:.2f}% > {NULL_MAX_THRESHOLD_PCT:.1f}%")

        passed = len(reasons) == 0
        if passed:
            passing_tracks.append(track)
        else:
            experimental_tracks.append(track)

        track_verdicts[track] = {
            "status": "PASS" if passed else "EXPERIMENTAL",
            "oracle_score_pct": oracle_score,
            "null_memory_score_pct": null_score,
            "oracle_question_accuracy_pct": oracle_stats["question_accuracy"],
            "null_question_accuracy_pct": null_stats["question_accuracy"],
            "oracle_scenario_pass_pct": oracle_stats["scenario_pass_rate"],
            "null_scenario_pass_pct": null_stats["scenario_pass_rate"],
            "scoring_metric": "joint_scenario_pass_rate" if track == "E" else "question_accuracy",
            "reasons": reasons,
        }

    report: Dict[str, Any] = {
        "gate": "epibench_validity_gate",
        "scorer_version": SCORER_VERSION,
        "generator_version": GENERATOR_VERSION,
        "reader_prompt_version": READER_PROMPT_VERSION,
        "git_commit_sha": get_git_commit_sha(),
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "live_verified": not offline,
        "status_note": "live verified with LLM API" if not offline else "validity unverified live (offline fakes)",
        "reader_model": runner.reader_model,
        "judge_model": runner.judge_model,
        "seed": seed,
        "profile": resolved["profile"],
        "distractor_mode": eff_mode,
        "manifest_sha256": run_res["manifest_sha256"],
        "scenarios_per_track": scenarios_per_track,
        "filler_count": eff_fillers,
        "thresholds": {
            "oracle_min_pct": ORACLE_MIN_THRESHOLD_PCT,
            "null_memory_max_pct": NULL_MAX_THRESHOLD_PCT,
        },
        "all_tracks_passed": len(experimental_tracks) == 0,
        "passing_tracks": passing_tracks,
        "experimental_tracks": experimental_tracks,
        "tracks": track_verdicts,
        "spend_summary": spend_tracker.summary(),
    }

    out_name = report_filename or (
        f"validity_{resolved['profile']}_report.json"
        if resolved["profile"] != "dev"
        else "validity_report.json"
    )
    report_path = output_dir / out_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    return report


def format_validity_table(report: Dict[str, Any]) -> str:
    """Render a concise human-readable validity gate table."""
    lines = [
        f"EpiBench Validity Gate ({report['status_note']})",
        f"Reader: {report['reader_model']} | Judge: {report['judge_model']} | N={report['scenarios_per_track']}/track",
        f"{'Track':<7}{'Status':<14}{'Oracle %':<12}{'Null %':<12}{'Metric':<26}{'Notes'}",
        "-" * 85,
    ]
    for track, info in report["tracks"].items():
        note = "; ".join(info["reasons"]) if info["reasons"] else "OK"
        lines.append(
            f"{track:<7}{info['status']:<14}{info['oracle_score_pct']:<12.2f}"
            f"{info['null_memory_score_pct']:<12.2f}{info['scoring_metric']:<26}{note}"
        )
    lines.append("-" * 85)
    lines.append(
        f"Passing: {report['passing_tracks']} | Experimental: {report['experimental_tracks']} | "
        f"Spend: ${report['spend_summary']['total_spend_usd']:.4f}"
    )
    return "\n".join(lines)
