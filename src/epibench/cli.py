"""Command-line interface for epibench."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys


def _cmd_manifest(args: argparse.Namespace) -> int:
    from epibench.generator.manifest import generate_manifest, write_manifest
    from epibench.generator.pools import validate_pools

    validate_pools()
    tracks = [t.strip().upper() for t in args.tracks.split(",") if t.strip()]
    distractor_file = Path(args.distractor_file) if args.distractor_file else None
    manifest = generate_manifest(
        split=args.split,
        seed=args.seed,
        scenarios_per_track=args.scenarios_per_track,
        filler_count=args.filler_count,
        tracks=tracks,
        profile=args.profile,
        distractor_mode=args.distractor_mode,
        distractor_file=distractor_file,
    )
    default_out = (
        f"manifests/{manifest['profile']}_manifest.json"
        if args.output is None
        else args.output
    )
    out = Path(default_out)
    write_manifest(manifest, out)
    print(
        f"Manifest written to {out} ({manifest['total_scenarios']} scenarios, "
        f"profile={manifest['profile']}, mode={manifest['distractor_mode']}, "
        f"fillers={manifest['filler_count']}, sha256={manifest['manifest_sha256']})"
    )
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from epibench.runners.validate import format_validity_table, run_validity_gate

    tracks = [t.strip().upper() for t in args.tracks.split(",") if t.strip()]
    api_key = os.getenv("GEMINI_API_KEY", "")
    offline = bool(args.offline)
    if not offline and not api_key:
        print("GEMINI_API_KEY not set; falling back to --offline (validity unverified live).")
        offline = True

    distractor_file = Path(args.distractor_file) if args.distractor_file else None
    report = asyncio.run(
        run_validity_gate(
            tracks=tracks,
            scenarios_per_track=args.scenarios_per_track,
            filler_count=args.filler_count,
            seed=args.seed,
            budget_usd=args.budget_usd,
            api_key=api_key,
            offline=offline,
            output_dir=Path(args.output_dir),
            reader_model=args.reader_model,
            judge_model=args.judge_model,
            profile=args.profile,
            distractor_mode=args.distractor_mode,
            distractor_file=distractor_file,
            concurrency=args.concurrency,
        )
    )
    print(format_validity_table(report))
    return 0


def _cmd_eval_offline(args: argparse.Namespace) -> int:
    from epibench.runners.offline_eval import format_offline_report, run_offline_epistemic_eval

    tracks = [t.strip().upper() for t in args.tracks.split(",") if t.strip()]
    systems = [s.strip().lower() for s in args.systems.split(",") if s.strip()]
    distractor_file = Path(args.distractor_file) if args.distractor_file else None
    out_path = Path(args.output) if args.output else None
    report = asyncio.run(
        run_offline_epistemic_eval(
            profile=args.profile,
            tracks=tracks,
            systems=systems,
            scenarios_per_track=args.scenarios_per_track,
            filler_count=args.filler_count,
            distractor_mode=args.distractor_mode,
            distractor_file=distractor_file,
            token_budget=args.token_budget,
            tight_constraint_budget=args.tight_constraint_budget,
            seed=args.seed,
            use_fake_embedder=args.fake_embedder,
            output_path=out_path,
        )
    )
    print(format_offline_report(report))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="epibench",
        description="EpiBench: Epistemic long-term memory benchmark CLI.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("version", help="Print package version")

    p_manifest = subparsers.add_parser("manifest", help="Generate deterministic SHA-256 scenario manifest")
    p_manifest.add_argument("--split", type=str, default="dev")
    p_manifest.add_argument("--profile", choices=["dev", "hard", "stress", "epistemic_stress"], default="dev")
    p_manifest.add_argument("--seed", type=int, default=20261122)
    p_manifest.add_argument("--scenarios-per-track", type=int, default=12)
    p_manifest.add_argument("--filler-count", type=int, default=None)
    p_manifest.add_argument("--distractor-mode", choices=["neutral", "same_domain", "mixed"], default=None)
    p_manifest.add_argument("--distractor-file", type=str, default=None)
    p_manifest.add_argument("--tracks", type=str, default="R,D,X,F,I,P,K,E,Q")
    p_manifest.add_argument("--output", type=str, default=None)

    p_val = subparsers.add_parser("validate", help="Run null_memory vs oracle track validity gate")
    p_val.add_argument("--profile", choices=["dev", "hard", "stress", "epistemic_stress"], default="dev")
    p_val.add_argument("--tracks", type=str, default="R,D,X,F,I,P,K,E,Q")
    p_val.add_argument("--scenarios-per-track", type=int, default=12)
    p_val.add_argument("--filler-count", type=int, default=None)
    p_val.add_argument("--distractor-mode", choices=["neutral", "same_domain", "mixed"], default=None)
    p_val.add_argument("--distractor-file", type=str, default=None)
    p_val.add_argument("--seed", type=int, default=20261122)
    p_val.add_argument("--budget-usd", type=float, default=2.0)
    p_val.add_argument("--output-dir", type=str, default="results/validity")
    p_val.add_argument("--reader-model", type=str, default="gemini-3.5-flash-lite")
    p_val.add_argument("--judge-model", type=str, default="gemini-3.5-flash")
    p_val.add_argument("--concurrency", type=int, default=None)
    p_val.add_argument("--offline", action="store_true", help="Use deterministic offline fakes instead of LLM API")

    p_off = subparsers.add_parser("eval-offline", help="Run zero-cost ($0.00) offline retrieval & epistemic-hygiene suite")
    p_off.add_argument("--profile", choices=["dev", "hard", "stress", "epistemic_stress"], default="stress")
    p_off.add_argument("--tracks", type=str, default="R,D,X,F,I,P,K,E,Q")
    p_off.add_argument("--systems", type=str, default="null_memory,oracle,recent_context,vector_rag,hybrid_rag")
    p_off.add_argument("--scenarios-per-track", type=int, default=12)
    p_off.add_argument("--filler-count", type=int, default=None)
    p_off.add_argument("--distractor-mode", choices=["neutral", "same_domain", "mixed"], default=None)
    p_off.add_argument("--distractor-file", type=str, default=None)
    p_off.add_argument("--token-budget", type=int, default=None)
    p_off.add_argument("--tight-constraint-budget", type=int, default=600)
    p_off.add_argument("--seed", type=int, default=20261122)
    p_off.add_argument("--fake-embedder", action="store_true", help="Use deterministic FakeEmbedder (no ONNX)")
    p_off.add_argument("--output", type=str, default=None)

    args = parser.parse_args(argv)
    if args.command == "version":
        from epibench import __version__

        print(__version__)
        return 0
    if args.command == "manifest":
        return _cmd_manifest(args)
    if args.command == "validate":
        return _cmd_validate(args)
    if args.command == "eval-offline":
        return _cmd_eval_offline(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
