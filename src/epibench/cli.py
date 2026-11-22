"""Command-line interface for epibench."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _cmd_manifest(args: argparse.Namespace) -> int:
    from epibench.generator.manifest import generate_manifest, write_manifest
    from epibench.generator.pools import validate_pools

    validate_pools()
    tracks = [t.strip().upper() for t in args.tracks.split(",") if t.strip()]
    manifest = generate_manifest(
        split=args.split,
        seed=args.seed,
        scenarios_per_track=args.scenarios_per_track,
        filler_count=args.filler_count,
        tracks=tracks,
    )
    out = Path(args.output)
    write_manifest(manifest, out)
    print(f"Manifest written to {out} ({manifest['total_scenarios']} scenarios, sha256={manifest['manifest_sha256']})")
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
    p_manifest.add_argument("--seed", type=int, default=20261122)
    p_manifest.add_argument("--scenarios-per-track", type=int, default=12)
    p_manifest.add_argument("--filler-count", type=int, default=6)
    p_manifest.add_argument("--tracks", type=str, default="R,D,X,F,I,P,K,E,Q")
    p_manifest.add_argument("--output", type=str, default="manifests/dev_manifest.json")

    args = parser.parse_args(argv)
    if args.command == "version":
        from epibench import __version__

        print(__version__)
        return 0
    if args.command == "manifest":
        return _cmd_manifest(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
