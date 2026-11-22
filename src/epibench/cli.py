"""Command-line interface for epibench."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="epibench",
        description="EpiBench: Epistemic long-term memory benchmark CLI.",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("version", help="Print package version")

    args, remaining = parser.parse_known_args(argv)
    if args.command == "version":
        from epibench import __version__

        print(__version__)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
