"""Shared utilities for epibench."""

from __future__ import annotations

from pathlib import Path
import subprocess


def get_git_commit_sha(repo_dir: Path | None = None) -> str:
    """Return the current short git commit SHA, or 'unknown' if unavailable."""
    cwd = str(repo_dir) if repo_dir is not None else str(Path(__file__).resolve().parent)
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        sha = out.strip()
        return sha if sha else "unknown"
    except Exception:
        return "unknown"
