"""Verify README.md links, committed manifest SHA-256s, and clean documentation invariants."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_markdown_tables(text: str) -> list[list[list[str]]]:
    tables: list[list[list[str]]] = []
    cur: list[list[str]] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|") and s.endswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                continue
            cur.append(cells)
        else:
            if cur:
                tables.append(cur)
                cur = []
    if cur:
        tables.append(cur)
    return tables


def verify_readme(root: Path = REPO_ROOT) -> list[str]:
    errors: list[str] = []
    readme_path = root / "README.md"
    if not readme_path.exists():
        return ["README.md not found"]
    text = readme_path.read_text(encoding="utf-8")

    # 1. Verify every relative markdown file link points to an existing file in the repo
    for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", text):
        target = match.group(1).strip()
        if target.startswith(("http://", "https://", "#")):
            continue
        if not (root / target).exists():
            errors.append(f"Broken relative link in README.md: {target}")

    # 2. Verify committed v2.2.0 manifest SHA-256 values appear accurately and match artifacts
    for name in ("dev_manifest.json", "hard_manifest.json", "stress_manifest.json"):
        m_path = root / "manifests" / name
        if not m_path.exists():
            errors.append(f"Missing committed manifest: manifests/{name}")
            continue
        m = json.loads(m_path.read_text(encoding="utf-8"))
        sha = m["manifest_sha256"]
        if sha not in text:
            errors.append(f"Manifest SHA-256 for {name} ({sha}) missing from README.md")

    # 3. Verify zero banned placeholder/stale strings
    for banned in ("Memory System A", "50,000 turns", "Pilot Smoke Test", "Illustrative"):
        if banned in text:
            errors.append(f"Banned placeholder/stale phrase found in README.md: {banned!r}")

    # 4. Verify no unverified performance/leaderboard tables are embedded in README.md
    tables = _parse_markdown_tables(text)
    perf_headers = {"composite (%)", "question acc (%)", "oracle (%)", "fact recall (%)", "depth 6 acc (%)"}
    for idx, tbl in enumerate(tables):
        hdrs = {c.lower() for c in tbl[0]}
        if hdrs & perf_headers:
            errors.append(f"Unexpected performance table #{idx} found in README.md: {tbl[0]}")

    return errors


def main() -> int:
    errors = verify_readme()
    if errors:
        print("README VERIFICATION FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("README verification passed: all links, manifest SHA-256s (dev/hard/stress), and clean-doc checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
