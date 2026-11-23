"""Verify that the epibench repository contains no forbidden private or legacy strings."""

from __future__ import annotations

from pathlib import Path
import re
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent

EXCLUDED_DIRS = {
    ".git",
    "results_archive_invalid",
    ".pytest_cache",
    "__pycache__",
    ".venv",
    "venv",
}
EXCLUDED_FILES = {"CHANGES.md"}

# Constructed dynamically so this checker script itself never contains raw forbidden literals.
CASE_INSENSITIVE_PATTERNS = [
    "sig" + "kimeno",
    "ko" + "ra",
    r"from " + r"app\.",
    "Context-" + "MVP",
]
# Generic English term checked across all code, tests, docs, configs, and scripts
# (excluded only in raw LLM-generated results/*.json transcripts such as finance Track E).
CODE_AND_DOC_ONLY_PATTERNS = [
    "bal" + "ance",
]
CASE_SENSITIVE_PATTERNS = [
    "Epi" + r"Bench\.",
]


def scan_repo(root: Path = REPO_ROOT) -> list[str]:
    ci_re = re.compile("|".join(CASE_INSENSITIVE_PATTERNS), re.IGNORECASE)
    doc_re = re.compile("|".join(CODE_AND_DOC_ONLY_PATTERNS), re.IGNORECASE)
    cs_re = re.compile("|".join(CASE_SENSITIVE_PATTERNS))
    hits: list[str] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in rel.parts):
            continue
        if rel.name in EXCLUDED_FILES:
            continue
        is_raw_result_json = rel.parts[0] == "results" and rel.suffix == ".json"
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            if ci_re.search(line) or cs_re.search(line) or (not is_raw_result_json and doc_re.search(line)):
                hits.append(f"{rel.as_posix()}:{lineno}: {line.strip()}")

    return hits


def main() -> int:
    hits = scan_repo()
    if hits:
        print("FORBIDDEN STRINGS DETECTED:")
        for h in hits:
            print(f"  {h}")
        return 1
    print("Forbidden-string check passed (0 hits).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
