"""Verify that every numerical table and artifact claim in README.md matches committed files."""

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
            # Skip separator rows like | --- | ---: |
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

    # 2. Load committed artifacts
    manifest = json.loads((root / "manifests/dev_manifest.json").read_text(encoding="utf-8"))
    validity = json.loads((root / "results/validity/validity_report.json").read_text(encoding="utf-8"))
    dev_res = json.loads((root / "results/dev_v2/results.json").read_text(encoding="utf-8"))
    depth_res = json.loads((root / "results/depth_v2/results_depth.json").read_text(encoding="utf-8"))
    eff_res = json.loads((root / "results/efficiency_v2.json").read_text(encoding="utf-8"))

    # Verify manifest SHA256 appears accurately
    sha = manifest["manifest_sha256"]
    if sha not in text:
        errors.append(f"Manifest SHA-256 {sha} missing from README.md")
    if dev_res["manifest_sha256"] != sha:
        errors.append(f"dev_v2 manifest_sha256 mismatch: {dev_res['manifest_sha256']} != {sha}")

    # Verify headline numbers in prose
    sec = dev_res["performance_summary"]["secondary_audit"]
    expected_snippets = [
        f"agreement = `{sec['agreement_rate']:.2f}%`",
        f"Cohen's kappa = `{sec['cohens_kappa']:.4f}`",
        f"`${dev_res['spend_summary']['total_spend_usd']:.6f} USD`",
        f"`${depth_res['total_spend_summary']['total_spend_usd']:.6f} USD`",
    ]
    for snip in expected_snippets:
        if snip not in text:
            errors.append(f"Expected verified prose snippet missing from README.md: {snip}")

    # Verify zero forbidden placeholder claims
    for banned in ("Memory System A", "50,000 turns", "Pilot Smoke Test", "Illustrative"):
        if banned in text:
            errors.append(f"Banned placeholder/stale phrase found in README.md: {banned!r}")

    # 3. Verify every markdown table against committed artifacts & pools
    from epibench.generator.pools import ALL_POOLS

    tables = _parse_markdown_tables(text)
    if len(tables) != 6:
        errors.append(f"Expected 6 markdown tables in README.md, found {len(tables)}")
        return errors

    # Table 0: Track template counts
    for row in tables[0][1:]:
        track = row[0].strip("* ")
        claimed_n = int(row[1])
        actual_n = len(ALL_POOLS[track])
        if claimed_n != actual_n:
            errors.append(f"Table 0 track {track} count mismatch: README={claimed_n}, actual={actual_n}")

    # Table 1: Validity gate table
    for row in tables[1][1:]:
        track = row[0].strip("* ")
        metric = row[1].strip("` ")
        o_val = float(row[2])
        n_val = float(row[3])
        status = row[4]
        actual = validity["tracks"][track]
        if metric != actual["scoring_metric"]:
            errors.append(f"Validity {track} metric mismatch: {metric} != {actual['scoring_metric']}")
        if abs(o_val - actual["oracle_score_pct"]) > 1e-2:
            errors.append(f"Validity {track} oracle mismatch: {o_val} != {actual['oracle_score_pct']}")
        if abs(n_val - actual["null_memory_score_pct"]) > 1e-2:
            errors.append(f"Validity {track} null mismatch: {n_val} != {actual['null_memory_score_pct']}")
        if status != actual["status"]:
            errors.append(f"Validity {track} status mismatch: {status} != {actual['status']}")

    # Table 2: Dev_v2 main leaderboard
    ps = dev_res["performance_summary"]
    for row in tables[2][1:]:
        sys_name = row[0].strip("` ")
        bs = ps["by_system"][sys_name]
        dr = ps["dimensional_reports"][sys_name]
        checks = [
            ("composite", float(row[1]), dr["composite_score"]),
            ("question_accuracy", float(row[2]), bs["question_accuracy"]),
            ("scenario_pass_rate", float(row[4]), bs["scenario_pass_rate"]),
            ("avg_context_tokens", float(row[5]), bs["avg_context_tokens"]),
            ("avg_retrieval_latency_ms", float(row[6]), bs["avg_retrieval_latency_ms"]),
            ("total_spend_usd", float(row[7]), bs["total_spend_usd"]),
        ]
        for label, claimed, actual in checks:
            if abs(claimed - actual) > 1e-4:
                errors.append(f"Dev table {sys_name}.{label} mismatch: README={claimed}, json={actual}")
        expected_ci = f"[{bs['question_ci_95'][0]:.2f}, {bs['question_ci_95'][1]:.2f}]"
        if row[3] != expected_ci:
            errors.append(f"Dev table {sys_name}.CI mismatch: README={row[3]}, json={expected_ci}")

    # Table 3: Dev_v2 dimensional breakdown
    dim_keys = [
        "factual_accuracy",
        "temporal_consistency",
        "dependency_reasoning",
        "constraint_compliance",
        "provenance",
        "calibration",
        "privacy_and_deletion",
        "retention_control",
    ]
    for row in tables[3][1:]:
        sys_name = row[0].strip("` ")
        dr = ps["dimensional_reports"][sys_name]
        for idx, dk in enumerate(dim_keys, start=1):
            claimed = float(row[idx])
            actual = float(dr[dk])
            if abs(claimed - actual) > 1e-2:
                errors.append(f"Dim table {sys_name}.{dk} mismatch: README={claimed}, json={actual}")

    # Table 4: Depth stress comparison
    cm = depth_res["comparison_matrix"]
    for row in tables[4][1:]:
        sys_name = row[0].strip("` ")
        entry = cm[sys_name]
        checks = [
            ("d6_acc", float(row[1]), entry["question_accuracy_by_depth"]["6"]),
            ("d6_tok", float(row[3]), entry["avg_tokens_by_depth"]["6"]),
            ("d40_acc", float(row[4]), entry["question_accuracy_by_depth"]["40"]),
            ("d40_tok", float(row[6]), entry["avg_tokens_by_depth"]["40"]),
            ("d40_scen", float(row[7]), entry["scenario_pass_rate_by_depth"]["40"]),
        ]
        for label, claimed, actual in checks:
            if abs(claimed - actual) > 1e-2:
                errors.append(f"Depth table {sys_name}.{label} mismatch: README={claimed}, json={actual}")
        d6_ci = entry["question_ci_95_by_depth"]["6"]
        d40_ci = entry["question_ci_95_by_depth"]["40"]
        if row[2] != f"[{d6_ci[0]:.2f}, {d6_ci[1]:.2f}]":
            errors.append(f"Depth table {sys_name}.d6_ci mismatch: {row[2]}")
        if row[5] != f"[{d40_ci[0]:.2f}, {d40_ci[1]:.2f}]":
            errors.append(f"Depth table {sys_name}.d40_ci mismatch: {row[5]}")

    # Table 5: Efficiency & planted-fact recall
    for row in tables[5][1:]:
        scale = row[0].strip()
        sys_name = row[1].strip("` ")
        entry = eff_res["by_scale"][scale][sys_name]
        checks = [
            ("fact_retrieval_recall_pct", float(row[2]), entry["fact_retrieval_recall_pct"]),
            ("ingest_time_s", float(row[3]), entry["ingest_time_s"]),
            ("avg_retrieval_latency_ms", float(row[4]), entry["avg_retrieval_latency_ms"]),
            ("avg_context_tokens", float(row[5]), entry["avg_context_tokens"]),
        ]
        for label, claimed, actual in checks:
            if abs(claimed - actual) > 1e-3:
                errors.append(f"Efficiency table {scale}/{sys_name}.{label} mismatch: README={claimed}, json={actual}")

    return errors


def main() -> int:
    errors = verify_readme()
    if errors:
        print("README CLAIM VERIFICATION FAILED:")
        for err in errors:
            print(f"  - {err}")
        return 1
    print("README claim verification passed: all 6 tables, links, manifest SHA-256, and headline metrics match committed artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
