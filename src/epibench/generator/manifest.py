"""Deterministic Scenario Manifest Generator for epibench."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from epibench.generator.scenario_generator import (
    GENERATOR_VERSION,
    VALID_TRACKS,
    ScenarioGenerator,
    resolve_profile,
)


def canonical_json_bytes(obj: Any) -> bytes:
    """Serialize an object to canonical UTF-8 JSON bytes for deterministic hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def scenario_sha256(scenario_dict: Dict[str, Any]) -> str:
    """Compute SHA-256 hex digest of a serialized Scenario dict."""
    return hashlib.sha256(canonical_json_bytes(scenario_dict)).hexdigest()


def generate_manifest(
    split: str = "dev",
    seed: int = 20261122,
    scenarios_per_track: int = 12,
    filler_count: Optional[int] = None,
    tracks: Optional[Sequence[str]] = None,
    generator: Optional[ScenarioGenerator] = None,
    profile: str = "dev",
    distractor_mode: Optional[str] = None,
    distractor_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate a reproducible SHA-256 manifest for a benchmark split and profile."""
    resolved = resolve_profile(
        profile=profile,
        filler_count=filler_count,
        distractor_mode=distractor_mode,
    )
    eff_fillers = resolved["filler_count"]
    eff_mode = "custom" if distractor_file is not None else resolved["distractor_mode"]

    gen = generator or ScenarioGenerator(
        distractor_mode=resolved["distractor_mode"],
        distractor_file=distractor_file,
    )
    track_list: List[str] = [t.upper() for t in (tracks or VALID_TRACKS)]

    entries: List[Dict[str, Any]] = []
    for track in track_list:
        for idx in range(scenarios_per_track):
            sc = gen.generate_scenario(
                track=track,
                scenario_idx=idx,
                split=split,
                seed=seed,
                filler_count=eff_fillers,
            )
            sc_dict = sc.to_dict()
            entries.append({
                "scenario_id": sc.scenario_id,
                "track": sc.track,
                "template_id": sc.metadata.get("template_id", ""),
                "session_count": len(sc.sessions),
                "question_count": len(sc.questions),
                "sha256": scenario_sha256(sc_dict),
            })

    manifest_digest = hashlib.sha256(canonical_json_bytes(entries)).hexdigest()
    return {
        "generator_version": GENERATOR_VERSION,
        "profile": resolved["profile"],
        "distractor_mode": eff_mode,
        "split": split,
        "seed": seed,
        "scenarios_per_track": scenarios_per_track,
        "filler_count": eff_fillers,
        "tracks": track_list,
        "total_scenarios": len(entries),
        "manifest_sha256": manifest_digest,
        "scenarios": entries,
    }


def write_manifest(manifest: Dict[str, Any], output_path: Path) -> Path:
    """Write a manifest dictionary to disk as formatted JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
        f.write("\n")
    return output_path
