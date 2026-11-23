"""JSON Schema validation for EpiBench evaluation result artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import jsonschema

RESULTS_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "results_schema.json"


def load_results_schema() -> Dict[str, Any]:
    with open(RESULTS_SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_results_schema(results_obj: Dict[str, Any]) -> None:
    """Validate an EpiBench results dictionary against the canonical v2 JSON Schema."""
    schema = load_results_schema()
    jsonschema.validate(instance=results_obj, schema=schema)
