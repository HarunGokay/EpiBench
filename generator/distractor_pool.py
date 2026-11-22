"""LongMemEval Distractor Pool Extractor.

Extracts non-evidence conversational sessions from the LongMemEval-S benchmark dataset
to serve as realistic filler sessions interleaved into EpiBench timelines.
Deterministic and seeded.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional

LONGMEMEVAL_PATH = Path("data/benchmarks/longmemeval_s.json")
BUNDLED_PATH = Path(__file__).parent.parent / "data" / "sample_distractors.json"


class DistractorPool:
    """Pool of filler conversational sessions extracted from LongMemEval or bundled fallback."""

    _instance: Optional["DistractorPool"] = None

    def __init__(self, dataset_path: Path = LONGMEMEVAL_PATH):
        self.dataset_path = dataset_path
        self._pool: List[Dict[str, Any]] = []
        self._load_pool()

    @classmethod
    def get_instance(cls) -> "DistractorPool":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_pool(self) -> None:
        if self.dataset_path.exists():
            with open(self.dataset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._load_longmemeval_data(data)
        elif BUNDLED_PATH.exists():
            with open(BUNDLED_PATH, "r", encoding="utf-8") as f:
                self._pool = json.load(f)
        else:
            raise FileNotFoundError(
                f"Neither LongMemEval dataset ({self.dataset_path}) nor bundled fallback ({BUNDLED_PATH}) was found."
            )

    def _load_longmemeval_data(self, data: List[Dict[str, Any]]) -> None:
        seen_first_turns = set()
        for q in data:
            sessions = q.get("haystack_sessions", [])
            dates = q.get("haystack_dates", [])
            s_ids = q.get("haystack_session_ids", [])
            ans_s_ids = set(q.get("answer_session_ids", []))

            for idx, turns in enumerate(sessions):
                sid = s_ids[idx] if idx < len(s_ids) else f"s_{idx}"
                if sid in ans_s_ids:
                    continue  # Skip evidence sessions

                if not turns:
                    continue

                first_turn_sig = turns[0].get("content", "")[:100]
                if first_turn_sig in seen_first_turns:
                    continue
                seen_first_turns.add(first_turn_sig)

                self._pool.append({
                    "original_question_id": q.get("question_id"),
                    "session_id": sid,
                    "date": dates[idx] if idx < len(dates) else "2024/01/01 12:00:00",
                    "turns": turns,
                })

    @property
    def size(self) -> int:
        return len(self._pool)

    def sample_distractors(self, n: int, seed: int) -> List[Dict[str, Any]]:
        """Deterministically sample N filler sessions using a fixed random seed."""
        rng = random.Random(seed)
        if n > len(self._pool):
            return rng.choices(self._pool, k=n)
        return rng.sample(self._pool, n)
