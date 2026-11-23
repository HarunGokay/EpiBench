"""Distractor Pool Loader for epibench.

Loads non-evidence conversational sessions to interleave as realistic haystacks
around scenario evidence sessions. Supports three built-in distractor modes:
- ``neutral``: general hobby and everyday Q&A sessions (``sample_distractors.json``)
- ``same_domain``: realistic domain-matched work/finance/health/ops/legal sessions
  (``same_domain_distractors.json``) creating genuine semantic interference
- ``mixed``: deterministic 50/50 mix of ``same_domain`` and ``neutral`` sessions
or any custom JSON file passed via ``dataset_path`` (``--distractor-file``).
"""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

BUNDLED_NEUTRAL_PATH = Path(__file__).parent.parent / "data" / "sample_distractors.json"
BUNDLED_SAME_DOMAIN_PATH = Path(__file__).parent.parent / "data" / "same_domain_distractors.json"
BUNDLED_PATH = BUNDLED_NEUTRAL_PATH  # Backward-compatible alias

VALID_DISTRACTOR_MODES = ("neutral", "same_domain", "mixed")


def _load_sessions_from_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Distractor dataset not found at {path}.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data and isinstance(data, list) and "haystack_sessions" in data[0]:
        return _extract_longmemeval_sessions(data)
    if not isinstance(data, list):
        raise ValueError(f"Distractor file {path} must contain a JSON list of sessions.")
    return data


def _extract_longmemeval_sessions(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen_first_turns = set()
    extracted: List[Dict[str, Any]] = []
    for q in data:
        sessions = q.get("haystack_sessions", [])
        dates = q.get("haystack_dates", [])
        s_ids = q.get("haystack_session_ids", [])
        ans_s_ids = set(q.get("answer_session_ids", []))

        for idx, turns in enumerate(sessions):
            sid = s_ids[idx] if idx < len(s_ids) else f"s_{idx}"
            if sid in ans_s_ids or not turns:
                continue

            first_turn_sig = turns[0].get("content", "")[:100]
            if first_turn_sig in seen_first_turns:
                continue
            seen_first_turns.add(first_turn_sig)

            extracted.append({
                "original_question_id": q.get("question_id"),
                "session_id": sid,
                "date": dates[idx] if idx < len(dates) else "2024/01/01 12:00:00",
                "turns": turns,
            })
    return extracted


class DistractorPool:
    """Pool of filler conversational sessions loaded from bundled datasets or an explicit path."""

    _instances: Dict[Tuple[str, Optional[str]], "DistractorPool"] = {}
    _instance: Optional["DistractorPool"] = None

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        mode: str = "neutral",
    ):
        norm_mode = (mode or "neutral").strip().lower()
        if dataset_path is None and norm_mode not in VALID_DISTRACTOR_MODES:
            raise ValueError(
                f"Invalid distractor mode {mode!r}; expected one of {VALID_DISTRACTOR_MODES}."
            )
        self.mode = norm_mode
        self.dataset_path = Path(dataset_path) if dataset_path is not None else (
            BUNDLED_SAME_DOMAIN_PATH if norm_mode == "same_domain" else BUNDLED_NEUTRAL_PATH
        )
        self._explicit_path = dataset_path is not None
        self._pool: List[Dict[str, Any]] = []
        self._neutral_pool: List[Dict[str, Any]] = []
        self._same_domain_pool: List[Dict[str, Any]] = []
        self._load_pool()

    @classmethod
    def get_instance(
        cls,
        mode: str = "neutral",
        dataset_path: Optional[Path] = None,
    ) -> "DistractorPool":
        norm_mode = (mode or "neutral").strip().lower()
        resolved_key = str(Path(dataset_path).resolve()) if dataset_path is not None else None
        cache_key = (norm_mode, resolved_key)
        if cache_key not in cls._instances:
            inst = cls(dataset_path=dataset_path, mode=norm_mode)
            cls._instances[cache_key] = inst
            if norm_mode == "neutral" and resolved_key is None:
                cls._instance = inst
        return cls._instances[cache_key]

    def _load_pool(self) -> None:
        if self._explicit_path:
            self._pool = _load_sessions_from_json(self.dataset_path)
            return

        if self.mode == "neutral":
            self._neutral_pool = _load_sessions_from_json(BUNDLED_NEUTRAL_PATH)
            self._pool = list(self._neutral_pool)
        elif self.mode == "same_domain":
            self._same_domain_pool = _load_sessions_from_json(BUNDLED_SAME_DOMAIN_PATH)
            self._pool = list(self._same_domain_pool)
        elif self.mode == "mixed":
            self._neutral_pool = _load_sessions_from_json(BUNDLED_NEUTRAL_PATH)
            self._same_domain_pool = _load_sessions_from_json(BUNDLED_SAME_DOMAIN_PATH)
            self._pool = list(self._same_domain_pool) + list(self._neutral_pool)

    @property
    def size(self) -> int:
        return len(self._pool)

    @staticmethod
    def _draw_from(pool: List[Dict[str, Any]], k: int, rng: random.Random) -> List[Dict[str, Any]]:
        if k <= 0:
            return []
        if k <= len(pool):
            return rng.sample(pool, k)
        # Draw full shuffled passes first to maximize diversity before repeating
        result: List[Dict[str, Any]] = []
        remaining = k
        while remaining > 0:
            batch = list(pool)
            rng.shuffle(batch)
            take = min(remaining, len(batch))
            result.extend(batch[:take])
            remaining -= take
        return result

    def sample_distractors(self, n: int, seed: int) -> List[Dict[str, Any]]:
        """Deterministically sample N filler sessions using a fixed random seed."""
        if n <= 0:
            return []
        rng = random.Random(seed)
        if not self._explicit_path and self.mode == "mixed":
            n_same = n // 2 + (n % 2)
            n_neutral = n - n_same
            chosen = self._draw_from(self._same_domain_pool, n_same, rng) + self._draw_from(
                self._neutral_pool, n_neutral, rng
            )
            rng.shuffle(chosen)
            return chosen
        if n > len(self._pool):
            return rng.choices(self._pool, k=n)
        return rng.sample(self._pool, n)
