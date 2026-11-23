"""EpiBench Memory Adapter Interface (ADAPTER_CONTRACT_VERSION 2.0.0).

Defines the standardized contract for memory layer evaluation across:
- Incremental or batch ingestion of dated multi-turn conversation sessions
- Retrieval bounded by token budget and optional point-in-time `as_of` cutoff
- Targeted or tenant-level deletion (`delete(tenant_id, session_id=None, target=None)`)
- Shared `as_of` date-filtering and unified token-counting helpers
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Any, Dict, List, Optional

from epibench.generator.scenario_generator import parse_session_date

ADAPTER_CONTRACT_VERSION = "2.0.0"

_DATE_ONLY_RE = re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}$")


def count_tokens(text: str) -> int:
    """Unified deterministic token estimator used across all adapters and runners.

    Uses ceil(len(text) / 4) for non-empty strings and 0 for empty/whitespace-only strings.
    """
    if not text or not text.strip():
        return 0
    return max(1, (len(text) + 3) // 4)


def parse_as_of_cutoff(as_of: Optional[str]) -> Optional[datetime]:
    """Parse an `as_of` cutoff into a UTC datetime.

    Date-only strings (e.g. '2024-01-25') represent the inclusive end of that day
    (23:59:59.999999 UTC) so sessions occurring on that date are included while
    all subsequent days are excluded.
    """
    if as_of is None or not str(as_of).strip():
        return None
    raw = str(as_of).strip()
    dt = parse_session_date(raw)
    if _DATE_ONLY_RE.match(raw):
        return dt.replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc)
    return dt


def is_on_or_before_as_of(session_date: Optional[str], as_of: Optional[str]) -> bool:
    """Return True iff `session_date` is on or before the `as_of` cutoff."""
    cutoff = parse_as_of_cutoff(as_of)
    if cutoff is None:
        return True
    if not session_date:
        return False
    sess_dt = parse_session_date(session_date)
    return sess_dt <= cutoff


def filter_sessions_as_of(
    sessions: List[Dict[str, Any]],
    as_of: Optional[str],
) -> List[Dict[str, Any]]:
    """Filter out any session dated after `as_of` BEFORE ranking or packing."""
    cutoff = parse_as_of_cutoff(as_of)
    if cutoff is None:
        return list(sessions)
    kept: List[Dict[str, Any]] = []
    for s in sessions:
        s_date = s.get("session_date")
        if s_date and parse_session_date(s_date) <= cutoff:
            kept.append(s)
    return kept


def format_session_text(session: Dict[str, Any]) -> str:
    """Format a dated conversation session into canonical textual context."""
    s_date = session.get("session_date", "")
    lines = []
    for turn in session.get("turns", []):
        role = str(turn.get("role", "user")).capitalize()
        content = turn.get("content") or turn.get("text") or ""
        lines.append(f"[{s_date}] {role}: {content}")
    return "\n".join(lines)


@dataclass
class IngestResult:
    tenant_id: str
    session_count: int
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    context: str
    token_count: int
    retrieval_latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseMemoryAdapter(ABC):
    """Abstract base adapter for any memory architecture evaluated by epibench."""

    contract_version: str = ADAPTER_CONTRACT_VERSION

    def __init__(self, system_name: str):
        self.system_name = system_name

    @abstractmethod
    async def ingest(self, tenant_id: str, sessions: List[Dict[str, Any]]) -> IngestResult:
        """Ingest dated multi-turn conversational sessions into the memory store.

        Repeated calls for the same `tenant_id` should append rather than wipe
        prior sessions so chronological replay with interleaved deletions works.
        """
        raise NotImplementedError

    @abstractmethod
    async def retrieve(
        self,
        tenant_id: str,
        query: str,
        token_budget: int = 1500,
        as_of: Optional[str] = None,
    ) -> RetrievalResult:
        """Retrieve bounded textual context relevant to `query` as of `as_of`.

        Adapters that track source sessions should include `source_session_ids`
        (List[str]) in `RetrievalResult.metadata` so the runner can audit `as_of` compliance.
        """
        raise NotImplementedError

    async def delete(
        self,
        tenant_id: str,
        session_id: Optional[str] = None,
        target: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Optional: Delete memories by session_id, target string, or entire tenant.

        Default returns unsupported=False; systems supporting erasure override this.
        """
        return {"supported": False, "deleted": False, "reason": "Not implemented"}

    async def ask_clarification(self, tenant_id: str, query: str) -> List[str]:
        """Optional: Interactive memory layer proposing clarification questions."""
        return []

    async def cleanup(self, tenant_id: str) -> None:
        """Clean up tenant storage resources after evaluation."""
        pass
