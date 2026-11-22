"""EpiBench Memory Adapter Interface.

Defines the standardized contract for memory layer evaluation across:
- Ingestion of dated multi-turn conversation sessions
- Retrieval bounded by token budget and optional point-in-time timestamp
- Optional deletion for lifecycle / erasure testing
- Optional clarification questioning for interactive memory evaluation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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

    def __init__(self, system_name: str):
        self.system_name = system_name

    @abstractmethod
    async def ingest(self, tenant_id: str, sessions: List[Dict[str, Any]]) -> IngestResult:
        """Ingest dated multi-turn conversational sessions into the memory store."""
        pass

    @abstractmethod
    async def retrieve(
        self,
        tenant_id: str,
        query: str,
        token_budget: int = 1500,
        as_of: Optional[str] = None,
    ) -> RetrievalResult:
        """Retrieve bounded textual context relevant to the query as-of a given point in time."""
        pass

    async def delete(self, tenant_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Optional: Delete user or session memories. Default: unsupported."""
        return {"supported": False, "deleted": False, "reason": "Not implemented"}

    async def ask_clarification(self, tenant_id: str, query: str) -> List[str]:
        """Optional: Interactive memory layer proposing clarification questions."""
        return []

    async def cleanup(self, tenant_id: str) -> None:
        """Clean up tenant storage resources after evaluation."""
        pass
