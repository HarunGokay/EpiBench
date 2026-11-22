"""Full-Context Reader Adapter.

Reference baseline feeding all chronological conversation turns directly
into the reader up to the token budget.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from epibench.adapters.base import BaseMemoryAdapter, IngestResult, RetrievalResult


class FullContextAdapter(BaseMemoryAdapter):
    """Reference adapter feeding all chronological context to the reader."""

    def __init__(self, system_name: str = "full_context"):
        super().__init__(system_name)
        self._stores: Dict[str, List[Dict[str, Any]]] = {}

    async def ingest(self, tenant_id: str, sessions: List[Dict[str, Any]]) -> IngestResult:
        t0 = time.perf_counter()
        # Sort chronologically by session_date
        sorted_sessions = sorted(sessions, key=lambda s: s.get("session_date", ""))
        self._stores[tenant_id] = sorted_sessions
        latency = (time.perf_counter() - t0) * 1000
        return IngestResult(tenant_id=tenant_id, session_count=len(sessions), latency_ms=latency)

    async def retrieve(
        self,
        tenant_id: str,
        query: str,
        token_budget: int = 1500,
        as_of: Optional[str] = None,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        sessions = self._stores.get(tenant_id, [])

        lines = []
        tot_tok = 0
        for s in sessions:
            s_date = s.get("session_date", "")
            for turn in s.get("turns", []):
                role = turn.get("role", "user").capitalize()
                content = turn.get("content") or turn.get("text") or ""
                line = f"[{s_date}] {role}: {content}"
                c_tok = max(1, len(line) // 4)
                if tot_tok + c_tok > token_budget:
                    break
                lines.append(line)
                tot_tok += c_tok

        context = "\n\n".join(lines)
        latency = (time.perf_counter() - t0) * 1000
        return RetrievalResult(
            context=context,
            token_count=tot_tok,
            retrieval_latency_ms=latency,
            metadata={"turns_count": len(lines)},
        )

    async def delete(self, tenant_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        if tenant_id in self._stores:
            del self._stores[tenant_id]
            return {"supported": True, "deleted": True}
        return {"supported": True, "deleted": False}

    async def cleanup(self, tenant_id: str) -> None:
        if tenant_id in self._stores:
            del self._stores[tenant_id]
