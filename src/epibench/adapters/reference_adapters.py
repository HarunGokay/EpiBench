"""Reference Bound Adapters: NullMemoryAdapter (lower bound) and OracleAdapter (upper bound).

- NullMemoryAdapter (`null_memory`): Always returns empty context. Measures what a
  reader LLM can answer purely from question phrasing without any memory.
- OracleAdapter (`oracle`): Retains ONLY ground-truth evidence sessions (`is_evidence=True`),
  strictly filtered by `as_of` and updated by `delete()`. Measures the ceiling of
  perfect retrieval under the same reader prompt and scorer.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from epibench.adapters.base import (
    BaseMemoryAdapter,
    IngestResult,
    RetrievalResult,
    count_tokens,
    filter_sessions_as_of,
    format_session_text,
)
from epibench.judging.programmatic import normalize


class NullMemoryAdapter(BaseMemoryAdapter):
    """Lower-bound reference adapter that stores nothing and always returns empty context."""

    def __init__(self, system_name: str = "null_memory"):
        super().__init__(system_name)

    async def ingest(self, tenant_id: str, sessions: List[Dict[str, Any]]) -> IngestResult:
        return IngestResult(
            tenant_id=tenant_id,
            session_count=len(sessions),
            latency_ms=0.0,
            metadata={"reference_bound": "lower"},
        )

    async def retrieve(
        self,
        tenant_id: str,
        query: str,
        token_budget: int = 1500,
        as_of: Optional[str] = None,
    ) -> RetrievalResult:
        return RetrievalResult(
            context="",
            token_count=0,
            retrieval_latency_ms=0.0,
            metadata={"source_session_ids": [], "reference_bound": "lower"},
        )

    async def delete(
        self,
        tenant_id: str,
        session_id: Optional[str] = None,
        target: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {"supported": True, "deleted": True, "removed_count": 0}

    async def cleanup(self, tenant_id: str) -> None:
        return None


class OracleAdapter(BaseMemoryAdapter):
    """Upper-bound reference adapter returning exact as_of-filtered, deletion-aware evidence sessions."""

    def __init__(self, system_name: str = "oracle"):
        super().__init__(system_name)
        self._stores: Dict[str, List[Dict[str, Any]]] = {}

    async def ingest(self, tenant_id: str, sessions: List[Dict[str, Any]]) -> IngestResult:
        t0 = time.perf_counter()
        existing = self._stores.setdefault(tenant_id, [])
        for s in sessions:
            if s.get("is_evidence", False):
                existing.append(s)
        existing.sort(key=lambda s: (s.get("session_date", ""), s.get("session_id", "")))
        latency = (time.perf_counter() - t0) * 1000.0
        return IngestResult(
            tenant_id=tenant_id,
            session_count=len(sessions),
            latency_ms=latency,
            metadata={"reference_bound": "upper", "evidence_stored": len(existing)},
        )

    async def retrieve(
        self,
        tenant_id: str,
        query: str,
        token_budget: int = 1500,
        as_of: Optional[str] = None,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        sessions = filter_sessions_as_of(self._stores.get(tenant_id, []), as_of)
        sessions.sort(key=lambda s: (s.get("session_date", ""), s.get("session_id", "")))
        blocks = [format_session_text(s) for s in sessions]
        selected_ids = [s.get("session_id", f"ev_{i}") for i, s in enumerate(sessions)]
        context = "\n\n".join(b for b in blocks if b)
        latency = (time.perf_counter() - t0) * 1000.0
        return RetrievalResult(
            context=context,
            token_count=count_tokens(context),
            retrieval_latency_ms=latency,
            metadata={
                "source_session_ids": selected_ids,
                "reference_bound": "upper",
            },
        )

    async def delete(
        self,
        tenant_id: str,
        session_id: Optional[str] = None,
        target: Optional[str] = None,
    ) -> Dict[str, Any]:
        sessions = self._stores.get(tenant_id)
        if sessions is None:
            return {"supported": True, "deleted": False, "removed_count": 0}

        if session_id is None and target is None:
            removed = len(sessions)
            del self._stores[tenant_id]
            return {"supported": True, "deleted": removed > 0, "removed_count": removed}

        norm_target = normalize(target) if target else ""
        kept: List[Dict[str, Any]] = []
        removed = 0
        for s in sessions:
            s_text_norm = normalize(format_session_text(s))
            match_sid = session_id is not None and s.get("session_id") == session_id
            match_target = bool(norm_target) and (
                norm_target in s_text_norm
                or normalize(str(s.get("target_item", ""))) == norm_target
                or normalize(str(s.get("deleted_target", ""))) == norm_target
            )
            if match_sid or match_target:
                removed += 1
            else:
                kept.append(s)

        self._stores[tenant_id] = kept
        return {"supported": True, "deleted": removed > 0, "removed_count": removed}

    async def cleanup(self, tenant_id: str) -> None:
        self._stores.pop(tenant_id, None)
