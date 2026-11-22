"""Relational Graph Memory Ablation Adapters.

Implements free zero-ingestion retrieval ablations against persisted graph memories:
1. No Relational Expansion (relational links disabled)
2. No History-Aware Retrieval (superseded values omitted from history queries)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
import uuid

try:
    from app.sigkimeno.context_engine import context_retriever
    from app.sigkimeno.settings import settings
    HAS_GRAPH_BACKEND = True
except ImportError:
    HAS_GRAPH_BACKEND = False

from EpiBench.adapters.base import RetrievalResult
from EpiBench.adapters.graph_memory_adapter import RelationalGraphAdapter, parse_timestamp


class GraphNoRelationsAdapter(RelationalGraphAdapter):
    """Relational Graph Memory with relational expansion disabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, system_name="graph_no_relations", **kwargs)

    async def retrieve(
        self,
        tenant_id: str,
        query: str,
        token_budget: int = 1500,
        as_of: Optional[str] = None,
    ) -> RetrievalResult:
        if not HAS_GRAPH_BACKEND:
            raise ImportError("Graph backend dependencies not available.")
        orig_weight = settings.retrieval_weight_graph
        try:
            object.__setattr__(settings, "retrieval_weight_graph", 0.0)
            return await super().retrieve(tenant_id, query, token_budget, as_of)
        finally:
            object.__setattr__(settings, "retrieval_weight_graph", orig_weight)


class GraphNoHistoryAdapter(RelationalGraphAdapter):
    """Relational Graph Memory with history-aware retrieval for earlier values disabled."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, system_name="graph_no_history", **kwargs)

    async def retrieve(
        self,
        tenant_id: str,
        query: str,
        token_budget: int = 1500,
        as_of: Optional[str] = None,
    ) -> RetrievalResult:
        if not HAS_GRAPH_BACKEND:
            raise ImportError("Graph backend dependencies not available.")
        t0 = time.perf_counter()
        uid = self._get_uuid(tenant_id)
        async with self.db.sessions() as sess:
            ctx_result = await context_retriever.retrieve_context(
                session=sess,
                user_id=uid,
                query=query,
                token_budget=token_budget,
                as_of=None,  # Suppress as-of time Travel
            )

        wall_time = time.perf_counter() - t0
        packed_text = ctx_result.text or ""
        token_count = ctx_result.tokens_used or (len(packed_text) // 4)

        return RetrievalResult(
            context=packed_text,
            token_count=token_count,
            retrieval_latency_ms=wall_time * 1000,
            metadata={"ablation": "no_history"},
        )
