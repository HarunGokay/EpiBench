"""Naive Vector RAG Adapter.

Standard dense vector retrieval using dense embeddings (BAAI/bge-small-en-v1.5)
and cosine similarity, bounded strictly by token budget.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from epibench.adapters.base import BaseMemoryAdapter, IngestResult, RetrievalResult
from epibench.embedding import EmbeddingService


class VectorRAGAdapter(BaseMemoryAdapter):
    """Naive Vector RAG implementation using local dense embeddings."""

    def __init__(self, system_name: str = "vector_rag"):
        super().__init__(system_name)
        self.embedder = EmbeddingService.get_instance()
        # In-memory storage keyed by tenant_id
        self._stores: Dict[str, List[Dict[str, Any]]] = {}

    async def ingest(self, tenant_id: str, sessions: List[Dict[str, Any]]) -> IngestResult:
        t0 = time.perf_counter()
        chunks: List[str] = []
        for s in sessions:
            s_date = s.get("session_date", "")
            for turn in s.get("turns", []):
                role = turn.get("role", "user")
                content = turn.get("content") or turn.get("text") or ""
                chunk_str = f"[{s_date}] {role.capitalize()}: {content}"
                chunks.append(chunk_str)

        if not chunks:
            self._stores[tenant_id] = []
            return IngestResult(tenant_id=tenant_id, session_count=len(sessions), latency_ms=0.0)

        embeddings = self.embedder.embed_batch(chunks)
        stored_items = []
        for c, emb in zip(chunks, embeddings):
            stored_items.append({"text": c, "embedding": emb})

        self._stores[tenant_id] = stored_items
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
        items = self._stores.get(tenant_id, [])
        if not items:
            return RetrievalResult(context="", token_count=0, retrieval_latency_ms=0.0)

        q_emb = self.embedder.embed_text(query)
        scored = []
        for item in items:
            sim = self.embedder.cosine_similarity(q_emb, item["embedding"])
            scored.append((sim, item["text"]))

        scored.sort(key=lambda x: x[0], reverse=True)

        selected = []
        tot_tok = 0
        for _, text in scored:
            # 1 token approx 4 characters
            c_tok = max(1, len(text) // 4)
            if tot_tok + c_tok > token_budget:
                break
            selected.append(text)
            tot_tok += c_tok

        context = "\n\n---\n\n".join(selected)
        latency = (time.perf_counter() - t0) * 1000
        return RetrievalResult(
            context=context,
            token_count=tot_tok,
            retrieval_latency_ms=latency,
            metadata={"chunks_retrieved": len(selected)},
        )

    async def delete(self, tenant_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        if tenant_id in self._stores:
            del self._stores[tenant_id]
            return {"supported": True, "deleted": True}
        return {"supported": True, "deleted": False}

    async def cleanup(self, tenant_id: str) -> None:
        if tenant_id in self._stores:
            del self._stores[tenant_id]
