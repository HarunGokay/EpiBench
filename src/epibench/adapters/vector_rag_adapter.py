"""Naive Vector RAG Adapter.

Standard dense vector retrieval using local FastEmbed embeddings
(BAAI/bge-small-en-v1.5) and cosine similarity, strictly respecting
`as_of` cutoffs before ranking, targeted deletion, and unified token budgets.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from epibench.adapters.base import (
    BaseMemoryAdapter,
    IngestResult,
    RetrievalResult,
    count_tokens,
    is_on_or_before_as_of,
)
from epibench.embedding import EmbeddingService
from epibench.judging.programmatic import normalize


class VectorRAGAdapter(BaseMemoryAdapter):
    """Dense Vector RAG baseline using local FastEmbed embeddings."""

    def __init__(
        self,
        system_name: str = "vector_rag",
        embedder: Optional[Any] = None,
    ):
        super().__init__(system_name)
        self.embedder = embedder or EmbeddingService.get_instance()
        self._stores: Dict[str, List[Dict[str, Any]]] = {}

    async def ingest(self, tenant_id: str, sessions: List[Dict[str, Any]]) -> IngestResult:
        t0 = time.perf_counter()
        raw_items: List[Dict[str, Any]] = []
        for s in sessions:
            s_id = s.get("session_id", "")
            s_date = s.get("session_date", "")
            for turn in s.get("turns", []):
                role = str(turn.get("role", "user")).capitalize()
                content = turn.get("content") or turn.get("text") or ""
                chunk_str = f"[{s_date}] {role}: {content}"
                raw_items.append({
                    "text": chunk_str,
                    "session_id": s_id,
                    "session_date": s_date,
                })

        store = self._stores.setdefault(tenant_id, [])
        if raw_items:
            embeddings = self.embedder.embed_batch([it["text"] for it in raw_items])
            for it, emb in zip(raw_items, embeddings):
                it["embedding"] = emb
                store.append(it)

        latency = (time.perf_counter() - t0) * 1000.0
        return IngestResult(
            tenant_id=tenant_id,
            session_count=len(sessions),
            latency_ms=latency,
        )

    async def retrieve(
        self,
        tenant_id: str,
        query: str,
        token_budget: int = 1500,
        as_of: Optional[str] = None,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        all_items = self._stores.get(tenant_id, [])
        # Filter strictly by as_of BEFORE ranking
        items = [it for it in all_items if is_on_or_before_as_of(it.get("session_date"), as_of)]
        if not items:
            return RetrievalResult(
                context="",
                token_count=0,
                retrieval_latency_ms=(time.perf_counter() - t0) * 1000.0,
                metadata={"chunks_retrieved": 0, "source_session_ids": []},
            )

        q_emb = self.embedder.embed_text(query)
        scored = [
            (self.embedder.cosine_similarity(q_emb, item["embedding"]), item)
            for item in items
        ]
        scored.sort(key=lambda x: x[0], reverse=True)

        selected_texts: List[str] = []
        source_session_ids: List[str] = []
        seen_sids = set()
        tot_tok = 0
        for _, item in scored:
            text = item["text"]
            c_tok = count_tokens(text)
            if tot_tok + c_tok > token_budget:
                break
            selected_texts.append(text)
            tot_tok += c_tok
            sid = item.get("session_id", "")
            if sid and sid not in seen_sids:
                seen_sids.add(sid)
                source_session_ids.append(sid)

        context = "\n\n---\n\n".join(selected_texts)
        latency = (time.perf_counter() - t0) * 1000.0
        return RetrievalResult(
            context=context,
            token_count=count_tokens(context),
            retrieval_latency_ms=latency,
            metadata={
                "chunks_retrieved": len(selected_texts),
                "source_session_ids": source_session_ids,
            },
        )

    async def delete(
        self,
        tenant_id: str,
        session_id: Optional[str] = None,
        target: Optional[str] = None,
    ) -> Dict[str, Any]:
        items = self._stores.get(tenant_id)
        if items is None:
            return {"supported": True, "deleted": False, "removed_count": 0}

        if session_id is None and target is None:
            removed = len(items)
            del self._stores[tenant_id]
            return {"supported": True, "deleted": removed > 0, "removed_count": removed}

        norm_target = normalize(target) if target else ""
        # Identify sessions containing target so sibling assistant turns in same session are also removed
        target_sids = set()
        if norm_target:
            for it in items:
                if norm_target in normalize(it["text"]):
                    if it.get("session_id"):
                        target_sids.add(it["session_id"])

        kept: List[Dict[str, Any]] = []
        removed = 0
        for it in items:
            sid = it.get("session_id")
            match_sid = session_id is not None and sid == session_id
            match_target = bool(norm_target) and (
                norm_target in normalize(it["text"]) or (sid in target_sids)
            )
            if match_sid or match_target:
                removed += 1
            else:
                kept.append(it)

        self._stores[tenant_id] = kept
        return {"supported": True, "deleted": removed > 0, "removed_count": removed}

    async def cleanup(self, tenant_id: str) -> None:
        self._stores.pop(tenant_id, None)
