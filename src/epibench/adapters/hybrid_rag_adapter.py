"""Hybrid RAG Adapter.

Combines sparse Okapi BM25 lexical search with dense FastEmbed vector similarity,
fused via Reciprocal Rank Fusion (RRF). Enforces `as_of` filtering BEFORE both
BM25 indexing and dense ranking, supports targeted deletion, and uses the unified
token budget counter.
"""

from __future__ import annotations

import math
import re
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


class BM25Index:
    """Okapi BM25 implementation in pure Python."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.doc_count: int = 0
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.term_doc_freqs: Dict[str, int] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())

    def index(self, documents: List[str]) -> None:
        self.doc_count = len(documents)
        self.doc_lengths = []
        self.doc_term_freqs = []
        self.term_doc_freqs = {}

        if not documents:
            self.avg_doc_length = 0.0
            return

        total_tokens = 0
        for doc in documents:
            tokens = self.tokenize(doc)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_tokens += length

            tf: Dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            self.doc_term_freqs.append(tf)

            for tok in tf.keys():
                self.term_doc_freqs[tok] = self.term_doc_freqs.get(tok, 0) + 1

        self.avg_doc_length = total_tokens / max(1, self.doc_count)

    def score(self, query: str) -> List[float]:
        if self.doc_count == 0:
            return []

        q_tokens = self.tokenize(query)
        scores = [0.0] * self.doc_count

        for tok in q_tokens:
            df = self.term_doc_freqs.get(tok, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))

            for idx in range(self.doc_count):
                tf = self.doc_term_freqs[idx].get(tok, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lengths[idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avg_doc_length)))
                scores[idx] += idf * ((tf * (self.k1 + 1.0)) / denom)

        return scores


class HybridRAGAdapter(BaseMemoryAdapter):
    """Hybrid RAG combining BM25 lexical search and dense embeddings with RRF reranking."""

    def __init__(
        self,
        system_name: str = "hybrid_rag",
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
        # Strictly filter by as_of BEFORE indexing BM25 or computing dense similarity
        items = [it for it in all_items if is_on_or_before_as_of(it.get("session_date"), as_of)]
        if not items:
            return RetrievalResult(
                context="",
                token_count=0,
                retrieval_latency_ms=(time.perf_counter() - t0) * 1000.0,
                metadata={"chunks_retrieved": 0, "source_session_ids": []},
            )

        texts = [it["text"] for it in items]
        bm25 = BM25Index()
        bm25.index(texts)

        # 1. Lexical BM25 ranking
        bm25_scores = bm25.score(query)
        bm25_ranked = sorted(range(len(items)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_rank_map = {idx: rank for rank, idx in enumerate(bm25_ranked)}

        # 2. Dense vector ranking
        q_emb = self.embedder.embed_text(query)
        dense_scores = [self.embedder.cosine_similarity(q_emb, it["embedding"]) for it in items]
        dense_ranked = sorted(range(len(items)), key=lambda i: dense_scores[i], reverse=True)
        dense_rank_map = {idx: rank for rank, idx in enumerate(dense_ranked)}

        # 3. Reciprocal Rank Fusion (RRF, k=60)
        rrf_k = 60.0
        rrf_scored = []
        for i in range(len(items)):
            score = (1.0 / (rrf_k + bm25_rank_map[i])) + (1.0 / (rrf_k + dense_rank_map[i]))
            rrf_scored.append((score, items[i]))
        rrf_scored.sort(key=lambda x: x[0], reverse=True)

        selected_texts: List[str] = []
        source_session_ids: List[str] = []
        seen_sids = set()
        tot_tok = 0
        for _, item in rrf_scored:
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
