"""Hybrid RAG Adapter.

Combines sparse BM25 lexical search with dense vector similarity,
fused via Reciprocal Rank Fusion (RRF) reranking.
Bounded strictly by the identical token budget.
"""

from __future__ import annotations

import math
import re
import time
from typing import Any, Dict, List, Optional

from app.sigkimeno.context_engine import EmbeddingService
from EpiBench.adapters.base import BaseMemoryAdapter, IngestResult, RetrievalResult


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
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.doc_term_freqs.append(tf)

            for t in tf.keys():
                self.term_doc_freqs[t] = self.term_doc_freqs.get(t, 0) + 1

        self.avg_doc_length = total_tokens / max(1, self.doc_count)

    def score(self, query: str) -> List[float]:
        q_tokens = self.tokenize(query)
        scores = [0.0] * self.doc_count
        if self.doc_count == 0 or not q_tokens:
            return scores

        for t in q_tokens:
            df = self.term_doc_freqs.get(t, 0)
            if df == 0:
                continue
            # Standard Lucene/BM25 IDF
            idf = math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))
            for i in range(self.doc_count):
                tf = self.doc_term_freqs[i].get(t, 0)
                if tf == 0:
                    continue
                num = tf * (self.k1 + 1.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_lengths[i] / max(1.0, self.avg_doc_length)))
                scores[i] += idf * (num / denom)
        return scores


class HybridRAGAdapter(BaseMemoryAdapter):
    """Hybrid RAG combining BM25 lexical search and dense embeddings with RRF reranking."""

    def __init__(self, system_name: str = "hybrid_rag"):
        super().__init__(system_name)
        self.embedder = EmbeddingService.get_instance()
        self._stores: Dict[str, Dict[str, Any]] = {}

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
            self._stores[tenant_id] = {"chunks": [], "embeddings": [], "bm25": BM25Index()}
            return IngestResult(tenant_id=tenant_id, session_count=len(sessions), latency_ms=0.0)

        bm25 = BM25Index()
        bm25.index(chunks)
        embeddings = self.embedder.embed_batch(chunks)

        self._stores[tenant_id] = {
            "chunks": chunks,
            "embeddings": embeddings,
            "bm25": bm25,
        }
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
        store = self._stores.get(tenant_id)
        if not store or not store["chunks"]:
            return RetrievalResult(context="", token_count=0, retrieval_latency_ms=0.0)

        chunks = store["chunks"]
        embeddings = store["embeddings"]
        bm25: BM25Index = store["bm25"]

        # 1. Lexical BM25 Ranking
        bm25_scores = bm25.score(query)
        bm25_ranked = sorted(range(len(chunks)), key=lambda i: bm25_scores[i], reverse=True)
        bm25_rank_map = {idx: rank for rank, idx in enumerate(bm25_ranked)}

        # 2. Dense Semantic Ranking
        q_emb = self.embedder.embed_text(query)
        dense_scores = [self.embedder.cosine_similarity(q_emb, emb) for emb in embeddings]
        dense_ranked = sorted(range(len(chunks)), key=lambda i: dense_scores[i], reverse=True)
        dense_rank_map = {idx: rank for rank, idx in enumerate(dense_ranked)}

        # 3. Reciprocal Rank Fusion (RRF) Reranker (k=60)
        rrf_k = 60
        rrf_scores = []
        for i in range(len(chunks)):
            r_bm25 = bm25_rank_map[i]
            r_dense = dense_rank_map[i]
            score = (1.0 / (rrf_k + r_bm25)) + (1.0 / (rrf_k + r_dense))
            rrf_scores.append((score, chunks[i]))

        rrf_scores.sort(key=lambda x: x[0], reverse=True)

        selected = []
        tot_tok = 0
        for _, text in rrf_scores:
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
