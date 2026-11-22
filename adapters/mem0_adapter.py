"""Mem0 Memory Adapter.

Open-source Mem0 framework using local embedded Qdrant vector store,
fastembed (BAAI/bge-small-en-v1.5) embeddings, and Gemini 3.5 Flash-Lite LLM.
Metered under identical spend controls and token budget.
Uses singleton client with user_id multi-tenant scoping.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional

from EpiBench.adapters.base import BaseMemoryAdapter, IngestResult, RetrievalResult


class Mem0Adapter(BaseMemoryAdapter):
    """Adapter for Mem0 memory architecture with isolated user_id partitioning."""

    _shared_memory: Optional[Any] = None
    _shared_storage_dir: Optional[str] = None

    def __init__(
        self,
        api_key: str,
        system_name: str = "mem0",
        base_storage_dir: Optional[Path] = None,
        spend_tracker: Optional[Any] = None,
    ):
        super().__init__(system_name)
        self.api_key = api_key
        self.spend_tracker = spend_tracker
        self.base_storage_dir = base_storage_dir

    def _get_memory(self) -> Any:
        if Mem0Adapter._shared_memory is not None:
            return Mem0Adapter._shared_memory

        if self.base_storage_dir:
            db_path = str(self.base_storage_dir)
            os.makedirs(db_path, exist_ok=True)
        else:
            db_path = tempfile.mkdtemp(prefix="mem0_shared_qdrant_")

        Mem0Adapter._shared_storage_dir = db_path

        from mem0 import Memory

        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": db_path,
                    "embedding_model_dims": 384,
                },
            },
            "llm": {
                "provider": "gemini",
                "config": {
                    "model": "gemini-3.5-flash-lite",
                    "api_key": self.api_key,
                    "temperature": 0.0,
                },
            },
            "embedder": {
                "provider": "fastembed",
                "config": {
                    "model": "BAAI/bge-small-en-v1.5",
                },
            },
        }

        m = Memory.from_config(config)

        # Meter internal LLM extraction calls if spend_tracker is attached
        if self.spend_tracker:
            orig_gen = m.llm.client.models.generate_content

            def metered_gen(*args, **kwargs):
                res = orig_gen(*args, **kwargs)
                if hasattr(res, "usage_metadata") and res.usage_metadata:
                    m_in = getattr(res.usage_metadata, "prompt_token_count", 0)
                    m_out = getattr(res.usage_metadata, "candidates_token_count", 0)
                    if m_in > 0 or m_out > 0:
                        self.spend_tracker.record(m_in, m_out, stage="mem0_extraction")
                return res

            m.llm.client.models.generate_content = metered_gen

        Mem0Adapter._shared_memory = m
        return m

    async def ingest(self, tenant_id: str, sessions: List[Dict[str, Any]]) -> IngestResult:
        t0 = time.perf_counter()
        m = self._get_memory()

        in_before = self.spend_tracker.total_input_tokens if self.spend_tracker else 0
        out_before = self.spend_tracker.total_output_tokens if self.spend_tracker else 0
        spend_before = self.spend_tracker.total_spend_usd if self.spend_tracker else 0.0

        for s in sessions:
            turns = s.get("turns", [])
            s_date = s.get("session_date", "")
            sess_str = f"Date: {s_date}\n" + "\n".join(
                f"{t.get('role', 'user').capitalize()}: {t.get('content') or t.get('text') or ''}" for t in turns
            )
            m.add(sess_str, user_id=tenant_id)

        latency = (time.perf_counter() - t0) * 1000
        in_tok = (self.spend_tracker.total_input_tokens - in_before) if self.spend_tracker else 0
        out_tok = (self.spend_tracker.total_output_tokens - out_before) if self.spend_tracker else 0
        cost = (self.spend_tracker.total_spend_usd - spend_before) if self.spend_tracker else 0.0

        return IngestResult(
            tenant_id=tenant_id,
            session_count=len(sessions),
            latency_ms=latency,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
        )

    async def retrieve(
        self,
        tenant_id: str,
        query: str,
        token_budget: int = 1500,
        as_of: Optional[str] = None,
    ) -> RetrievalResult:
        t0 = time.perf_counter()
        m = self._get_memory()

        search_res = m.search(query, filters={"user_id": tenant_id})
        retrieval_latency = (time.perf_counter() - t0) * 1000

        results_list = search_res.get("results", []) if isinstance(search_res, dict) else search_res
        all_mem_texts = [r.get("memory", "") for r in results_list if isinstance(r, dict)]

        selected = []
        tot_tok = 0
        for txt in all_mem_texts:
            c_tok = max(1, len(txt) // 4)
            if tot_tok + c_tok > token_budget:
                break
            selected.append(txt)
            tot_tok += c_tok

        context = "\n\n".join(selected)
        return RetrievalResult(
            context=context,
            token_count=tot_tok,
            retrieval_latency_ms=retrieval_latency,
            metadata={"mems_count": len(selected)},
        )

    async def delete(self, tenant_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        m = self._get_memory()
        try:
            m.delete_all(user_id=tenant_id)
            return {"supported": True, "deleted": True}
        except Exception as e:
            return {"supported": True, "deleted": False, "error": str(e)}

    async def cleanup(self, tenant_id: str) -> None:
        await self.delete(tenant_id)
