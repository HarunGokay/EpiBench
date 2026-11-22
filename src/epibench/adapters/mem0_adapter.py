"""Mem0 Memory Adapter.

Open-source Mem0 framework using local embedded Qdrant vector store,
fastembed (BAAI/bge-small-en-v1.5) embeddings, and configured LLM.

Notes:
- Does NOT monkey-patch Mem0's internal LLM client; Mem0's internal extraction
  LLM calls are therefore flagged `internal_llm_cost_unmetered=True` in metadata.
- Stores `session_date` and `session_id` in Mem0 metadata and filters retrieved
  memories strictly by `as_of` when an `as_of` cutoff is specified.
- Supports targeted deletion by `session_id` or `target` string via Mem0's public
  `get_all` + `delete` APIs.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Optional

from epibench.adapters.base import (
    BaseMemoryAdapter,
    IngestResult,
    RetrievalResult,
    count_tokens,
    is_on_or_before_as_of,
)
from epibench.judging.programmatic import normalize


class Mem0Adapter(BaseMemoryAdapter):
    """Adapter for Mem0 memory architecture with isolated user_id partitioning."""

    _shared_memory: Optional[Any] = None
    _shared_storage_dir: Optional[str] = None

    def __init__(
        self,
        api_key: str,
        system_name: str = "mem0",
        model_name: str = "gemini-3.5-flash-lite",
        base_storage_dir: Optional[Path] = None,
        spend_tracker: Optional[Any] = None,
        memory_client: Optional[Any] = None,
    ):
        super().__init__(system_name)
        self.api_key = api_key
        self.model_name = model_name
        self.spend_tracker = spend_tracker
        self.base_storage_dir = base_storage_dir
        self._injected_client = memory_client

    def _get_memory(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
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
                    "model": self.model_name,
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
        Mem0Adapter._shared_memory = m
        return m

    async def ingest(self, tenant_id: str, sessions: List[Dict[str, Any]]) -> IngestResult:
        t0 = time.perf_counter()
        m = self._get_memory()

        for s in sessions:
            turns = s.get("turns", [])
            s_date = s.get("session_date", "")
            s_id = s.get("session_id", "")
            sess_str = f"[Date: {s_date}]\n" + "\n".join(
                f"{str(t.get('role', 'user')).capitalize()}: {t.get('content') or t.get('text') or ''}"
                for t in turns
            )
            m.add(
                sess_str,
                user_id=tenant_id,
                metadata={"session_date": s_date, "session_id": s_id},
            )

        latency = (time.perf_counter() - t0) * 1000.0
        return IngestResult(
            tenant_id=tenant_id,
            session_count=len(sessions),
            latency_ms=latency,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            metadata={"internal_llm_cost_unmetered": True},
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
        results_list = search_res.get("results", []) if isinstance(search_res, dict) else (search_res or [])

        selected: List[str] = []
        source_session_ids: List[str] = []
        seen_sids = set()
        tot_tok = 0

        for r in results_list:
            if not isinstance(r, dict):
                continue
            meta = r.get("metadata") or {}
            s_date = meta.get("session_date")
            if as_of is not None:
                # Filter out memories dated after as_of (or missing session_date when as_of enforced)
                if not is_on_or_before_as_of(s_date, as_of):
                    continue
            txt = r.get("memory", "")
            if not txt:
                continue
            formatted = f"[{s_date}] {txt}" if s_date else txt
            c_tok = count_tokens(formatted)
            if tot_tok + c_tok > token_budget:
                break
            selected.append(formatted)
            tot_tok += c_tok
            sid = meta.get("session_id", "")
            if sid and sid not in seen_sids:
                seen_sids.add(sid)
                source_session_ids.append(sid)

        context = "\n\n".join(selected)
        retrieval_latency = (time.perf_counter() - t0) * 1000.0
        return RetrievalResult(
            context=context,
            token_count=count_tokens(context),
            retrieval_latency_ms=retrieval_latency,
            metadata={
                "mems_count": len(selected),
                "source_session_ids": source_session_ids,
                "internal_llm_cost_unmetered": True,
            },
        )

    async def delete(
        self,
        tenant_id: str,
        session_id: Optional[str] = None,
        target: Optional[str] = None,
    ) -> Dict[str, Any]:
        m = self._get_memory()
        try:
            if session_id is None and target is None:
                m.delete_all(user_id=tenant_id)
                return {"supported": True, "deleted": True}

            all_res = m.get_all(user_id=tenant_id)
            mems = all_res.get("results", []) if isinstance(all_res, dict) else (all_res or [])
            norm_target = normalize(target) if target else ""
            removed = 0
            for r in mems:
                if not isinstance(r, dict):
                    continue
                mem_id = r.get("id")
                meta = r.get("metadata") or {}
                txt = r.get("memory", "")
                match_sid = session_id is not None and meta.get("session_id") == session_id
                match_target = bool(norm_target) and norm_target in normalize(txt)
                if (match_sid or match_target) and mem_id is not None:
                    m.delete(mem_id)
                    removed += 1
            return {"supported": True, "deleted": removed > 0, "removed_count": removed}
        except Exception as e:
            return {"supported": True, "deleted": False, "error": str(e)}

    async def cleanup(self, tenant_id: str) -> None:
        await self.delete(tenant_id)
