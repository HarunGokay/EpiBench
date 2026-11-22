"""Relational Graph Memory Adapter.

Evaluates a relational graph memory architecture with temporal ledgers:
- Multi-session ingestion with tenant isolation
- Semantic extraction & relational graph compilation
- Point-in-time state reconstruction (as_of query support)
- History-aware retrieval for superseded state queries
- True database account erasure support
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional
import uuid

from sqlalchemy import func, select

try:
    from app.sigkimeno.content_store import create_content_store
    from app.sigkimeno.context_engine import context_retriever
    from app.sigkimeno.database import Database
    from app.sigkimeno.erase_service import delete_user_account
    from app.sigkimeno.large_input import split_conversation_turns
    from app.sigkimeno.models import ProcessingJob, User, UserState
    from app.sigkimeno.providers import create_provider
    from app.sigkimeno.service import enqueue_source
    from app.sigkimeno.settings import settings
    from app.sigkimeno.worker import claim_job, process_claim
    HAS_GRAPH_BACKEND = True
except ImportError:
    HAS_GRAPH_BACKEND = False

from EpiBench.adapters.base import BaseMemoryAdapter, IngestResult, RetrievalResult


def parse_timestamp(val: Any) -> datetime:
    if isinstance(val, datetime):
        if val.tzinfo is None:
            return val.replace(tzinfo=timezone.utc)
        return val.astimezone(timezone.utc)
    s = re.sub(r"\([A-Za-z]+\)", "", str(val)).strip()
    s = re.sub(r"\s+", " ", s)
    formats = [
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


class RelationalGraphAdapter(BaseMemoryAdapter):
    """Adapter for Relational Graph Memory architecture with temporal ledger."""

    def __init__(
        self,
        database: Optional[Any] = None,
        system_name: str = "graph_memory",
        spend_tracker: Optional[Any] = None,
    ):
        super().__init__(system_name)
        if not HAS_GRAPH_BACKEND:
            raise ImportError(
                "Graph memory backend dependencies are not available in this environment. "
                "Ensure required database dependencies are installed to evaluate this adapter."
            )
        self.db = database or Database(settings.database_url)
        self.storage = create_content_store(settings)
        self.provider = create_provider(settings)
        self.spend_tracker = spend_tracker
        self._tenant_uuids: Dict[str, uuid.UUID] = {}

    def _get_uuid(self, tenant_id: str) -> uuid.UUID:
        if tenant_id not in self._tenant_uuids:
            try:
                self._tenant_uuids[tenant_id] = uuid.UUID(tenant_id)
            except ValueError:
                self._tenant_uuids[tenant_id] = uuid.uuid5(uuid.NAMESPACE_DNS, tenant_id)
        return self._tenant_uuids[tenant_id]

    async def ingest(self, tenant_id: str, sessions: List[Dict[str, Any]]) -> IngestResult:
        """Ingest conversational sessions into graph store with background worker processing."""
        uid = self._get_uuid(tenant_id)
        t0 = time.perf_counter()

        async with self.db.sessions() as sess:
            user = await sess.get(User, uid)
            if not user:
                user = User(
                    id=uid,
                    tenant_id=tenant_id,
                    tier="free",
                    monthly_token_allowance=10_000_000,
                    tokens_used_this_month=0,
                )
                sess.add(user)
                user_state = UserState(user_id=uid)
                sess.add(user_state)
                await sess.commit()

        total_turns = 0
        total_chars = 0
        job_ids = []

        for s in sessions:
            turns = s.get("turns", [])
            total_turns += len(turns)
            s_time = parse_timestamp(s.get("date_str") or s.get("timestamp") or datetime.now(timezone.utc))

            convo_text = ""
            for turn in turns:
                role = turn.get("role", "user")
                txt = turn.get("content") or turn.get("text") or ""
                convo_text += f"{role.capitalize()}: {txt}\n"
            total_chars += len(convo_text)

            chunks = split_conversation_turns(convo_text, max_chars=4000)
            for chunk_idx, chunk_text in enumerate(chunks):
                async with self.db.sessions() as sess:
                    job = await enqueue_source(
                        session=sess,
                        storage=self.storage,
                        user_id=uid,
                        source_type="chat",
                        payload={"text": chunk_text, "session_id": s.get("session_id", "s")},
                        source_timestamp=s_time,
                        source_id=f"{tenant_id}_{s.get('session_id', 's')}_{chunk_idx}",
                    )
                    await sess.commit()
                    job_ids.append(job.id)

        # Process all queued jobs synchronously
        for job_id in job_ids:
            async with self.db.sessions() as sess:
                claim = await claim_job(sess)
                if claim:
                    await process_claim(
                        claim=claim,
                        session=sess,
                        storage=self.storage,
                        provider=self.provider,
                        settings=settings,
                    )
                    await sess.commit()

        wall_time = time.perf_counter() - t0

        # Query provider tokens used
        async with self.db.sessions() as sess:
            res = await sess.execute(
                select(
                    func.sum(ProcessingJob.extraction_input_tokens),
                    func.sum(ProcessingJob.extraction_output_tokens),
                ).where(ProcessingJob.user_id == uid)
            )
            row = res.one_or_none()
            in_tokens = int(row[0] or 0) if row else 0
            out_tokens = int(row[1] or 0) if row else 0

        if self.spend_tracker and (in_tokens > 0 or out_tokens > 0):
            self.spend_tracker.record(in_tokens, out_tokens, stage="graph_extraction")

        return IngestResult(
            tenant_id=tenant_id,
            session_count=len(sessions),
            latency_ms=wall_time * 1000,
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            metadata={
                "total_turns": total_turns,
                "job_count": len(job_ids),
            },
        )

    async def retrieve(
        self,
        tenant_id: str,
        query: str,
        token_budget: int = 1500,
        as_of: Optional[str] = None,
    ) -> RetrievalResult:
        """Point-in-time graph-structured memory retrieval."""
        t0 = time.perf_counter()
        uid = self._get_uuid(tenant_id)

        as_of_dt = parse_timestamp(as_of) if as_of else None

        async with self.db.sessions() as sess:
            ctx_result = await context_retriever.retrieve_context(
                session=sess,
                user_id=uid,
                query=query,
                token_budget=token_budget,
                as_of=as_of_dt,
            )

        wall_time = time.perf_counter() - t0

        packed_text = ctx_result.text or ""
        token_count = ctx_result.tokens_used or (len(packed_text) // 4)

        return RetrievalResult(
            context=packed_text,
            token_count=token_count,
            retrieval_latency_ms=wall_time * 1000,
            metadata={
                "as_of": as_of_dt.isoformat() if as_of_dt else None,
                "node_count": len(ctx_result.nodes) if hasattr(ctx_result, "nodes") else 0,
            },
        )

    async def delete(self, tenant_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Tenant memory erasure."""
        uid = self._get_uuid(tenant_id)
        async with self.db.sessions() as sess:
            res = await delete_user_account(session=sess, user_id=uid, storage=self.storage)
            await sess.commit()
            return {"supported": True, "deleted": res.get("deleted", True)}

    async def cleanup(self, tenant_id: str) -> None:
        """Clean up tenant state in database."""
        try:
            await self.delete(tenant_id)
        except Exception:
            pass
