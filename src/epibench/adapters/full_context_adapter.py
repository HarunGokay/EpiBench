"""Recent-Context (budget-limited) and Full-Context-Unbounded Baseline Adapters.

- RecentContextAdapter (`recent_context`, aliased as `FullContextAdapter` for compatibility):
  Honors `as_of` filtering, keeps the NEWEST contiguous sessions that fit within
  `token_budget`, and returns them in chronological order (truncating oldest first).
- FullContextUnboundedAdapter (`full_context_unbounded`):
  Upper-reference baseline that honors `as_of` and `delete` with no token truncation.
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


class RecentContextAdapter(BaseMemoryAdapter):
    """Budget-limited context baseline retaining the newest contiguous turns in chronological order."""

    def __init__(self, system_name: str = "recent_context", enforce_budget: bool = True):
        super().__init__(system_name)
        self.enforce_budget = enforce_budget
        self._stores: Dict[str, List[Dict[str, Any]]] = {}

    async def ingest(self, tenant_id: str, sessions: List[Dict[str, Any]]) -> IngestResult:
        t0 = time.perf_counter()
        existing = self._stores.setdefault(tenant_id, [])
        existing.extend(sessions)
        existing.sort(key=lambda s: (s.get("session_date", ""), s.get("session_id", "")))
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
        sessions = filter_sessions_as_of(self._stores.get(tenant_id, []), as_of)
        sessions.sort(key=lambda s: (s.get("session_date", ""), s.get("session_id", "")))

        if not self.enforce_budget:
            blocks = [format_session_text(s) for s in sessions]
            selected_ids = [s.get("session_id", f"idx_{i}") for i, s in enumerate(sessions)]
            context = "\n\n".join(b for b in blocks if b)
            tot_tok = count_tokens(context)
            latency = (time.perf_counter() - t0) * 1000.0
            return RetrievalResult(
                context=context,
                token_count=tot_tok,
                retrieval_latency_ms=latency,
                metadata={
                    "source_session_ids": selected_ids,
                    "sessions_count": len(selected_ids),
                    "truncation_mode": "unbounded",
                },
            )

        # Pack NEWEST contiguous whole sessions first, then reverse back to chronological order
        selected_rev: List[tuple[str, str]] = []
        tot_tok = 0
        for idx in range(len(sessions) - 1, -1, -1):
            s = sessions[idx]
            sid = s.get("session_id", f"idx_{idx}")
            block = format_session_text(s)
            if not block:
                continue
            b_tok = count_tokens(block)
            if tot_tok + b_tok > token_budget:
                # If nothing fits yet, keep the newest turns of this latest session
                if not selected_rev:
                    s_date = s.get("session_date", "")
                    turn_lines_rev: List[str] = []
                    for turn in reversed(s.get("turns", [])):
                        role = str(turn.get("role", "user")).capitalize()
                        content = turn.get("content") or turn.get("text") or ""
                        line = f"[{s_date}] {role}: {content}"
                        l_tok = count_tokens(line)
                        if tot_tok + l_tok > token_budget:
                            break
                        turn_lines_rev.append(line)
                        tot_tok += l_tok
                    if turn_lines_rev:
                        turn_lines_rev.reverse()
                        selected_rev.append((sid, "\n".join(turn_lines_rev)))
                break
            selected_rev.append((sid, block))
            tot_tok += b_tok

        selected_rev.reverse()
        selected_ids = [sid for sid, _ in selected_rev]
        context = "\n\n".join(block for _, block in selected_rev)
        actual_tok = count_tokens(context)
        latency = (time.perf_counter() - t0) * 1000.0
        return RetrievalResult(
            context=context,
            token_count=actual_tok,
            retrieval_latency_ms=latency,
            metadata={
                "source_session_ids": selected_ids,
                "sessions_count": len(selected_ids),
                "truncation_mode": "newest_first",
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


class FullContextUnboundedAdapter(RecentContextAdapter):
    """Upper-reference baseline returning all as_of-eligible sessions without token truncation."""

    def __init__(self, system_name: str = "full_context_unbounded"):
        super().__init__(system_name=system_name, enforce_budget=False)


# Backwards-compatible name mapping to RecentContextAdapter (budget-limited, newest-first)
class FullContextAdapter(RecentContextAdapter):
    """Alias for RecentContextAdapter keeping newest turns within token_budget."""

    def __init__(self, system_name: str = "recent_context"):
        super().__init__(system_name=system_name, enforce_budget=True)
