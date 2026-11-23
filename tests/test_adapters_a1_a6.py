"""Regression and gate tests for Phase 4 adapter contract and baselines (A1, A2, A3, A5, A6)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import pytest

from epibench.adapters import (
    ADAPTER_CONTRACT_VERSION,
    FullContextUnboundedAdapter,
    HybridRAGAdapter,
    Mem0Adapter,
    NullMemoryAdapter,
    OracleAdapter,
    RecentContextAdapter,
    VectorRAGAdapter,
    count_tokens,
)
from epibench.generator.pools import ALL_POOLS
from epibench.generator.scenario_generator import ScenarioGenerator, parse_session_date
from epibench.judging.programmatic import (
    check_forbidden_phrases,
    score_integer_match,
    score_set_overlap,
    score_string_match,
    score_yes_no,
)
from epibench.runners.fakes import FakeEmbedder, FakeReader

REPO_ROOT = Path(__file__).resolve().parent.parent


class _InMemoryFakeMem0Client:
    """Minimal stand-in implementing mem0 Memory's public add/search/get_all/delete API."""

    def __init__(self):
        self._items: List[Dict[str, Any]] = []
        self._seq = 0

    def add(self, text: str, user_id: str, metadata: Dict[str, Any] | None = None):
        self._seq += 1
        self._items.append({
            "id": f"mem_{self._seq}",
            "user_id": user_id,
            "memory": text,
            "metadata": dict(metadata or {}),
        })

    def search(self, query: str, filters: Dict[str, Any] | None = None):
        uid = (filters or {}).get("user_id")
        return {"results": [it for it in self._items if it["user_id"] == uid]}

    def get_all(self, user_id: str):
        return {"results": [it for it in self._items if it["user_id"] == user_id]}

    def delete(self, memory_id: str):
        self._items = [it for it in self._items if it["id"] != memory_id]

    def delete_all(self, user_id: str):
        self._items = [it for it in self._items if it["user_id"] != user_id]


@pytest.mark.asyncio
async def test_A1_as_of_excludes_future_sessions_all_adapters():
    """A1: Every built-in adapter strictly excludes sessions dated after as_of before ranking."""
    assert ADAPTER_CONTRACT_VERSION == "2.0.0"
    gen = ScenarioGenerator()
    sc = gen.generate_scenario("R", scenario_idx=0, split="dev", seed=20261122, filler_count=6)
    as_of_q = next(q for q in sc.questions if q.question_subkind == "as_of")
    as_of_date = as_of_q.metadata["as_of_date"]
    session_dates_by_id = {s["session_id"]: parse_session_date(s["session_date"]) for s in sc.sessions}
    cutoff_dt = parse_session_date(as_of_date).replace(hour=23, minute=59, second=59)

    embedder = FakeEmbedder()
    adapters = [
        RecentContextAdapter(),
        FullContextUnboundedAdapter(),
        VectorRAGAdapter(embedder=embedder),
        HybridRAGAdapter(embedder=embedder),
        OracleAdapter(),
        NullMemoryAdapter(),
        Mem0Adapter(api_key="unused", memory_client=_InMemoryFakeMem0Client()),
    ]

    for adapter in adapters:
        tid = f"t_{adapter.system_name}"
        await adapter.ingest(tid, sc.sessions)
        ret = await adapter.retrieve(tid, as_of_q.question_text, token_budget=1500, as_of=as_of_date)

        # Future values (CockroachDB, PostgreSQL 16) must NEVER appear in retrieved context
        assert "CockroachDB" not in ret.context, f"{adapter.system_name} leaked future CockroachDB"
        assert "PostgreSQL 16" not in ret.context, f"{adapter.system_name} leaked future PostgreSQL 16"

        # Every returned source_session_id must satisfy date <= as_of
        sids = ret.metadata.get("source_session_ids")
        assert isinstance(sids, list)
        for sid in sids:
            assert session_dates_by_id[sid] <= cutoff_dt, (
                f"{adapter.system_name} returned future session {sid}"
            )
        if adapter.system_name != "null_memory":
            assert "PostgreSQL 14" in ret.context, f"{adapter.system_name} missed valid Jan value"
        await adapter.cleanup(tid)


@pytest.mark.asyncio
async def test_A2_recent_context_keeps_newest_contiguous_sessions():
    """A2: RecentContextAdapter retains newest contiguous sessions (including PostgreSQL 16 at depth 60)."""
    gen = ScenarioGenerator()
    sc = gen.generate_scenario("R", scenario_idx=0, split="dev", seed=20261122, filler_count=60)

    recent = RecentContextAdapter()
    unbounded = FullContextUnboundedAdapter()
    await recent.ingest("t_recent", sc.sessions)
    await unbounded.ingest("t_unbounded", sc.sessions)

    # Standard benchmark budget (1500 tokens) at depth 60 where old oldest-first lost PostgreSQL 16
    r_std = await recent.retrieve("t_recent", "current database?", token_budget=1500)
    assert r_std.token_count <= 1500
    assert "PostgreSQL 16" in r_std.context, "RecentContextAdapter dropped the newest evidence session!"

    # Returned session IDs must form a contiguous suffix of sc.sessions in chronological order
    all_sids = [s["session_id"] for s in sc.sessions]
    ret_sids = r_std.metadata["source_session_ids"]
    assert 1 <= len(ret_sids) < len(all_sids)
    assert all_sids[-len(ret_sids):] == ret_sids, "RecentContextAdapter output is not a contiguous newest suffix"

    # Unbounded upper reference retains all 63 sessions
    r_all = await unbounded.retrieve("t_unbounded", "current database?", token_budget=1500)
    assert len(r_all.metadata["source_session_ids"]) == len(sc.sessions)
    assert "PostgreSQL 14" in r_all.context and "PostgreSQL 16" in r_all.context


@pytest.mark.asyncio
async def test_A3_targeted_delete_removes_secret_preserves_control():
    """A3: delete(target=...) removes the secret from vector, hybrid, recent, and oracle while keeping control."""
    gen = ScenarioGenerator()
    sc = gen.generate_scenario("E", scenario_idx=0, split="dev", seed=20261122, filler_count=6)
    target = sc.metadata["target_item"]  # sk-live-alpha-8821
    q_direct = next(q for q in sc.questions if q.question_subkind == "deletion_direct")
    q_ctrl = next(q for q in sc.questions if q.question_subkind == "deletion_control_retention")

    embedder = FakeEmbedder()
    adapters = [
        VectorRAGAdapter(embedder=embedder),
        HybridRAGAdapter(embedder=embedder),
        RecentContextAdapter(),
        OracleAdapter(),
    ]

    for adapter in adapters:
        tid = f"del_{adapter.system_name}"
        await adapter.ingest(tid, sc.sessions)
        del_res = await adapter.delete(tid, target=target)
        assert del_res["supported"] is True
        assert del_res["deleted"] is True

        # Secret must NOT appear in direct retrieval
        ret_sec = await adapter.retrieve(tid, q_direct.question_text, token_budget=1500)
        leak = check_forbidden_phrases(ret_sec.context, q_direct.metadata["forbidden_aliases"])
        assert leak["contaminated"] is False, f"{adapter.system_name} leaked deleted secret: {leak}"

        # Unrelated control fact must STILL be retained
        ret_ctrl = await adapter.retrieve(tid, q_ctrl.question_text, token_budget=1500)
        assert q_ctrl.gold_answer in ret_ctrl.context, f"{adapter.system_name} lost control fact on targeted delete"
        await adapter.cleanup(tid)


@pytest.mark.asyncio
async def test_A5_oracle_and_null_memory_reference_bounds():
    """A5: NullMemoryAdapter returns empty; OracleAdapter + FakeReader passes every programmatic question."""
    null_adapter = NullMemoryAdapter()
    oracle = OracleAdapter()
    reader = FakeReader()
    gen = ScenarioGenerator()

    total_prog = 0
    oracle_correct = 0

    for track, pool in ALL_POOLS.items():
        for idx in range(len(pool)):
            sc = gen.generate_scenario(track, scenario_idx=idx, split="dev", seed=20261122, filler_count=4)
            tid_null = f"null_{sc.scenario_id}"
            tid_ora = f"ora_{sc.scenario_id}"

            await null_adapter.ingest(tid_null, sc.sessions)
            # Replay chronologically for oracle, triggering delete on deletion event
            for s in sc.sessions:
                await oracle.ingest(tid_ora, [s])
                if s.get("is_deletion_event"):
                    await oracle.delete(tid_ora, target=s.get("deleted_target"))

            for q in sc.questions:
                as_of_arg = q.metadata.get("as_of_date")

                r_null = await null_adapter.retrieve(tid_null, q.question_text, token_budget=1500, as_of=as_of_arg)
                assert r_null.context == ""
                assert r_null.token_count == 0

                if q.scoring_type in ("string_match", "integer_match", "yes_no", "set_overlap"):
                    total_prog += 1
                    r_ora = await oracle.retrieve(tid_ora, q.question_text, token_budget=1500, as_of=as_of_arg)
                    hyp = await reader.answer_question(
                        r_ora.context, q.question_text, question_date=sc.question_date, scoring_type=q.scoring_type
                    )
                    if q.scoring_type == "string_match":
                        ok = score_string_match(hyp, str(q.gold_answer), aliases=q.metadata.get("gold_aliases"))["is_correct"]
                    elif q.scoring_type == "integer_match":
                        ok = score_integer_match(hyp, int(q.gold_answer))["is_correct"]
                    elif q.scoring_type == "yes_no":
                        ok = score_yes_no(hyp, str(q.gold_answer))["is_correct"]
                    else:
                        ok = score_set_overlap(hyp, q.metadata["gold_decisions"], q.metadata["unaffected_decisions"])["is_correct"]
                    assert ok is True, f"Oracle + FakeReader failed on {q.question_id}: hyp={hyp!r}, gold={q.gold_answer!r}"
                    oracle_correct += 1

            await oracle.cleanup(tid_ora)

    assert total_prog > 100
    assert oracle_correct == total_prog


@pytest.mark.asyncio
async def test_A6_mem0_no_monkeypatch_as_of_and_delete():
    """A6: Mem0Adapter does not monkeypatch internal clients, passes session dates, honors as_of and delete."""
    mem0_src = (REPO_ROOT / "src" / "epibench" / "adapters" / "mem0_adapter.py").read_text(encoding="utf-8")
    assert "generate_content" not in mem0_src
    assert "metered_gen" not in mem0_src

    client = _InMemoryFakeMem0Client()
    adapter = Mem0Adapter(api_key="unused", memory_client=client)
    sessions = [
        {"session_id": "s1", "session_date": "2024-01-15 10:00:00", "turns": [{"role": "user", "content": "Secret key sk-test-999"}]},
        {"session_id": "s2", "session_date": "2024-03-15 10:00:00", "turns": [{"role": "user", "content": "Office is in Denver"}]},
    ]
    ing = await adapter.ingest("u1", sessions)
    assert ing.metadata.get("internal_llm_cost_unmetered") is True

    # as_of Jan 25 must exclude March session
    r_jan = await adapter.retrieve("u1", "office", as_of="2024-01-25")
    assert "Denver" not in r_jan.context
    assert "sk-test-999" in r_jan.context

    # Targeted delete removes only sk-test-999 and retains Denver
    d_res = await adapter.delete("u1", target="sk-test-999")
    assert d_res["deleted"] is True
    r_after = await adapter.retrieve("u1", "everything")
    assert "sk-test-999" not in r_after.context
    assert "Denver" in r_after.context


def test_unified_token_counter_used_everywhere():
    """Verify count_tokens behavior and that no adapter defines an ad-hoc '// 4' estimator."""
    assert count_tokens("") == 0
    assert count_tokens("   ") == 0
    assert count_tokens("abcd") == 1
    assert count_tokens("abcde") == 2

    adapters_dir = REPO_ROOT / "src" / "epibench" / "adapters"
    for py_file in adapters_dir.glob("*.py"):
        if py_file.name == "base.py":
            continue
        src = py_file.read_text(encoding="utf-8")
        assert "// 4" not in src, f"Ad-hoc '// 4' token counting left in {py_file.name}"
