"""Zero-cost ($0.00) offline retrieval & epistemic-hygiene evaluation runner.

Evaluates any built-in or custom BaseMemoryAdapter across dev, hard, or stress
profiles without making a single LLM API call. Directly measures the epistemic
failure modes that flat chunk-based RAG hides behind reader LLMs:

1. Gold-Session Recall@Budget (overall & per-track)
2. Current-State Supersession Precision (1 - Stale-Value Leak Rate on Track R current)
3. Multi-Hop Bridge & Terminal Recall (Track D 3-hop cascades: s_hop2_bridge + s_hop3_terminal)
4. Zero-Overlap Constraint Recall (Track X at both primary budget and tight 600-token budget)
5. Overflow Chain Coverage (Track R 28-step / Track F 24-step / Track I 16-session overflow)
6. Signal-to-Noise Density Ratio (evidence tokens / total context tokens)
7. Track E Selective Erasure (Leak Rate & Control Retention)
8. Strict Bitemporal as_of Compliance, Token Budget Compliance, and Latency (p50/p95)
"""

from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence

from epibench.adapters.base import (
    ADAPTER_CONTRACT_VERSION,
    BaseMemoryAdapter,
    count_tokens,
    format_session_text,
    parse_as_of_cutoff,
    parse_session_date,
)
from epibench.adapters.full_context_adapter import (
    FullContextUnboundedAdapter,
    RecentContextAdapter,
)
from epibench.adapters.hybrid_rag_adapter import HybridRAGAdapter
from epibench.adapters.reference_adapters import NullMemoryAdapter, OracleAdapter
from epibench.adapters.vector_rag_adapter import VectorRAGAdapter
from epibench.embedding import EmbeddingService
from epibench.generator.manifest import canonical_json_bytes, scenario_sha256
from epibench.generator.pools import validate_pools
from epibench.generator.scenario_generator import (
    BENCHMARK_PROFILES,
    GENERATOR_VERSION,
    VALID_TRACKS,
    Scenario,
    ScenarioGenerator,
    resolve_profile,
)
from epibench.judging.programmatic import check_forbidden_phrases, normalize
from epibench.runners.fakes import FakeEmbedder


def _pct(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(len(ordered) * q))
    return round(float(ordered[idx]), 2)


def _required_gold_sessions(scenario: Scenario, q: Any) -> List[str]:
    ev_sessions = [s for s in scenario.sessions if s.get("is_evidence") and not s.get("is_deletion_event")]
    ev_sessions.sort(key=lambda s: (s["session_date"], s["session_id"]))
    track = scenario.track
    sub = q.question_subkind

    if track == "R":
        if sub == "current":
            return [ev_sessions[-1]["session_id"]]
        if sub == "previous":
            return [ev_sessions[-2]["session_id"]] if len(ev_sessions) >= 2 else [ev_sessions[-1]["session_id"]]
        if sub == "as_of":
            cutoff = parse_as_of_cutoff(q.metadata.get("as_of_date"))
            elig = [
                s["session_id"]
                for s in ev_sessions
                if cutoff is None or parse_session_date(s["session_date"]) <= cutoff
            ]
            return [elig[-1]] if elig else [ev_sessions[0]["session_id"]]
        if sub == "count":
            return [s["session_id"] for s in ev_sessions]

    if track == "E":
        if sub in ("deletion_direct", "deletion_semantic"):
            return []
        if sub == "deletion_control_retention":
            return ["s_non_deleted_record"]

    return [s["session_id"] for s in ev_sessions]


def _audit_as_of(scenario: Scenario, as_of_param: Optional[str], source_sids: Sequence[str]) -> bool:
    cutoff = parse_as_of_cutoff(as_of_param)
    if cutoff is None:
        return False
    sess_dates = {s["session_id"]: parse_session_date(s["session_date"]) for s in scenario.sessions}
    return any(sid in sess_dates and sess_dates[sid] > cutoff for sid in source_sids)


async def _ingest_with_deletions(
    adapter: BaseMemoryAdapter,
    tenant_id: str,
    sessions: List[Dict[str, Any]],
) -> None:
    ordered = sorted(sessions, key=lambda s: (s.get("session_date", ""), s.get("session_id", "")))
    has_del = any(s.get("is_deletion_event") for s in ordered)
    if not has_del:
        await adapter.ingest(tenant_id, ordered)
        return

    batch: List[Dict[str, Any]] = []
    for s in ordered:
        batch.append(s)
        if s.get("is_deletion_event"):
            await adapter.ingest(tenant_id, batch)
            batch = []
            await adapter.delete(tenant_id, target=s.get("deleted_target"))
    if batch:
        await adapter.ingest(tenant_id, batch)


class _SharedBatchEmbedder:
    """Bit-identical exact-string embedding cache shared across adapters during ingest only."""

    def __init__(self, inner: Any):
        self._inner = inner
        self._cache: Dict[str, List[float]] = {}
        self.caching_enabled: bool = True

    def embed_text(self, text: str) -> List[float]:
        if self.caching_enabled and text in self._cache:
            return self._cache[text]
        vec = self._inner.embed_text(text)
        if self.caching_enabled:
            self._cache[text] = vec
        return vec

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        if not self.caching_enabled:
            return self._inner.embed_batch(texts)
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            vecs = self._inner.embed_batch(missing)
            for t, v in zip(missing, vecs):
                self._cache[t] = v
        return [self._cache[t] for t in texts]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _build_offline_adapter(name: str, embedder: Optional[Any] = None) -> BaseMemoryAdapter:
    key = name.lower().strip()
    if key == "null_memory":
        return NullMemoryAdapter()
    if key == "oracle":
        return OracleAdapter()
    if key in ("recent_context", "full_context"):
        return RecentContextAdapter(system_name=key)
    if key == "full_context_unbounded":
        return FullContextUnboundedAdapter()
    if key == "vector_rag":
        return VectorRAGAdapter(embedder=embedder)
    if key == "hybrid_rag":
        return HybridRAGAdapter(embedder=embedder)
    raise ValueError(f"Unknown built-in offline adapter: {name!r}")


def _estimate_signal_density(
    scenario: Scenario,
    ret_sids: Sequence[str],
    context_tokens: int,
    ev_token_map: Dict[str, int],
) -> float:
    """Compute fraction (0.0..1.0) of returned context tokens attributable to evidence sessions."""
    if context_tokens <= 0:
        return 0.0
    ev_toks = sum(ev_token_map.get(sid, 0) for sid in set(ret_sids))
    return min(1.0, ev_toks / max(1, context_tokens))


async def run_offline_epistemic_eval(
    *,
    profile: str = "stress",
    tracks: Optional[Sequence[str]] = None,
    systems: Optional[Sequence[str]] = None,
    custom_adapters: Optional[Dict[str, BaseMemoryAdapter]] = None,
    scenarios_per_track: int = 12,
    filler_count: Optional[int] = None,
    distractor_mode: Optional[str] = None,
    distractor_file: Optional[Path] = None,
    token_budget: Optional[int] = None,
    tight_constraint_budget: int = 600,
    seed: int = 20261122,
    split: str = "dev",
    use_fake_embedder: bool = False,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute zero-cost ($0.00) offline retrieval & epistemic-hygiene evaluation."""
    t_start = time.perf_counter()
    validate_pools()

    resolved = resolve_profile(
        profile=profile,
        filler_count=filler_count,
        distractor_mode=distractor_mode,
        token_budget=token_budget,
    )
    eff_profile = resolved["profile"]
    eff_fillers = resolved["filler_count"]
    eff_mode = "custom" if distractor_file is not None else resolved["distractor_mode"]
    eff_budget = resolved["token_budget"]
    eff_stress = resolved["epistemic_stress"]

    track_list = [t.upper() for t in (tracks or VALID_TRACKS)]
    gen = ScenarioGenerator(
        distractor_mode=resolved["distractor_mode"],
        distractor_file=distractor_file,
        epistemic_stress=eff_stress,
    )

    scenarios: List[Scenario] = []
    manifest_entries: List[Dict[str, Any]] = []
    for tr in track_list:
        for idx in range(scenarios_per_track):
            sc = gen.generate_scenario(
                track=tr,
                scenario_idx=idx,
                split=split,
                seed=seed,
                filler_count=eff_fillers,
            )
            scenarios.append(sc)
            sc_dict = sc.to_dict()
            manifest_entries.append({
                "scenario_id": sc.scenario_id,
                "track": sc.track,
                "template_id": sc.metadata.get("template_id", ""),
                "session_count": len(sc.sessions),
                "question_count": len(sc.questions),
                "sha256": scenario_sha256(sc_dict),
            })

    manifest_sha = hashlib.sha256(canonical_json_bytes(manifest_entries)).hexdigest()

    raw_embedder: Any = FakeEmbedder() if use_fake_embedder else EmbeddingService.get_instance()
    embedder = _SharedBatchEmbedder(raw_embedder)
    sys_names = list(systems or ["null_memory", "oracle", "recent_context", "vector_rag", "hybrid_rag"])
    adapters: Dict[str, BaseMemoryAdapter] = {}
    for name in sys_names:
        if custom_adapters and name in custom_adapters:
            adapters[name] = custom_adapters[name]
        else:
            adapters[name] = _build_offline_adapter(name, embedder=embedder)

    acc: Dict[str, Dict[str, Any]] = {
        name: {
            "session_recalls": [],
            "track_session_recalls": defaultdict(list),
            "r_current_clean": 0,
            "r_current_total": 0,
            "d_bridge_recalls": [],
            "d_full_chain_hits": 0,
            "d_full_chain_total": 0,
            "x_primary_hits": 0,
            "x_tight_hits": 0,
            "x_total": 0,
            "overflow_coverages": [],
            "overflow_complete_hits": 0,
            "overflow_total": 0,
            "signal_densities": [],
            "e_leak_hits": 0,
            "e_leak_total": 0,
            "e_control_hits": 0,
            "e_control_total": 0,
            "as_of_violations": 0,
            "budget_overshoots": 0,
            "max_tokens": 0,
            "token_counts": [],
            "latencies_ms": [],
        }
        for name in adapters
    }

    for sc in scenarios:
        ev_token_map = {
            s["session_id"]: count_tokens(format_session_text(s))
            for s in sc.sessions
            if s.get("is_evidence") and not s.get("is_deletion_event")
        }

        for sys_name, adapter in adapters.items():
            tenant_id = f"off_{eff_profile}_{sc.scenario_id}_{sys_name}"
            embedder.caching_enabled = True
            await _ingest_with_deletions(adapter, tenant_id, sc.sessions)
            embedder.caching_enabled = False
            st = acc[sys_name]

            for q in sc.questions:
                as_of_param = q.metadata.get("as_of_date", sc.question_date)
                res = await adapter.retrieve(
                    tenant_id=tenant_id,
                    query=q.question_text,
                    token_budget=eff_budget,
                    as_of=as_of_param,
                )
                ctx = res.context or ""
                tok = count_tokens(ctx)
                ret_sids = list(res.metadata.get("source_session_ids") or [])
                ret_set = set(ret_sids)

                st["token_counts"].append(tok)
                st["max_tokens"] = max(st["max_tokens"], tok)
                if sys_name not in ("oracle", "full_context_unbounded") and tok > eff_budget:
                    st["budget_overshoots"] += 1
                st["latencies_ms"].append(float(res.retrieval_latency_ms))

                if _audit_as_of(sc, as_of_param, ret_sids):
                    st["as_of_violations"] += 1

                if sc.track == "E" and q.question_subkind in ("deletion_direct", "deletion_semantic"):
                    st["e_leak_total"] += 1
                    contam = check_forbidden_phrases(ctx, q.metadata.get("forbidden_aliases", []))
                    if contam["contaminated"]:
                        st["e_leak_hits"] += 1
                    continue

                req_sids = _required_gold_sessions(sc, q)
                if req_sids:
                    rec_frac = sum(1 for sid in req_sids if sid in ret_set) / len(req_sids)
                    st["session_recalls"].append(rec_frac)
                    st["track_session_recalls"][sc.track].append(rec_frac)
                    st["signal_densities"].append(
                        _estimate_signal_density(sc, ret_sids, tok, ev_token_map)
                    )

                # Track R current-state supersession precision (penalizes stale historical leakage!)
                if sc.track == "R" and q.question_subkind == "current":
                    st["r_current_total"] += 1
                    cur_aliases = [str(q.gold_answer)] + [str(a) for a in q.metadata.get("gold_aliases", [])]
                    stale_aliases = list(q.metadata.get("all_stale_aliases") or [])
                    if not stale_aliases and "stale_value" in q.metadata:
                        stale_aliases = [str(q.metadata["stale_value"])] + [str(a) for a in q.metadata.get("stale_aliases", [])]
                    ctx_norm = normalize(ctx)
                    has_current = any(normalize(a) in ctx_norm for a in cur_aliases if a)
                    has_stale = check_forbidden_phrases(ctx, stale_aliases)["contaminated"] if stale_aliases else False
                    if has_current and not has_stale:
                        st["r_current_clean"] += 1

                # Track D bridge-hop & full-chain recall
                if sc.track == "D" and q.question_subkind in ("revisit_set", "why"):
                    bridge_sids = list(q.metadata.get("bridge_sessions") or [])
                    term_sids = list(q.metadata.get("terminal_sessions") or [])
                    hop_sids = list(q.metadata.get("hop_sessions") or req_sids)
                    targets = bridge_sids + term_sids if (bridge_sids or term_sids) else hop_sids
                    if targets:
                        st["d_bridge_recalls"].append(
                            sum(1 for sid in targets if sid in ret_set) / len(targets)
                        )
                    st["d_full_chain_total"] += 1
                    if hop_sids and all(sid in ret_set for sid in hop_sids):
                        st["d_full_chain_hits"] += 1

                # Track X zero-overlap constraint recall at primary budget AND tight 600-token budget
                if sc.track == "X" and q.question_subkind == "hidden_constraint":
                    st["x_total"] += 1
                    kw_aliases = [str(q.metadata.get("constraint_keyword", ""))] + [
                        str(a) for a in q.metadata.get("constraint_aliases", [])
                    ]
                    kw_aliases = [a for a in kw_aliases if a]
                    ctx_norm = normalize(ctx)
                    hit_primary = ("s_constraint" in ret_set) or any(normalize(a) in ctx_norm for a in kw_aliases)
                    if hit_primary:
                        st["x_primary_hits"] += 1

                    # Secondary tight-budget probe
                    res_tight = await adapter.retrieve(
                        tenant_id=tenant_id,
                        query=q.question_text,
                        token_budget=tight_constraint_budget,
                        as_of=as_of_param,
                    )
                    tight_ctx_norm = normalize(res_tight.context or "")
                    tight_sids = set(res_tight.metadata.get("source_session_ids") or [])
                    hit_tight = ("s_constraint" in tight_sids) or any(normalize(a) in tight_ctx_norm for a in kw_aliases)
                    if hit_tight:
                        st["x_tight_hits"] += 1

                # Overflow-chain coverage (Tracks R count, F move_count/slip_days, I portfolio)
                overflow_sids = sc.metadata.get("overflow_evidence_sessions")
                if overflow_sids and (
                    (sc.track == "R" and q.question_subkind == "count")
                    or (sc.track == "F" and q.question_subkind in ("move_count", "slip_days"))
                    or (sc.track == "I" and q.question_subkind == "causal_claim")
                ):
                    cov = sum(1 for sid in overflow_sids if sid in ret_set) / len(overflow_sids)
                    st["overflow_coverages"].append(cov)
                    st["overflow_total"] += 1
                    if cov >= 0.999:
                        st["overflow_complete_hits"] += 1

                # Track E control retention
                if sc.track == "E" and q.question_subkind == "deletion_control_retention":
                    st["e_control_total"] += 1
                    c_aliases = [str(q.gold_answer)] + [str(a) for a in q.metadata.get("gold_aliases", [])]
                    ctx_norm = normalize(ctx)
                    if any(normalize(a) in ctx_norm for a in c_aliases if a):
                        st["e_control_hits"] += 1

            await adapter.cleanup(tenant_id)

    report: Dict[str, Any] = {
        "benchmark": "EpiBench Offline Epistemic-Hygiene Evaluation ($0.00 LLM spend)",
        "generator_version": GENERATOR_VERSION,
        "adapter_contract_version": ADAPTER_CONTRACT_VERSION,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "profile": eff_profile,
        "epistemic_stress": eff_stress,
        "distractor_mode": eff_mode,
        "filler_count": eff_fillers,
        "token_budget": eff_budget,
        "tight_constraint_budget": tight_constraint_budget,
        "seed": seed,
        "tracks": track_list,
        "total_scenarios": len(scenarios),
        "total_questions": sum(len(sc.questions) for sc in scenarios),
        "manifest_sha256": manifest_sha,
        "total_llm_spend_usd": 0.0,
        "wall_time_seconds": round(time.perf_counter() - t_start, 2),
        "systems": {},
    }

    for sys_name, st in acc.items():
        sess_rec = round(100.0 * statistics.mean(st["session_recalls"]), 2) if st["session_recalls"] else 0.0
        sup_prec = round(100.0 * st["r_current_clean"] / max(1, st["r_current_total"]), 2) if st["r_current_total"] else None
        bridge_rec = round(100.0 * statistics.mean(st["d_bridge_recalls"]), 2) if st["d_bridge_recalls"] else None
        full_chain_rec = round(100.0 * st["d_full_chain_hits"] / max(1, st["d_full_chain_total"]), 2) if st["d_full_chain_total"] else None
        x_primary = round(100.0 * st["x_primary_hits"] / max(1, st["x_total"]), 2) if st["x_total"] else None
        x_tight = round(100.0 * st["x_tight_hits"] / max(1, st["x_total"]), 2) if st["x_total"] else None
        ov_cov = round(100.0 * statistics.mean(st["overflow_coverages"]), 2) if st["overflow_coverages"] else None
        ov_complete = round(100.0 * st["overflow_complete_hits"] / max(1, st["overflow_total"]), 2) if st["overflow_total"] else None
        sig_density = round(100.0 * statistics.mean(st["signal_densities"]), 2) if st["signal_densities"] else 0.0
        e_leak = round(100.0 * st["e_leak_hits"] / max(1, st["e_leak_total"]), 2) if st["e_leak_total"] else None
        e_ctrl = round(100.0 * st["e_control_hits"] / max(1, st["e_control_total"]), 2) if st["e_control_total"] else None

        # Composite diagnostic averaging available epistemic hygiene pillars
        pillars = [sess_rec, sig_density]
        if sup_prec is not None:
            pillars.append(sup_prec)
        if bridge_rec is not None:
            pillars.append(bridge_rec)
        if x_tight is not None:
            pillars.append(x_tight)
        if e_ctrl is not None and e_leak is not None:
            pillars.append(e_ctrl * (1.0 - e_leak / 100.0))
        hygiene_idx = round(statistics.mean(pillars), 2) if pillars else 0.0

        report["systems"][sys_name] = {
            "epistemic_hygiene_index_pct": hygiene_idx,
            "gold_session_recall_pct": sess_rec,
            "per_track_session_recall_pct": {
                tr: round(100.0 * statistics.mean(st["track_session_recalls"][tr]), 2)
                for tr in track_list
                if st["track_session_recalls"][tr]
            },
            "current_state_supersession_precision_pct": sup_prec,
            "bridge_hop_recall_pct": bridge_rec,
            "full_chain_all_hops_pct": full_chain_rec,
            "zero_overlap_constraint_recall_pct": x_primary,
            "zero_overlap_constraint_recall_tight_pct": x_tight,
            "overflow_chain_coverage_pct": ov_cov,
            "overflow_complete_chain_pct": ov_complete,
            "signal_to_noise_density_pct": sig_density,
            "track_e_leak_rate_pct": e_leak,
            "track_e_control_retention_pct": e_ctrl,
            "as_of_violations": st["as_of_violations"],
            "budget_overshoots": st["budget_overshoots"],
            "max_context_tokens": st["max_tokens"],
            "mean_context_tokens": round(statistics.mean(st["token_counts"]), 1) if st["token_counts"] else 0.0,
            "latency_p50_ms": _pct(st["latencies_ms"], 0.50),
            "latency_p95_ms": _pct(st["latencies_ms"], 0.95),
        }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    return report


def format_offline_report(report: Dict[str, Any]) -> str:
    """Format offline epistemic-hygiene evaluation results as a readable terminal table."""
    lines = [
        "=" * 106,
        (
            f"EPIBENCH OFFLINE EPISTEMIC-HYGIENE REPORT "
            f"(profile={report['profile']}, scenarios={report['total_scenarios']}, "
            f"fillers={report['filler_count']}, budget={report['token_budget']}, spend=$0.00)"
        ),
        "=" * 106,
        (
            f"{'System':<18} {'Hygiene':>8} {'SessRec':>8} {'SupPrec':>8} {'BridgeD':>8} "
            f"{'X@600':>7} {'OvCov':>7} {'SigDens':>8} {'ELeak':>6} {'asOfV':>6} {'p95ms':>7}"
        ),
        "-" * 106,
    ]

    def _fmt(v: Optional[float], suffix: str = "%") -> str:
        return f"{v:.1f}{suffix}" if v is not None else "n/a"

    for name, s in report["systems"].items():
        lines.append(
            f"{name:<18} "
            f"{_fmt(s['epistemic_hygiene_index_pct']):>8} "
            f"{_fmt(s['gold_session_recall_pct']):>8} "
            f"{_fmt(s['current_state_supersession_precision_pct']):>8} "
            f"{_fmt(s['bridge_hop_recall_pct']):>8} "
            f"{_fmt(s['zero_overlap_constraint_recall_tight_pct']):>7} "
            f"{_fmt(s['overflow_chain_coverage_pct']):>7} "
            f"{_fmt(s['signal_to_noise_density_pct']):>8} "
            f"{_fmt(s['track_e_leak_rate_pct']):>6} "
            f"{s['as_of_violations']:>6d} "
            f"{s['latency_p95_ms']:>7.1f}"
        )
    lines.append("=" * 106)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run zero-cost ($0.00) EpiBench offline epistemic-hygiene evaluation.")
    parser.add_argument("--profile", choices=list(BENCHMARK_PROFILES.keys()) + ["epistemic_stress"], default="stress")
    parser.add_argument("--tracks", type=str, default=",".join(VALID_TRACKS))
    parser.add_argument("--systems", type=str, default="null_memory,oracle,recent_context,vector_rag,hybrid_rag")
    parser.add_argument("--scenarios-per-track", type=int, default=12)
    parser.add_argument("--filler-count", type=int, default=None)
    parser.add_argument("--distractor-mode", choices=["neutral", "same_domain", "mixed"], default=None)
    parser.add_argument("--distractor-file", type=str, default=None)
    parser.add_argument("--token-budget", type=int, default=None)
    parser.add_argument("--tight-constraint-budget", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--fake-embedder", action="store_true", help="Use deterministic fake embedder (fast CI smoke).")
    parser.add_argument("--output", type=str, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    tracks = [t.strip().upper() for t in args.tracks.split(",") if t.strip()]
    systems = [s.strip().lower() for s in args.systems.split(",") if s.strip()]
    report = asyncio.run(
        run_offline_epistemic_eval(
            profile=args.profile,
            tracks=tracks,
            systems=systems,
            scenarios_per_track=args.scenarios_per_track,
            filler_count=args.filler_count,
            distractor_mode=args.distractor_mode,
            distractor_file=Path(args.distractor_file) if args.distractor_file else None,
            token_budget=args.token_budget,
            tight_constraint_budget=args.tight_constraint_budget,
            seed=args.seed,
            use_fake_embedder=args.fake_embedder,
            output_path=Path(args.output) if args.output else None,
        )
    )
    print(format_offline_report(report))


if __name__ == "__main__":
    main()
