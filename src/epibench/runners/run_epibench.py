"""EpiBench Benchmark Orchestration Runner (SCORER_VERSION 2.0.0).

Orchestrates deterministic scenario generation, chronological ingestion with
interleaved deletion replay, point-in-time (`as_of`) retrieval auditing,
standardized reader answering, strict programmatic + LLM rubric judging,
and summary compilation across evaluated memory architectures.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from epibench.adapters.base import (
    ADAPTER_CONTRACT_VERSION,
    BaseMemoryAdapter,
    parse_as_of_cutoff,
)
from epibench.adapters.full_context_adapter import (
    FullContextUnboundedAdapter,
    RecentContextAdapter,
)
from epibench.adapters.hybrid_rag_adapter import HybridRAGAdapter
from epibench.adapters.mem0_adapter import Mem0Adapter
from epibench.adapters.reference_adapters import NullMemoryAdapter, OracleAdapter
from epibench.adapters.vector_rag_adapter import VectorRAGAdapter
from epibench.generator.manifest import canonical_json_bytes, scenario_sha256
from epibench.generator.scenario_generator import (
    GENERATOR_VERSION,
    VALID_TRACKS,
    Scenario,
    ScenarioGenerator,
    derive_seed,
    parse_session_date,
)
from epibench.judging.programmatic import (
    SCORER_VERSION,
    check_forbidden_phrases,
    check_stale_answer,
    score_integer_match,
    score_set_overlap,
    score_string_match,
    score_yes_no,
)
from epibench.judging.rubric_judge import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_SECONDARY_AUDIT_RATE,
    RubricJudge,
    compute_cohens_kappa,
)
from epibench.metrics.eval_metrics import compute_dimensional_report
from epibench.metrics.schema import validate_results_schema
from epibench.metrics.statistics import compute_pairwise_comparisons, wilson_ci
from epibench.runners.fakes import FakeEmbedder, FakeJudge, FakeReader
from epibench.runners.metering import CURRENT_SYSTEM_VAR, SpendTracker
from epibench.runners.reader import READER_PROMPT_VERSION, ReaderLLM
from epibench.utils import get_git_commit_sha

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("epibench_runner")


def _build_default_adapter(
    name: str,
    api_key: str,
    spend_tracker: SpendTracker,
    output_dir: Path,
    embedder: Optional[Any] = None,
) -> BaseMemoryAdapter:
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
    if key == "mem0":
        return Mem0Adapter(
            api_key=api_key,
            base_storage_dir=output_dir / "mem0_qdrant",
            spend_tracker=spend_tracker,
        )
    raise ValueError(f"Unknown built-in system adapter: {name!r}")


class EpiBenchRunner:
    """Orchestrates EpiBench evaluation runs across memory architectures."""

    def __init__(
        self,
        api_key: str = "",
        spend_tracker: Optional[SpendTracker] = None,
        token_budget: int = 1500,
        output_dir: Path = Path("results/run"),
        reader_model: str = "gemini-3.5-flash-lite",
        judge_model: str = DEFAULT_JUDGE_MODEL,
        secondary_audit_rate: float = DEFAULT_SECONDARY_AUDIT_RATE,
        reader: Optional[Any] = None,
        judge: Optional[Any] = None,
        use_fakes: bool = False,
        strict_as_of_audit: bool = False,
        custom_adapters: Optional[Dict[str, BaseMemoryAdapter]] = None,
        experimental_tracks: Optional[Sequence[str]] = None,
    ):
        if spend_tracker is None:
            raise ValueError("spend_tracker with an explicit hard_cap_usd is required.")
        self.api_key = api_key
        self.spend_tracker = spend_tracker
        self.token_budget = token_budget
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.secondary_audit_rate = float(secondary_audit_rate)
        self.strict_as_of_audit = strict_as_of_audit
        self.custom_adapters: Dict[str, BaseMemoryAdapter] = dict(custom_adapters or {})
        self.experimental_tracks: Set[str] = {t.upper() for t in (experimental_tracks or [])}
        self.use_fakes = use_fakes

        if use_fakes:
            self.reader = reader or FakeReader()
            self.judge = judge or FakeJudge()
            self.embedder: Optional[Any] = FakeEmbedder()
        else:
            self.reader = reader or ReaderLLM(
                api_key=api_key,
                model_name=reader_model,
                spend_tracker=spend_tracker,
            )
            self.judge = judge or RubricJudge(
                api_key=api_key,
                model_name=judge_model,
                spend_tracker=spend_tracker,
                reader_model_name=getattr(self.reader, "model_name", reader_model),
            )
            self.embedder = None

        self.reader_model = getattr(self.reader, "model_name", reader_model)
        self.judge_model = getattr(self.judge, "model_name", judge_model)
        self.generator = ScenarioGenerator()

    def build_adapters(self, systems: Sequence[str]) -> Dict[str, BaseMemoryAdapter]:
        adapters: Dict[str, BaseMemoryAdapter] = {}
        for sys_name in systems:
            if sys_name in self.custom_adapters:
                adapters[sys_name] = self.custom_adapters[sys_name]
            else:
                adapters[sys_name] = _build_default_adapter(
                    sys_name,
                    api_key=self.api_key,
                    spend_tracker=self.spend_tracker,
                    output_dir=self.output_dir,
                    embedder=self.embedder,
                )
        return adapters

    async def _ingest_with_deletion_replay(
        self,
        adapter: BaseMemoryAdapter,
        tenant_id: str,
        sessions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Replay sessions chronologically, triggering adapter.delete() on deletion events."""
        ordered = sorted(sessions, key=lambda s: (s.get("session_date", ""), s.get("session_id", "")))
        has_deletion = any(s.get("is_deletion_event") for s in ordered)

        total_latency_ms = 0.0
        total_cost_usd = 0.0
        delete_events: List[Dict[str, Any]] = []
        delete_unsupported = False

        if not has_deletion:
            res = await adapter.ingest(tenant_id, ordered)
            return {
                "ingest_latency_ms": res.latency_ms,
                "ingest_cost_usd": res.cost_usd,
                "delete_events": [],
                "delete_unsupported": False,
            }

        batch: List[Dict[str, Any]] = []
        for s in ordered:
            batch.append(s)
            if s.get("is_deletion_event"):
                res = await adapter.ingest(tenant_id, batch)
                total_latency_ms += res.latency_ms
                total_cost_usd += res.cost_usd
                batch = []

                target = s.get("deleted_target")
                del_res = await adapter.delete(tenant_id, target=target)
                supported = bool(del_res.get("supported", False))
                if not supported:
                    delete_unsupported = True
                delete_events.append({
                    "session_id": s.get("session_id"),
                    "target": target,
                    "supported": supported,
                    "deleted": bool(del_res.get("deleted", False)),
                    "details": del_res,
                })

        if batch:
            res = await adapter.ingest(tenant_id, batch)
            total_latency_ms += res.latency_ms
            total_cost_usd += res.cost_usd

        return {
            "ingest_latency_ms": total_latency_ms,
            "ingest_cost_usd": total_cost_usd,
            "delete_events": delete_events,
            "delete_unsupported": delete_unsupported,
        }

    def _audit_as_of(
        self,
        scenario: Scenario,
        as_of_param: Optional[str],
        retrieval_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Verify that no returned source_session_id postdates as_of_param."""
        source_ids = retrieval_metadata.get("source_session_ids")
        if source_ids is None:
            return {"as_of_unverified": True, "as_of_violation": False, "violating_session_ids": []}

        cutoff = parse_as_of_cutoff(as_of_param)
        if cutoff is None:
            return {"as_of_unverified": False, "as_of_violation": False, "violating_session_ids": []}

        sess_dates = {s["session_id"]: parse_session_date(s["session_date"]) for s in scenario.sessions}
        violating = [
            sid for sid in source_ids
            if sid in sess_dates and sess_dates[sid] > cutoff
        ]
        if violating and self.strict_as_of_audit:
            raise AssertionError(
                f"as_of leakage violation in {scenario.scenario_id}: sessions {violating} postdate {as_of_param}"
            )
        return {
            "as_of_unverified": False,
            "as_of_violation": len(violating) > 0,
            "violating_session_ids": violating,
        }

    async def evaluate_scenario(
        self,
        scenario: Scenario,
        adapters: Dict[str, BaseMemoryAdapter],
        audit_rng: Optional[random.Random] = None,
    ) -> Dict[str, Any]:
        """Run ingestion (with deletion replay), retrieval, answering, and judging for one scenario."""
        rng = audit_rng or random.Random(20261122)
        sc_result: Dict[str, Any] = {
            "scenario_id": scenario.scenario_id,
            "track": scenario.track,
            "split": scenario.split,
            "template_id": scenario.metadata.get("template_id", ""),
            "user_name": scenario.user_name,
            "question_date": scenario.question_date,
            "session_count": len(scenario.sessions),
            "question_count": len(scenario.questions),
            "scenario_sha256": scenario_sha256(scenario.to_dict()),
            "system_results": {},
        }

        for sys_name, adapter in adapters.items():
            sys_token = CURRENT_SYSTEM_VAR.set(sys_name)
            try:
                sc_result["system_results"][sys_name] = await self._evaluate_single_system(
                    scenario=scenario,
                    sys_name=sys_name,
                    adapter=adapter,
                    rng=rng,
                )
            finally:
                CURRENT_SYSTEM_VAR.reset(sys_token)

        return sc_result

    async def _evaluate_single_system(
        self,
        scenario: Scenario,
        sys_name: str,
        adapter: BaseMemoryAdapter,
        rng: random.Random,
    ) -> Dict[str, Any]:
        tenant_id = f"epi_{scenario.scenario_id}_{sys_name}"
        spend_before_sys = self.spend_tracker.get_system_spend(sys_name)

        ingest_info = await self._ingest_with_deletion_replay(adapter, tenant_id, scenario.sessions)

        q_results: List[Dict[str, Any]] = []
        total_retrieval_latency = 0.0
        total_context_tokens = 0
        parse_failures = 0
        judge_errors = 0
        as_of_unverified_count = 0
        as_of_violation_count = 0

        for q in scenario.questions:
            as_of_param = q.metadata.get("as_of_date", scenario.question_date)
            retrieval = await adapter.retrieve(
                tenant_id=tenant_id,
                query=q.question_text,
                token_budget=self.token_budget,
                as_of=as_of_param,
            )
            total_retrieval_latency += retrieval.retrieval_latency_ms
            total_context_tokens += retrieval.token_count

            as_of_audit = self._audit_as_of(scenario, as_of_param, retrieval.metadata)
            if as_of_audit["as_of_unverified"]:
                as_of_unverified_count += 1
            if as_of_audit["as_of_violation"]:
                as_of_violation_count += 1

            # Always pass TRUE scenario.question_date to reader (fixes A4)
            hypothesis = await self.reader.answer_question(
                context=retrieval.context,
                question=q.question_text,
                question_date=scenario.question_date,
                scoring_type=q.scoring_type,
            )

            eval_details: Dict[str, Any] = {}
            is_correct = False
            parse_failure = False
            judge_error = False

            if q.scoring_type == "string_match":
                res = score_string_match(
                    hypothesis,
                    str(q.gold_answer),
                    aliases=q.metadata.get("gold_aliases"),
                )
                is_correct = res["is_correct"]
                parse_failure = res["parse_failure"]
                eval_details = {"method": "programmatic_string_match", **res}
                if "stale_value" in q.metadata:
                    eval_details["is_stale_answer"] = check_stale_answer(
                        hypothesis,
                        q.metadata["stale_value"],
                        stale_aliases=q.metadata.get("stale_aliases"),
                    )

            elif q.scoring_type == "integer_match":
                res = score_integer_match(hypothesis, int(q.gold_answer))
                is_correct = res["is_correct"]
                parse_failure = res["parse_failure"]
                eval_details = {"method": "programmatic_integer_match", **res}

            elif q.scoring_type == "yes_no":
                res = score_yes_no(hypothesis, str(q.gold_answer))
                is_correct = res["is_correct"]
                parse_failure = res["parse_failure"]
                eval_details = {"method": "programmatic_yes_no", **res}

            elif q.scoring_type == "set_overlap":
                res = score_set_overlap(
                    hypothesis,
                    q.metadata.get("gold_decisions", q.gold_answer),
                    unaffected_decisions=q.metadata.get("unaffected_decisions"),
                )
                is_correct = res["is_correct"]
                parse_failure = res["parse_failure"]
                eval_details = {"method": "programmatic_set_overlap", **res}

            elif q.scoring_type == "llm_rubric":
                j_res = await self.judge.evaluate_rubric(
                    question=q.question_text,
                    gold_answer_or_rubric=q.rubric,
                    hypothesis=hypothesis,
                    scoring_type=q.scoring_type,
                    metadata=q.metadata,
                    use_secondary_prompt=False,
                )
                is_correct = bool(j_res["is_correct"])
                judge_error = bool(j_res.get("judge_error", False))
                eval_details = {"method": "llm_rubric", **j_res}

                # Contamination check using strict whole-phrase forbidden aliases (fixes S5)
                if q.metadata.get("forbidden_aliases"):
                    contam = check_forbidden_phrases(hypothesis, q.metadata["forbidden_aliases"])
                    eval_details["is_contaminated"] = contam["contaminated"]
                    eval_details["contaminated_phrases"] = contam["matched_phrases"]
                    if contam["contaminated"] and "target_type" in q.metadata:
                        is_correct = False

                # Configurable secondary-judge inter-rater audit (default 10%, fixes M5)
                if self.secondary_audit_rate > 0.0 and rng.random() < self.secondary_audit_rate:
                    sec_res = await self.judge.evaluate_rubric(
                        question=q.question_text,
                        gold_answer_or_rubric=q.rubric,
                        hypothesis=hypothesis,
                        scoring_type=q.scoring_type,
                        metadata=q.metadata,
                        use_secondary_prompt=True,
                    )
                    eval_details["secondary_audit"] = sec_res

            else:
                raise ValueError(f"Unsupported scoring_type: {q.scoring_type!r}")

            if parse_failure:
                parse_failures += 1
            if judge_error:
                judge_errors += 1

            q_results.append({
                "question_id": q.question_id,
                "track": scenario.track,
                "question_subkind": q.question_subkind,
                "question_text": q.question_text,
                "gold_answer": q.gold_answer,
                "hypothesis": hypothesis,
                "is_correct": is_correct,
                "parse_failure": parse_failure,
                "judge_error": judge_error,
                "scoring_type": q.scoring_type,
                "as_of_param": as_of_param,
                "as_of_audit": as_of_audit,
                "evaluation_details": eval_details,
                "retrieved_context_preview": (
                    retrieval.context[:300] + "..." if len(retrieval.context) > 300 else retrieval.context
                ),
                "context_tokens": retrieval.token_count,
                "retrieval_latency_ms": round(retrieval.retrieval_latency_ms, 2),
            })

        num_q = len(q_results)
        correct_q = sum(1 for qr in q_results if qr["is_correct"])
        scenario_acc = correct_q / max(1, num_q)

        # Track E joint scoring: scenario passes iff control retained AND direct & semantic leak-free
        track_e_joint: Optional[Dict[str, Any]] = None
        if scenario.track == "E":
            q_by_sub = {qr["question_subkind"]: qr for qr in q_results}
            control_retained = bool(q_by_sub.get("deletion_control_retention", {}).get("is_correct", False))
            direct_ok = bool(q_by_sub.get("deletion_direct", {}).get("is_correct", False))
            semantic_ok = bool(q_by_sub.get("deletion_semantic", {}).get("is_correct", False))
            leak_free = direct_ok and semantic_ok
            joint_pass = control_retained and leak_free
            track_e_joint = {
                "control_retained": control_retained,
                "direct_leak_free": direct_ok,
                "semantic_leak_free": semantic_ok,
                "leak_free": leak_free,
                "joint_pass": joint_pass,
            }
            scenario_pass = joint_pass
        else:
            scenario_pass = (correct_q == num_q)

        scenario_spend_usd = max(
            0.0,
            (self.spend_tracker.get_system_spend(sys_name) - spend_before_sys),
        )
        # Add adapter-reported ingest_cost_usd only if adapter didn't already record through spend_tracker
        if ingest_info["ingest_cost_usd"] > 0 and scenario_spend_usd < ingest_info["ingest_cost_usd"]:
            scenario_spend_usd += ingest_info["ingest_cost_usd"]

        return {
            "scenario_accuracy": round(scenario_acc, 4),
            "scenario_pass": scenario_pass,
            "correct_questions": correct_q,
            "total_questions": num_q,
            "parse_failures": parse_failures,
            "judge_errors": judge_errors,
            "as_of_unverified_count": as_of_unverified_count,
            "as_of_violation_count": as_of_violation_count,
            "delete_unsupported": ingest_info["delete_unsupported"],
            "delete_events": ingest_info["delete_events"],
            "track_e_joint": track_e_joint,
            "avg_retrieval_latency_ms": round(total_retrieval_latency / max(1, num_q), 2),
            "avg_context_tokens": round(total_context_tokens / max(1, num_q), 1),
            "ingest_latency_ms": round(ingest_info["ingest_latency_ms"], 2),
            "ingest_cost_usd": round(ingest_info["ingest_cost_usd"], 5),
            "scenario_spend_usd": round(scenario_spend_usd, 6),
            "questions": q_results,
        }

    async def run_benchmark(
        self,
        tracks: Sequence[str],
        systems: Sequence[str],
        scenarios_per_track: int = 3,
        split: str = "dev",
        seed: int = 20261122,
        filler_count: int = 6,
        resume: bool = True,
        save_checkpoints: bool = True,
        output_filename: str = "results.json",
        concurrency: int = 1,
    ) -> Dict[str, Any]:
        """Execute full benchmark evaluation across tracks and systems."""
        track_list = [t.upper() for t in tracks]
        system_list = [s.lower() for s in systems]
        adapters = self.build_adapters(system_list)

        all_scenarios: List[Tuple[int, Scenario]] = []
        manifest_entries: List[Dict[str, Any]] = []
        for track in track_list:
            for idx in range(scenarios_per_track):
                sc = self.generator.generate_scenario(
                    track=track,
                    scenario_idx=idx,
                    split=split,
                    seed=seed,
                    filler_count=filler_count,
                )
                all_scenarios.append((idx, sc))
                manifest_entries.append({
                    "scenario_id": sc.scenario_id,
                    "track": sc.track,
                    "sha256": scenario_sha256(sc.to_dict()),
                })

        manifest_sha256 = hashlib.sha256(canonical_json_bytes(manifest_entries)).hexdigest()
        total_n = len(all_scenarios)
        results_by_pos: List[Optional[Dict[str, Any]]] = [None] * total_n
        sem = asyncio.Semaphore(max(1, int(concurrency)))

        async def _run_one(pos: int, sc_idx: int, sc: Scenario) -> None:
            chk_path = self.output_dir / f"checkpoint_{sc.scenario_id}.json"
            if resume and save_checkpoints and chk_path.exists():
                with open(chk_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                cached_sys = set(cached.get("system_results", {}).keys())
                if set(system_list).issubset(cached_sys):
                    results_by_pos[pos] = cached
                    return

            async with sem:
                logger.info(
                    "[%d/%d] Running %s (Track %s, %d sessions, %d questions)",
                    pos + 1,
                    total_n,
                    sc.scenario_id,
                    sc.track,
                    len(sc.sessions),
                    len(sc.questions),
                )
                sc_rng = random.Random(derive_seed(seed, sc.track, sc_idx, "secondary_audit"))
                sc_res = await self.evaluate_scenario(sc, adapters, audit_rng=sc_rng)
                results_by_pos[pos] = sc_res

                if save_checkpoints:
                    with open(chk_path, "w", encoding="utf-8") as f:
                        json.dump(sc_res, f, indent=2)

                for sys_name, adapter in adapters.items():
                    t_id = f"epi_{sc.scenario_id}_{sys_name}"
                    try:
                        await adapter.cleanup(t_id)
                    except Exception:
                        pass

        if concurrency > 1:
            await asyncio.gather(*(_run_one(pos, sc_idx, sc) for pos, (sc_idx, sc) in enumerate(all_scenarios)))
        else:
            for pos, (sc_idx, sc) in enumerate(all_scenarios):
                await _run_one(pos, sc_idx, sc)

        results: List[Dict[str, Any]] = [r for r in results_by_pos if r is not None]

        summary = self.compile_summary(results, system_list, track_list, seed=seed)

        validity_summary: Optional[Dict[str, Any]] = None
        default_val_path = Path("results/validity/validity_report.json")
        if default_val_path.exists():
            try:
                vr = json.loads(default_val_path.read_text(encoding="utf-8"))
                validity_summary = {
                    "live_verified": bool(vr.get("live_verified", False)),
                    "all_tracks_passed": bool(vr.get("all_tracks_passed", False)),
                    "passing_tracks": list(vr.get("passing_tracks", [])),
                    "experimental_tracks": list(vr.get("experimental_tracks", [])),
                }
            except Exception:
                validity_summary = None

        master_output: Dict[str, Any] = {
            "benchmark": "EpiBench",
            "scorer_version": SCORER_VERSION,
            "generator_version": GENERATOR_VERSION,
            "reader_prompt_version": READER_PROMPT_VERSION,
            "adapter_contract_version": ADAPTER_CONTRACT_VERSION,
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit_sha": get_git_commit_sha(),
            "manifest_sha256": manifest_sha256,
            "split": split,
            "seed": seed,
            "filler_count": filler_count,
            "token_budget": self.token_budget,
            "reader_model": self.reader_model,
            "judge_model": self.judge_model,
            "secondary_audit_rate": self.secondary_audit_rate,
            "use_fakes": self.use_fakes,
            "experimental_tracks": sorted(self.experimental_tracks),
            "tracks": track_list,
            "systems": system_list,
            "total_scenarios": len(all_scenarios),
            "validity_gate_summary": validity_summary,
            "spend_summary": self.spend_tracker.summary(),
            "performance_summary": summary,
            "scenario_results": results,
        }

        validate_results_schema(master_output)

        master_path = self.output_dir / output_filename
        with open(master_path, "w", encoding="utf-8") as f:
            json.dump(master_output, f, indent=2)

        logger.info("Master results written to %s", master_path)
        return master_output

    def compile_summary(
        self,
        results: List[Dict[str, Any]],
        systems: Sequence[str],
        tracks: Sequence[str],
        seed: int = 20261122,
    ) -> Dict[str, Any]:
        """Compile per-system, per-track, pairwise comparisons, secondary audit, and dimensional metrics."""
        summary: Dict[str, Any] = {
            "by_system": {},
            "by_track": {},
            "comparisons": {},
            "secondary_audit": {},
            "dimensional_reports": {},
        }

        audit_pairs: List[Tuple[bool, bool]] = []

        for sys_name in systems:
            sys_pairs = [
                (r, r["system_results"][sys_name])
                for r in results
                if sys_name in r["system_results"]
            ]
            sys_scenarios = [s for _, s in sys_pairs]
            total_sc = len(sys_scenarios)
            passed_sc = sum(1 for s in sys_scenarios if s.get("scenario_pass", False))
            total_q = sum(s.get("total_questions", 0) for s in sys_scenarios)
            correct_q = sum(s.get("correct_questions", 0) for s in sys_scenarios)
            total_parse_failures = sum(s.get("parse_failures", 0) for s in sys_scenarios)
            total_judge_errors = sum(s.get("judge_errors", 0) for s in sys_scenarios)
            total_as_of_unverified = sum(s.get("as_of_unverified_count", 0) for s in sys_scenarios)
            total_as_of_violations = sum(s.get("as_of_violation_count", 0) for s in sys_scenarios)
            total_delete_unsupported = sum(1 for s in sys_scenarios if s.get("delete_unsupported", False))

            q_acc = (correct_q / total_q) if total_q > 0 else 0.0
            sc_acc = (passed_sc / total_sc) if total_sc > 0 else 0.0
            ci_q = wilson_ci(correct_q, total_q)
            ci_sc = wilson_ci(passed_sc, total_sc)

            avg_lat = sum(s.get("avg_retrieval_latency_ms", 0.0) for s in sys_scenarios) / max(1, total_sc)
            avg_tok = sum(s.get("avg_context_tokens", 0.0) for s in sys_scenarios) / max(1, total_sc)
            total_ingest_cost = sum(s.get("ingest_cost_usd", 0.0) for s in sys_scenarios)
            tracked_spend = self.spend_tracker.get_system_spend(sys_name)
            recorded_sc_spend = sum(s.get("scenario_spend_usd", 0.0) for s in sys_scenarios)
            total_sys_spend = max(tracked_spend, recorded_sc_spend, total_ingest_cost)

            # Build dimensional evaluation inputs: Track E uses joint outcome for privacy_and_deletion
            # and deletion_control_retention separately for retention_control
            dim_evals: List[Dict[str, Any]] = []
            for sc_row, s_res in sys_pairs:
                tr = sc_row["track"]
                if tr == "E" and s_res.get("track_e_joint") is not None:
                    dim_evals.append({
                        "track": "E",
                        "question_subkind": "deletion_joint",
                        "is_correct": bool(s_res["track_e_joint"]["joint_pass"]),
                    })
                    for q in s_res.get("questions", []):
                        if q.get("question_subkind") == "deletion_control_retention":
                            dim_evals.append(q)
                else:
                    dim_evals.extend(s_res.get("questions", []))

            dim_report = compute_dimensional_report(
                evaluations=dim_evals,
                system_name=sys_name,
                total_spend_usd=total_sys_spend,
                experimental_tracks=sorted(self.experimental_tracks),
            )
            summary["dimensional_reports"][sys_name] = dim_report.to_dict()

            cost_per_ans = round(total_sys_spend / correct_q, 6) if correct_q > 0 else None

            summary["by_system"][sys_name] = {
                "question_accuracy": round(q_acc * 100.0, 2),
                "question_correct": correct_q,
                "question_total": total_q,
                "question_ci_95": [round(ci_q["lower"] * 100.0, 2), round(ci_q["upper"] * 100.0, 2)],
                "scenario_pass_rate": round(sc_acc * 100.0, 2),
                "scenario_passed": passed_sc,
                "scenario_total": total_sc,
                "scenario_ci_95": [round(ci_sc["lower"] * 100.0, 2), round(ci_sc["upper"] * 100.0, 2)],
                "parse_failures": total_parse_failures,
                "judge_errors": total_judge_errors,
                "as_of_unverified_count": total_as_of_unverified,
                "as_of_violation_count": total_as_of_violations,
                "delete_unsupported_count": total_delete_unsupported,
                "avg_retrieval_latency_ms": round(avg_lat, 2),
                "avg_context_tokens": round(avg_tok, 1),
                "total_ingest_cost_usd": round(total_ingest_cost, 6),
                "total_spend_usd": round(total_sys_spend, 6),
                "cost_per_successful_answer_usd": cost_per_ans,
            }

            # Collect secondary audit pairs
            for s in sys_scenarios:
                for q in s.get("questions", []):
                    sec = q.get("evaluation_details", {}).get("secondary_audit")
                    if sec is not None and not sec.get("judge_error", False) and not q.get("judge_error", False):
                        audit_pairs.append((bool(q["is_correct"]), bool(sec["is_correct"])))

        summary["secondary_audit"] = compute_cohens_kappa(audit_pairs)

        # Pairwise McNemar, cluster-aware paired scenario bootstrap, and Holm-Bonferroni correction
        if len(systems) >= 2 and results:
            summary["comparisons"] = compute_pairwise_comparisons(
                results,
                systems,
                n_resamples=10000,
                seed=seed,
            )

        # Track-level breakdown (with Track E joint scoring)
        for track in tracks:
            summary["by_track"][track] = {}
            track_res = [r for r in results if r["track"] == track]
            for sys_name in systems:
                t_sys = [r["system_results"][sys_name] for r in track_res if sys_name in r["system_results"]]
                t_total_q = sum(s.get("total_questions", 0) for s in t_sys)
                t_corr_q = sum(s.get("correct_questions", 0) for s in t_sys)
                t_total_sc = len(t_sys)
                t_pass_sc = sum(1 for s in t_sys if s.get("scenario_pass", False))
                q_pct = round((t_corr_q / max(1, t_total_q)) * 100.0, 2) if t_total_q > 0 else 0.0
                sc_pct = round((t_pass_sc / max(1, t_total_sc)) * 100.0, 2) if t_total_sc > 0 else 0.0
                entry: Dict[str, Any] = {
                    "question_accuracy": q_pct,
                    "question_correct": t_corr_q,
                    "question_total": t_total_q,
                    "scenario_pass_rate": sc_pct,
                    "scenario_passed": t_pass_sc,
                    "scenario_total": t_total_sc,
                    "effective_score": sc_pct if track == "E" else q_pct,
                    "experimental": track in self.experimental_tracks,
                }
                if track == "E" and t_sys:
                    ctrl_ok = sum(
                        1 for s in t_sys if (s.get("track_e_joint") or {}).get("control_retained", False)
                    )
                    leak_free = sum(
                        1 for s in t_sys if (s.get("track_e_joint") or {}).get("leak_free", False)
                    )
                    entry["retention_control_rate"] = round((ctrl_ok / max(1, t_total_sc)) * 100.0, 2)
                    entry["leak_free_rate"] = round((leak_free / max(1, t_total_sc)) * 100.0, 2)
                    entry["joint_pass_rate"] = sc_pct
                summary["by_track"][track][sys_name] = entry

        return summary


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run EpiBench Evaluation Suite")
    parser.add_argument("--split", choices=["dev", "held_out"], default="dev")
    parser.add_argument("--tracks", type=str, default=",".join(VALID_TRACKS))
    parser.add_argument("--systems", type=str, default="null_memory,oracle,vector_rag,hybrid_rag,recent_context")
    parser.add_argument("--token-budget", type=int, default=1500)
    parser.add_argument("--budget-usd", type=float, required=True, help="Hard spend ceiling in USD (required).")
    parser.add_argument("--output-dir", type=str, default="results/run")
    parser.add_argument("--scenarios-per-track", type=int, default=12)
    parser.add_argument("--filler-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--reader-model", type=str, default="gemini-3.5-flash-lite")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--secondary-audit-rate", type=float, default=DEFAULT_SECONDARY_AUDIT_RATE)
    parser.add_argument("--offline", action="store_true", help="Run offline with FakeReader and FakeJudge.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints.")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not args.offline and not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required unless --offline is specified.")

    tracks = [t.strip().upper() for t in args.tracks.split(",") if t.strip()]
    systems = [s.strip().lower() for s in args.systems.split(",") if s.strip()]

    spend_tracker = SpendTracker(hard_cap_usd=args.budget_usd)
    runner = EpiBenchRunner(
        api_key=api_key,
        spend_tracker=spend_tracker,
        token_budget=args.token_budget,
        output_dir=Path(args.output_dir),
        reader_model=args.reader_model,
        judge_model=args.judge_model,
        secondary_audit_rate=args.secondary_audit_rate,
        use_fakes=args.offline,
    )

    t_start = time.perf_counter()
    await runner.run_benchmark(
        tracks=tracks,
        systems=systems,
        scenarios_per_track=args.scenarios_per_track,
        split=args.split,
        seed=args.seed,
        filler_count=args.filler_count,
        resume=not args.no_resume,
    )
    total_sec = time.perf_counter() - t_start

    print("\n=======================================================")
    print("EPIBENCH RUN COMPLETE")
    print(f"Total Wall Time: {total_sec:.2f} s ({total_sec / 60:.2f} min)")
    print(f"Total Cumulative Spend: ${spend_tracker.total_spend_usd:.4f} USD")
    print(f"Hard Cap: ${spend_tracker.hard_cap_usd:.2f} USD")
    print("=======================================================\n")


if __name__ == "__main__":
    asyncio.run(main())
