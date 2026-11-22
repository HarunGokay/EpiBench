"""EpiBench Benchmark Orchestration Runner.

Orchestrates scenario ingestion, retrieval, answering, judging, and statistical compilation
across all evaluated memory architectures.
Enforces real-time spend metering and budget ceilings.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath("."))

from EpiBench.adapters.base import BaseMemoryAdapter
from EpiBench.adapters.full_context_adapter import FullContextAdapter
from EpiBench.adapters.graph_ablation_adapter import GraphNoHistoryAdapter, GraphNoRelationsAdapter
from EpiBench.adapters.graph_memory_adapter import RelationalGraphAdapter
from EpiBench.adapters.hybrid_rag_adapter import HybridRAGAdapter
from EpiBench.adapters.mem0_adapter import Mem0Adapter
from EpiBench.adapters.vector_rag_adapter import VectorRAGAdapter
from EpiBench.generator.scenario_generator import Scenario, ScenarioGenerator
from EpiBench.judging.programmatic import (
    check_integer_match,
    check_set_overlap,
    check_stale_answer,
    check_string_match,
)
from EpiBench.judging.rubric_judge import RubricJudge
from EpiBench.metrics.statistics import exact_mcnemar_test, paired_scenario_bootstrap, wilson_ci
from EpiBench.runners.metering import BUDGET_HARD_CAP_USD, SpendTracker
from EpiBench.runners.reader import ReaderLLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger("epibench_runner")


class EpiBenchRunner:
    """Benchmark runner executing multi-system capability evaluations."""

    def __init__(
        self,
        api_key: str,
        spend_tracker: SpendTracker,
        token_budget: int = 1500,
        output_dir: Path = Path("EpiBench/results/pilot"),
    ):
        self.api_key = api_key
        self.spend_tracker = spend_tracker
        self.token_budget = token_budget
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.reader = ReaderLLM(api_key=api_key, spend_tracker=spend_tracker)
        self.judge = RubricJudge(api_key=api_key, spend_tracker=spend_tracker)
        self.scenario_gen = ScenarioGenerator()

    def build_adapters(self, system_names: List[str]) -> Dict[str, BaseMemoryAdapter]:
        adapters = {}
        for name in system_names:
            name_lower = name.lower()
            if name_lower in ("graph_memory", "relational_graph"):
                adapters["graph_memory"] = RelationalGraphAdapter(spend_tracker=self.spend_tracker)
            elif name_lower == "hybrid_rag":
                adapters["hybrid_rag"] = HybridRAGAdapter()
            elif name_lower == "vector_rag":
                adapters["vector_rag"] = VectorRAGAdapter()
            elif name_lower == "mem0":
                adapters["mem0"] = Mem0Adapter(api_key=self.api_key, spend_tracker=self.spend_tracker)
            elif name_lower == "full_context":
                adapters["full_context"] = FullContextAdapter()
            elif name_lower in ("graph_no_relations", "graph_no_graph"):
                adapters["graph_no_relations"] = GraphNoRelationsAdapter(spend_tracker=self.spend_tracker)
            elif name_lower == "graph_no_history":
                adapters["graph_no_history"] = GraphNoHistoryAdapter(spend_tracker=self.spend_tracker)
            else:
                logger.warning(f"Unrecognized system name: {name}")
        return adapters

    async def evaluate_question(
        self,
        question_item: Any,
        hypothesis: str,
        metadata: Dict[str, Any],
        rng: random.Random,
    ) -> Dict[str, Any]:
        """Evaluate hypothesis against gold answer using programmatic checkers first, LLM rubric second."""
        scoring_type = question_item.scoring_type
        gold = question_item.gold_answer
        is_correct = False
        details: Dict[str, Any] = {}

        if scoring_type == "string_match":
            is_correct = check_string_match(hypothesis, str(gold))
            details["method"] = "programmatic_string_match"
        elif scoring_type == "integer_match":
            try:
                is_correct = check_integer_match(hypothesis, int(gold))
                details["method"] = "programmatic_integer_match"
            except (ValueError, TypeError):
                is_correct = check_string_match(hypothesis, str(gold))
                details["method"] = "programmatic_fallback_string"
        elif scoring_type == "set_overlap":
            gold_set = gold if isinstance(gold, list) else [str(gold)]
            unaffected = [metadata.get("unaffected")] if metadata.get("unaffected") else None
            overlap = check_set_overlap(hypothesis, gold_set, unaffected_items=unaffected)
            is_correct = overlap["is_correct"]
            details.update(overlap)
            details["method"] = "programmatic_set_overlap"
        elif scoring_type == "llm_rubric":
            # Primary judge
            eval_res = await self.judge.evaluate_rubric(
                question=question_item.question_text,
                gold_answer_or_rubric=question_item.rubric,
                hypothesis=hypothesis,
                metadata=question_item.metadata,
                use_secondary_prompt=False,
            )
            is_correct = eval_res["is_correct"]
            details["primary_judge"] = eval_res
            details["method"] = "llm_rubric_primary"

            # 10% secondary judge audit
            if rng.random() < 0.15:
                sec_res = await self.judge.evaluate_rubric(
                    question=question_item.question_text,
                    gold_answer_or_rubric=question_item.rubric,
                    hypothesis=hypothesis,
                    metadata=question_item.metadata,
                    use_secondary_prompt=True,
                )
                details["secondary_judge"] = sec_res
                details["judge_agreement"] = (eval_res["is_correct"] == sec_res["is_correct"])

        # Check track-specific diagnostic anomalies
        if question_item.question_subkind == "current" and "stale_value" in question_item.metadata:
            details["is_stale_answer"] = check_stale_answer(hypothesis, question_item.metadata["stale_value"])

        if "forbidden_distractor" in question_item.metadata:
            details["is_contaminated"] = check_string_match(hypothesis, question_item.metadata["forbidden_distractor"])

        return {
            "is_correct": is_correct,
            "scoring_type": scoring_type,
            "details": details,
        }

    async def run_scenario(
        self,
        scenario: Scenario,
        adapters: Dict[str, BaseMemoryAdapter],
        rng: random.Random,
    ) -> Dict[str, Any]:
        """Run all systems against a single scenario timeline."""
        scenario_results: Dict[str, Any] = {
            "scenario_id": scenario.scenario_id,
            "track": scenario.track,
            "split": scenario.split,
            "user_name": scenario.user_name,
            "question_date": scenario.question_date,
            "session_count": len(scenario.sessions),
            "question_count": len(scenario.questions),
            "system_results": {},
        }

        # Step 1: Ingest into each system
        for sys_name, adapter in adapters.items():
            tenant_id = f"epi_{scenario.scenario_id}_{sys_name}"
            # Check if ablation reusing persisted graph memory
            if sys_name.startswith("graph_no_"):
                tenant_id = f"epi_{scenario.scenario_id}_graph_memory"
                ingest_res = None
            else:
                logger.info(f"[{scenario.scenario_id}] Ingesting {len(scenario.sessions)} sessions into {sys_name}...")
                ingest_res = await adapter.ingest(tenant_id, scenario.sessions)

            # Step 2: Retrieve, Answer, and Judge each question
            q_results = []
            correct_count = 0
            total_retrieval_latency = 0.0
            total_context_tokens = 0

            for q in scenario.questions:
                t0_ret = time.perf_counter()
                as_of_param = scenario.question_date
                if q.question_subkind == "as_of":
                    # Parse as_of date if question specifies
                    as_of_param = q.metadata.get("as_of_date", scenario.question_date)

                retrieval = await adapter.retrieve(
                    tenant_id=tenant_id,
                    query=q.question_text,
                    token_budget=self.token_budget,
                    as_of=as_of_param,
                )
                ret_latency = retrieval.retrieval_latency_ms
                total_retrieval_latency += ret_latency
                total_context_tokens += retrieval.token_count

                # Reader LLM
                hyp = await self.reader.answer_question(
                    context=retrieval.context,
                    question=q.question_text,
                    question_date=as_of_param if q.question_subkind == "as_of" else scenario.question_date,
                )

                # Judge
                eval_res = await self.evaluate_question(q, hyp, q.metadata, rng)
                if eval_res["is_correct"]:
                    correct_count += 1

                q_results.append({
                    "question_id": q.question_id,
                    "question_subkind": q.question_subkind,
                    "question_text": q.question_text,
                    "gold_answer": q.gold_answer,
                    "hypothesis": hyp,
                    "is_correct": eval_res["is_correct"],
                    "scoring_type": q.scoring_type,
                    "evaluation_details": eval_res["details"],
                    "retrieved_context_preview": retrieval.context[:300] + ("..." if len(retrieval.context) > 300 else ""),
                    "context_tokens": retrieval.token_count,
                    "retrieval_latency_ms": round(ret_latency, 2),
                })

            acc = correct_count / max(1, len(scenario.questions))
            scenario_pass = (correct_count == len(scenario.questions))

            scenario_results["system_results"][sys_name] = {
                "scenario_accuracy": acc,
                "scenario_pass": scenario_pass,
                "correct_questions": correct_count,
                "total_questions": len(scenario.questions),
                "avg_retrieval_latency_ms": round(total_retrieval_latency / max(1, len(scenario.questions)), 2),
                "avg_context_tokens": round(total_context_tokens / max(1, len(scenario.questions)), 1),
                "ingest_latency_ms": round(ingest_res.latency_ms, 2) if ingest_res else 0.0,
                "ingest_cost_usd": round(ingest_res.cost_usd, 4) if ingest_res else 0.0,
                "questions": q_results,
            }

        return scenario_results

    async def run_benchmark(
        self,
        tracks: List[str],
        systems: List[str],
        scenarios_per_track: int = 3,
        split: str = "dev",
        seed: int = 20261122,
        filler_count: int = 6,
    ) -> Dict[str, Any]:
        """Execute benchmark across tracks and systems."""
        logger.info(f"Starting EpiBench run: Tracks={tracks}, Systems={systems}, Scenarios/Track={scenarios_per_track}, Split={split}, Fillers={filler_count}")
        rng = random.Random(seed)
        adapters = self.build_adapters(systems)

        all_scenarios: List[Scenario] = []
        for t in tracks:
            for idx in range(scenarios_per_track):
                sc = self.scenario_gen.generate_scenario(
                    track=t,
                    scenario_idx=idx,
                    split=split,
                    seed=seed + idx * 100,
                    filler_count=filler_count,
                )
                all_scenarios.append(sc)

        logger.info(f"Generated {len(all_scenarios)} total scenarios across {len(tracks)} tracks.")

        results = []
        for s_idx, sc in enumerate(all_scenarios):
            logger.info(f"\n--- Running Scenario {s_idx+1}/{len(all_scenarios)}: {sc.scenario_id} (Track {sc.track}) ---")
            sc_res = await self.run_scenario(sc, adapters, rng)
            results.append(sc_res)

            # Checkpoint scenario
            ckpt_path = self.output_dir / f"checkpoint_{sc.scenario_id}.json"
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump(sc_res, f, indent=2)

            # Log side-by-side verification preview for manual audit
            primary_sys = "graph_memory" if "graph_memory" in sc_res["system_results"] else list(sc_res["system_results"].keys())[0]
            for q in sc_res["system_results"].get(primary_sys, {}).get("questions", []):
                hyp_ans = q["hypothesis"]
                corr_ans = "PASS" if q["is_correct"] else "FAIL"
                logger.info(f"[{corr_ans}] Q: {q['question_text'][:70]}...")
                logger.info(f"       Gold: {str(q['gold_answer'])[:70]}")
                logger.info(f"       Hyp ({primary_sys}): {hyp_ans[:70]}")

        # Clean up all tenants
        for sys_name, adapter in adapters.items():
            for sc in all_scenarios:
                t_id = f"epi_{sc.scenario_id}_{sys_name}"
                try:
                    await adapter.cleanup(t_id)
                except Exception:
                    pass

        # Compile Master Empirical Summary
        summary = self.compile_summary(results, systems, tracks)
        master_output = {
            "benchmark": "EpiBench Pilot Run",
            "date_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit_sha": "c92afdac8fbd3b3c657722ca67ccdb394d053ea8",
            "tracks": tracks,
            "systems": systems,
            "total_scenarios": len(all_scenarios),
            "spend_summary": self.spend_tracker.summary(),
            "performance_summary": summary,
            "scenario_results": results,
        }

        master_path = self.output_dir / "results_pilot.json"
        with open(master_path, "w", encoding="utf-8") as f:
            json.dump(master_output, f, indent=2)

        logger.info(f"\nMaster pilot results written to {master_path}")
        return master_output

    def compile_summary(
        self,
        results: List[Dict[str, Any]],
        systems: List[str],
        tracks: List[str],
    ) -> Dict[str, Any]:
        """Compile scenario and track level empirical metrics, Wilson intervals, and McNemar tests."""
        summary: Dict[str, Any] = {"by_system": {}, "by_track": {}, "comparisons": {}}

        for sys_name in systems:
            sys_scenarios = [r["system_results"].get(sys_name, {}) for r in results if sys_name in r["system_results"]]
            total_sc = len(sys_scenarios)
            passed_sc = sum(1 for s in sys_scenarios if s.get("scenario_pass", False))
            total_q = sum(s.get("total_questions", 0) for s in sys_scenarios)
            correct_q = sum(s.get("correct_questions", 0) for s in sys_scenarios)

            q_acc = (correct_q / max(1, total_q)) if total_q > 0 else 0.0
            sc_acc = (passed_sc / max(1, total_sc)) if total_sc > 0 else 0.0

            ci_q = wilson_ci(correct_q, total_q)
            ci_sc = wilson_ci(passed_sc, total_sc)

            avg_lat = sum(s.get("avg_retrieval_latency_ms", 0.0) for s in sys_scenarios) / max(1, total_sc)
            avg_tok = sum(s.get("avg_context_tokens", 0.0) for s in sys_scenarios) / max(1, total_sc)
            total_ingest_cost = sum(s.get("ingest_cost_usd", 0.0) for s in sys_scenarios)

            summary["by_system"][sys_name] = {
                "question_accuracy": round(q_acc * 100, 2),
                "question_correct": correct_q,
                "total_questions": total_q,
                "question_wilson_ci": [round(ci_q["lower"] * 100, 2), round(ci_q["upper"] * 100, 2)],
                "scenario_pass_rate": round(sc_acc * 100, 2),
                "scenario_passed": passed_sc,
                "total_scenarios": total_sc,
                "scenario_wilson_ci": [round(ci_sc["lower"] * 100, 2), round(ci_sc["upper"] * 100, 2)],
                "avg_retrieval_latency_ms": round(avg_lat, 2),
                "avg_context_tokens": round(avg_tok, 1),
                "total_ingest_cost_usd": round(total_ingest_cost, 4),
            }

        # Track breakdown
        for t in tracks:
            summary["by_track"][t] = {}
            t_scenarios = [r for r in results if r["track"] == t]
            for sys_name in systems:
                sys_t = [s["system_results"].get(sys_name, {}) for s in t_scenarios if sys_name in s["system_results"]]
                tot_q = sum(s.get("total_questions", 0) for s in sys_t)
                corr_q = sum(s.get("correct_questions", 0) for s in sys_t)
                acc = (corr_q / max(1, tot_q)) if tot_q > 0 else 0.0
                summary["by_track"][t][sys_name] = {
                    "accuracy": round(acc * 100, 2),
                    "correct": corr_q,
                    "total": tot_q,
                }

        # Primary comparison: Relational Graph Memory vs Hybrid RAG
        graph_key = "graph_memory" if "graph_memory" in systems else None
        if graph_key and "hybrid_rag" in systems:
            b_wins = 0  # Graph Memory wins, Hybrid RAG loses
            c_wins = 0  # Hybrid RAG wins, Graph Memory loses
            both_wins = 0
            both_losses = 0

            graph_scores = []
            hrag_scores = []

            for r in results:
                g_res = r["system_results"].get(graph_key, {})
                h_res = r["system_results"].get("hybrid_rag", {})
                g_pass = g_res.get("scenario_pass", False)
                h_pass = h_res.get("scenario_pass", False)

                graph_scores.append(g_res.get("scenario_accuracy", 0.0))
                hrag_scores.append(h_res.get("scenario_accuracy", 0.0))

                if g_pass and not h_pass:
                    b_wins += 1
                elif h_pass and not g_pass:
                    c_wins += 1
                elif g_pass and h_pass:
                    both_wins += 1
                else:
                    both_losses += 1

            mcnemar = exact_mcnemar_test(b_wins, c_wins)
            boot = paired_scenario_bootstrap(graph_scores, hrag_scores, n_resamples=5000)

            summary["comparisons"]["graph_memory_vs_hybrid_rag"] = {
                "discordant_graph_memory_wins_b": b_wins,
                "discordant_hybrid_rag_wins_c": c_wins,
                "concordant_both_wins": both_wins,
                "concordant_both_losses": both_losses,
                "mcnemar_p_value": round(mcnemar["p_value"], 4),
                "odds_ratio": round(mcnemar["odds_ratio"], 2) if mcnemar["odds_ratio"] != float("inf") else "inf",
                "paired_bootstrap_mean_delta": round(boot["mean_diff"] * 100, 2),
                "paired_bootstrap_95_ci": [round(boot["ci_lower"] * 100, 2), round(boot["ci_upper"] * 100, 2)],
                "paired_bootstrap_p_value": round(boot["p_value"], 4),
            }

        return summary


async def main():
    parser = argparse.ArgumentParser(description="Run EpiBench Evaluation Suite")
    parser.add_argument("--mode", choices=["pilot", "main"], default="pilot")
    parser.add_argument("--tracks", type=str, default="R,D,X,F,I,P,K")
    parser.add_argument("--systems", type=str, default="graph_memory,hybrid_rag,vector_rag,mem0,full_context")
    parser.add_argument("--token-budget", type=int, default=1500)
    parser.add_argument("--output-dir", type=str, default="EpiBench/results/pilot")
    parser.add_argument("--scenarios-per-track", type=int, default=3)
    parser.add_argument("--filler-count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261122)
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required.")

    tracks = [t.strip().upper() for t in args.tracks.split(",") if t.strip()]
    systems = [s.strip().lower() for s in args.systems.split(",") if s.strip()]

    spend_tracker = SpendTracker(hard_cap_usd=BUDGET_HARD_CAP_USD)
    runner = EpiBenchRunner(
        api_key=api_key,
        spend_tracker=spend_tracker,
        token_budget=args.token_budget,
        output_dir=Path(args.output_dir),
    )

    t_start = time.perf_counter()
    res = await runner.run_benchmark(
        tracks=tracks,
        systems=systems,
        scenarios_per_track=args.scenarios_per_track,
        split="dev" if args.mode == "pilot" else "held_out",
        seed=args.seed,
        filler_count=args.filler_count,
    )
    total_sec = time.perf_counter() - t_start

    print("\n=======================================================")
    print("EPIBENCH PILOT RUN COMPLETE")
    print(f"Total Wall Time: {total_sec:.2f} s ({total_sec/60:.2f} min)")
    print(f"Total Cumulative Spend: ${spend_tracker.total_spend_usd:.4f} USD")
    print(f"Hard Cap: ${spend_tracker.hard_cap_usd:.2f} USD")
    print("=======================================================\n")


if __name__ == "__main__":
    asyncio.run(main())
