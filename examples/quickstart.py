"""EpiBench Quickstart Example.

Demonstrates how to:
1. Generate an epistemic scenario deterministically.
2. Implement a custom memory adapter using BaseMemoryAdapter.
3. Ingest sessions, perform point-in-time retrieval, and run programmatic evaluation.
"""

import asyncio
from typing import Any, Dict, List, Optional

from epibench.adapters.base import BaseMemoryAdapter, IngestResult, RetrievalResult
from epibench.generator.scenario_generator import ScenarioGenerator
from epibench.judging.programmatic import (
    check_integer_match,
    check_set_overlap,
    check_string_match,
)


class SimpleDictionaryAdapter(BaseMemoryAdapter):
    """Minimal memory adapter storing conversation turns in memory."""

    def __init__(self):
        super().__init__(system_name="simple_dictionary")
        self.stores: Dict[str, List[Dict[str, Any]]] = {}

    async def ingest(self, tenant_id: str, sessions: List[Dict[str, Any]]) -> IngestResult:
        self.stores[tenant_id] = sessions
        return IngestResult(
            tenant_id=tenant_id,
            session_count=len(sessions),
            latency_ms=1.0,
        )

    async def retrieve(
        self,
        tenant_id: str,
        query: str,
        token_budget: int = 1500,
        as_of: Optional[str] = None,
    ) -> RetrievalResult:
        sessions = self.stores.get(tenant_id, [])
        lines = []
        for s in sessions:
            for t in s.get("turns", []):
                lines.append(f"{t.get('role')}: {t.get('content') or t.get('text') or ''}")
        text = "\n".join(lines)
        return RetrievalResult(
            context=text[:token_budget * 4],
            token_count=len(text) // 4,
            retrieval_latency_ms=1.0,
        )


async def main():
    print("=== EpiBench Quickstart ===")

    # 1. Deterministic Scenario Generation
    gen = ScenarioGenerator()
    scenario = gen.generate_scenario(track="R", scenario_idx=0, split="dev", seed=20261122, filler_count=2)
    print(f"Generated Scenario ID: {scenario.scenario_id}")
    print(f"Total Sessions: {len(scenario.sessions)}")
    print(f"Number of Evaluation Questions: {len(scenario.questions)}")

    # 2. Ingestion
    adapter = SimpleDictionaryAdapter()
    tenant_id = f"demo_{scenario.scenario_id}"
    ingest_res = await adapter.ingest(tenant_id, scenario.sessions)
    print(f"Ingested {ingest_res.session_count} sessions in {ingest_res.latency_ms:.2f}ms")

    # 3. Retrieval & Programmatic Verification
    for idx, q in enumerate(scenario.questions):
        ret_res = await adapter.retrieve(tenant_id, q.question_text)
        print(f"\n[Q{idx+1}] {q.question_text}")
        print(f"Expected Gold: {q.gold_answer}")
        
        # Test evaluation checker
        if q.scoring_type == "string_match":
            passed = check_string_match(ret_res.context, str(q.gold_answer))
            print(f"Found in retrieved context: {passed}")
        elif q.scoring_type == "integer_match":
            passed = check_integer_match(ret_res.context, int(q.gold_answer))
            print(f"Integer match in context: {passed}")

    print("\nQuickstart finished successfully.")


if __name__ == "__main__":
    asyncio.run(main())
