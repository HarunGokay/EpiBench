"""EpiBench Metering and Cost Accounting Wrapper.

Tracks token consumption across evaluated architectures (ingestion, retrieval,
reader LLM, and judge LLM) and enforces caller-specified spend ceilings.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict

# Default token pricing rates (USD per 1M tokens)
RATE_INPUT_PER_M: float = 0.075
RATE_OUTPUT_PER_M: float = 0.30
RATE_CACHED_INPUT_PER_M: float = 0.01875


class HardBudgetCapReached(Exception):
    """Raised immediately when cumulative spend reaches the safety ceiling."""


class SpendTracker:
    """Spend tracker enforcing real-time budget limits."""

    def __init__(self, hard_cap_usd: float):
        if hard_cap_usd <= 0:
            raise ValueError("hard_cap_usd must be a positive number.")
        self.hard_cap_usd = float(hard_cap_usd)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0
        self.total_spend_usd = 0.0
        self.spend_by_stage: Dict[str, float] = defaultdict(float)
        self.tokens_by_stage: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"in": 0, "out": 0, "cached": 0}
        )

    def record(
        self,
        in_tokens: int,
        out_tokens: int,
        stage: str,
        cached_tokens: int = 0,
        in_cost_per_m: float = RATE_INPUT_PER_M,
        out_cost_per_m: float = RATE_OUTPUT_PER_M,
        cached_cost_per_m: float = RATE_CACHED_INPUT_PER_M,
    ) -> float:
        cost = (
            (in_tokens * in_cost_per_m)
            + (out_tokens * out_cost_per_m)
            + (cached_tokens * cached_cost_per_m)
        ) / 1_000_000.0

        self.total_input_tokens += in_tokens
        self.total_output_tokens += out_tokens
        self.total_cached_tokens += cached_tokens
        self.total_spend_usd += cost

        self.spend_by_stage[stage] += cost
        self.tokens_by_stage[stage]["in"] += in_tokens
        self.tokens_by_stage[stage]["out"] += out_tokens
        self.tokens_by_stage[stage]["cached"] += cached_tokens

        if self.total_spend_usd >= self.hard_cap_usd:
            raise HardBudgetCapReached(
                f"HARD BUDGET CEILING TRIGGERED: Current spend ${self.total_spend_usd:.4f} reached "
                f"the cap of ${self.hard_cap_usd:.2f} USD."
            )
        return cost

    def summary(self) -> Dict[str, Any]:
        return {
            "total_spend_usd": round(self.total_spend_usd, 4),
            "hard_cap_usd": self.hard_cap_usd,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "spend_by_stage": dict(self.spend_by_stage),
            "tokens_by_stage": {k: dict(v) for k, v in self.tokens_by_stage.items()},
        }
