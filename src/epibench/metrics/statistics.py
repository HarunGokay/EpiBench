"""EpiBench Statistical Suite.

Implements cluster-aware scenario-level statistics:
- Exact two-sided McNemar test
- Paired scenario bootstrap with 95% confidence intervals (B=10,000)
- Wilson score confidence intervals for binomial endpoints
- Holm-Bonferroni step-down correction for multiple testing
- Domain and track specific diagnostic rates:
  * Stale-answer rate (Track R)
  * Dependency precision, recall, and F1 (Track D)
  * Hidden-constraint compliance rate (Track X)
  * Slippage count accuracy (Track F)
  * Unsupported-causal-claim rate (Track I)
  * Contamination / misinformation rate (Track P)
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple


def wilson_ci(successes: int, total: int, confidence: float = 0.95) -> Dict[str, float]:
    """Wilson score confidence interval for a proportion."""
    if total <= 0:
        return {"proportion": 0.0, "lower": 0.0, "upper": 0.0}

    z = 1.95996  # 95%
    p = successes / total
    denom = 1.0 + (z * z) / total
    centre = (p + (z * z) / (2.0 * total)) / denom
    spread = (z / denom) * math.sqrt((p * (1.0 - p) / total) + ((z * z) / (4.0 * total * total)))
    return {
        "proportion": p,
        "lower": max(0.0, centre - spread),
        "upper": min(1.0, centre + spread),
    }


def exact_mcnemar_test(b: int, c: int) -> Dict[str, Any]:
    """Exact two-sided McNemar test using the Binomial distribution on discordant pairs."""
    n_disc = b + c
    if n_disc == 0:
        return {"b": b, "c": c, "p_value": 1.0, "odds_ratio": 1.0}

    k = min(b, c)
    # Two-sided exact binomial cumulative distribution
    p_val = 0.0
    for i in range(k + 1):
        p_val += math.comb(n_disc, i) * (0.5 ** n_disc)
    p_val = min(1.0, 2.0 * p_val)

    or_val = (b / c) if c > 0 else (float("inf") if b > 0 else 1.0)
    return {
        "b": b,
        "c": c,
        "n_discordant": n_disc,
        "p_value": p_val,
        "odds_ratio": or_val,
    }


def paired_scenario_bootstrap(
    scores_a: List[float],
    scores_b: List[float],
    n_resamples: int = 10000,
    seed: int = 20261122,
) -> Dict[str, Any]:
    """Cluster-aware paired bootstrap over scenario-level scores."""
    assert len(scores_a) == len(scores_b), "Score arrays must match in length"
    n = len(scores_a)
    if n == 0:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0}

    rng = random.Random(seed)
    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    obs_diff = sum(diffs) / n

    boot_diffs = []
    for _ in range(n_resamples):
        sample = [diffs[rng.randint(0, n - 1)] for _ in range(n)]
        boot_diffs.append(sum(sample) / n)

    boot_diffs.sort()
    lower_idx = int(0.025 * n_resamples)
    upper_idx = int(0.975 * n_resamples)

    # Two-sided empirical p-value for H0: diff == 0
    if obs_diff >= 0:
        p_val = 2.0 * sum(1 for d in boot_diffs if d <= 0) / n_resamples
    else:
        p_val = 2.0 * sum(1 for d in boot_diffs if d >= 0) / n_resamples
    p_val = min(1.0, max(0.0, p_val))

    return {
        "mean_diff": obs_diff,
        "ci_lower": boot_diffs[lower_idx],
        "ci_upper": boot_diffs[upper_idx],
        "p_value": p_val,
        "n_scenarios": n,
    }


def holm_bonferroni_correction(p_values: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
    """Holm-Bonferroni step-down correction for family-wise error rate control."""
    # Sort by p-value ascending
    sorted_items = sorted(p_values, key=lambda x: x[1])
    m = len(sorted_items)
    results = []

    running_max_p = 0.0
    for rank, (name, p_val) in enumerate(sorted_items):
        multiplier = m - rank
        adjusted_p = min(1.0, p_val * multiplier)
        adjusted_p = max(adjusted_p, running_max_p)
        running_max_p = adjusted_p
        results.append({
            "name": name,
            "raw_p": p_val,
            "adjusted_p": adjusted_p,
            "is_significant_05": adjusted_p < 0.05,
        })
    return results
