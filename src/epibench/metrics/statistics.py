"""EpiBench Statistical Suite.

Implements cluster-aware scenario-level and paired question-level statistics:
- Exact two-sided McNemar test on discordant question pairs
- Paired scenario bootstrap with 95% confidence intervals (B=10,000)
- Wilson score confidence intervals for binomial endpoints
- Holm-Bonferroni step-down correction for multiple testing
- Pairwise system comparison compiler across all evaluated architectures
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple


def wilson_ci(successes: int, total: int, confidence: float = 0.95) -> Dict[str, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if total <= 0:
        return {"proportion": 0.0, "lower": 0.0, "upper": 0.0}

    z = 1.95996  # 95% two-sided normal quantile
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
    """Exact two-sided McNemar test using the Binomial distribution on discordant pairs.

    `b`: system A correct, system B incorrect
    `c`: system A incorrect, system B correct
    """
    n_disc = b + c
    if n_disc == 0:
        return {
            "b": b,
            "c": c,
            "n_discordant": 0,
            "p_value": 1.0,
            "odds_ratio": 1.0,
            "odds_ratio_infinite": False,
        }

    k = min(b, c)
    p_val = 0.0
    for i in range(k + 1):
        p_val += math.comb(n_disc, i) * (0.5 ** n_disc)
    p_val = min(1.0, 2.0 * p_val)

    is_inf = (c == 0 and b > 0)
    or_val: Optional[float] = None if is_inf else (round(b / c, 4) if c > 0 else 1.0)
    return {
        "b": b,
        "c": c,
        "n_discordant": n_disc,
        "p_value": round(p_val, 8),
        "odds_ratio": or_val,
        "odds_ratio_infinite": is_inf,
    }


def paired_scenario_bootstrap(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    n_resamples: int = 10000,
    seed: int = 20261122,
) -> Dict[str, Any]:
    """Cluster-aware paired bootstrap over scenario-level scores."""
    assert len(scores_a) == len(scores_b), "Score arrays must match in length"
    n = len(scores_a)
    if n == 0:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0, "n_scenarios": 0}

    rng = random.Random(seed)
    diffs = [float(a) - float(b) for a, b in zip(scores_a, scores_b)]
    obs_diff = sum(diffs) / n

    boot_diffs: List[float] = []
    for _ in range(n_resamples):
        sample = [diffs[rng.randint(0, n - 1)] for _ in range(n)]
        boot_diffs.append(sum(sample) / n)

    boot_diffs.sort()
    lower_idx = int(0.025 * n_resamples)
    upper_idx = min(n_resamples - 1, int(0.975 * n_resamples))

    if obs_diff > 0:
        p_val = 2.0 * sum(1 for d in boot_diffs if d <= 0) / n_resamples
    elif obs_diff < 0:
        p_val = 2.0 * sum(1 for d in boot_diffs if d >= 0) / n_resamples
    else:
        p_val = 1.0
    p_val = min(1.0, max(0.0, p_val))

    return {
        "mean_diff": round(obs_diff, 6),
        "ci_lower": round(boot_diffs[lower_idx], 6),
        "ci_upper": round(boot_diffs[upper_idx], 6),
        "p_value": round(p_val, 6),
        "n_scenarios": n,
    }


def holm_bonferroni_correction(p_values: List[Tuple[str, float]]) -> List[Dict[str, Any]]:
    """Holm-Bonferroni step-down correction for family-wise error rate control."""
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
            "raw_p": round(p_val, 8),
            "adjusted_p": round(adjusted_p, 8),
            "is_significant_05": adjusted_p < 0.05,
        })
    return results


def compute_pairwise_comparisons(
    scenario_results: List[Dict[str, Any]],
    systems: Sequence[str],
    n_resamples: int = 10000,
    seed: int = 20261122,
) -> Dict[str, Any]:
    """Compute exact McNemar, paired scenario bootstrap, and Holm-Bonferroni across all pairs."""
    sys_list = list(systems)
    comparisons: Dict[str, Dict[str, Any]] = {}
    mcnemar_pvals: List[Tuple[str, float]] = []
    boot_pvals: List[Tuple[str, float]] = []

    for i in range(len(sys_list)):
        for j in range(i + 1, len(sys_list)):
            sys_a = sys_list[i]
            sys_b = sys_list[j]
            pair_key = f"{sys_a}_vs_{sys_b}"

            b_cnt = 0  # A right, B wrong
            c_cnt = 0  # A wrong, B right
            n_paired_q = 0
            sc_scores_a: List[float] = []
            sc_scores_b: List[float] = []

            for sc in scenario_results:
                sr = sc.get("system_results", {})
                if sys_a not in sr or sys_b not in sr:
                    continue
                res_a = sr[sys_a]
                res_b = sr[sys_b]

                # Scenario-level score (joint pass 1/0 for Track E, scenario_accuracy for others)
                if sc.get("track") == "E":
                    sc_scores_a.append(1.0 if res_a.get("scenario_pass") else 0.0)
                    sc_scores_b.append(1.0 if res_b.get("scenario_pass") else 0.0)
                else:
                    sc_scores_a.append(float(res_a.get("scenario_accuracy", 0.0)))
                    sc_scores_b.append(float(res_b.get("scenario_accuracy", 0.0)))

                qa_map = {q["question_id"]: bool(q.get("is_correct", False)) for q in res_a.get("questions", [])}
                qb_map = {q["question_id"]: bool(q.get("is_correct", False)) for q in res_b.get("questions", [])}
                for qid in sorted(set(qa_map.keys()) & set(qb_map.keys())):
                    n_paired_q += 1
                    a_ok = qa_map[qid]
                    b_ok = qb_map[qid]
                    if a_ok and not b_ok:
                        b_cnt += 1
                    elif not a_ok and b_ok:
                        c_cnt += 1

            mc = exact_mcnemar_test(b_cnt, c_cnt)
            boot = paired_scenario_bootstrap(
                sc_scores_a,
                sc_scores_b,
                n_resamples=n_resamples,
                seed=seed + i * 97 + j,
            )

            comparisons[pair_key] = {
                "system_a": sys_a,
                "system_b": sys_b,
                "paired_questions": n_paired_q,
                "mcnemar": mc,
                "paired_scenario_bootstrap": boot,
            }
            mcnemar_pvals.append((pair_key, float(mc["p_value"])))
            boot_pvals.append((pair_key, float(boot["p_value"])))

    if mcnemar_pvals:
        mc_holm = {item["name"]: item for item in holm_bonferroni_correction(mcnemar_pvals)}
        boot_holm = {item["name"]: item for item in holm_bonferroni_correction(boot_pvals)}
        for pair_key, entry in comparisons.items():
            entry["mcnemar"]["holm_adjusted_p"] = mc_holm[pair_key]["adjusted_p"]
            entry["mcnemar"]["holm_significant_05"] = mc_holm[pair_key]["is_significant_05"]
            entry["paired_scenario_bootstrap"]["holm_adjusted_p"] = boot_holm[pair_key]["adjusted_p"]
            entry["paired_scenario_bootstrap"]["holm_significant_05"] = boot_holm[pair_key]["is_significant_05"]

    return comparisons
