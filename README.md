# EpiBench

EpiBench is an open-source benchmark for evaluating long-term memory architectures on **curated, deterministic, templated world models with gold answers fixed by construction**. Rather than evaluating memory purely as static single-turn retrieval, EpiBench measures belief revision, point-in-time (`as_of`) reconstruction, dependency cascades, unprompted constraints, forecast velocity, causal calibration, provenance against assistant hallucinations, targeted erasure, and ambiguity detection across multi-session histories.

Because every scenario is constructed deterministically from validated templates with explicit aliases, computed arithmetic ground truth, and SHA-256 scenario manifests (`GENERATOR_VERSION = "2.1.0"`), runs are reproducible and every scoring decision can be audited line-by-line.

> **Version Notice (`2.1.0`)**: Current version constants are `SCORER_VERSION = "2.0.0"`, `GENERATOR_VERSION = "2.1.0"`, `READER_PROMPT_VERSION = "2.0.0"`, and `ADAPTER_CONTRACT_VERSION = "2.0.0"`. Version `2.1.0` adds 66 curated same-domain confuser distractors, `--distractor-mode`, `--distractor-file`, and `--profile hard` (`120` mixed fillers, ~`13.6k` history tokens vs `1,500`-token budget). All pre-2.0.0 results are archived in [`results_archive_invalid/`](results_archive_invalid/README.md) and must not be cited.

---

## What EpiBench Measures

### Nine Evaluated Tracks (114 Curated Templates)

All 9 tracks are built from validated domain templates (`validate_pools()`), each supplying explicit `gold_aliases` and `forbidden_phrases`, paired with two bundled distractor sets: 30 neutral everyday sessions (`src/epibench/data/sample_distractors.json`) and 66 curated same-domain confuser sessions (`src/epibench/data/same_domain_distractors.json`).

| Track | Templates | Evaluated Capability | Scoring Protocol (`SCORER_VERSION 2.0.0`) |
| --- | ---: | --- | --- |
| **R** | 16 | Contradiction & Belief Revision (current, previous, `as_of`, switch count) | Programmatic `string_match` and `integer_match` on `FINAL ANSWER:` |
| **D** | 14 | Dependency Cascades (invalidated set, unaffected `YES`/`NO`, rationale) | Programmatic `set_overlap` (`F1 == 1.0`), `yes_no`, and LLM rubric |
| **X** | 12 | Hidden Constraints (medical, financial, architectural limits under unprompted planning) | LLM rubric checking constraint compliance |
| **F** | 12 | Forecast Velocity & Slips (computed calendar slip days, realism assessment) | Programmatic `integer_match` (computed from milestone dates) and LLM rubric |
| **I** | 12 | Causal Calibration (tested intervention, confounded, unimplemented, anecdotal) | LLM rubric on causal claim strength |
| **P** | 12 | False Memory Injection & Provenance (rebutting assistant/summary hallucinations) | LLM rubric verifying user fact retention and forbidden-phrase exclusion |
| **K** | 12 | Static Recall Baseline (biographical and project facts across distractors) | Programmatic `string_match` on `FINAL ANSWER:` |
| **E** | 12 | Forgetting & Targeted Deletion (direct leak, semantic leak, control retention) | Joint scenario pass (`control_retained AND leak_free`) after `adapter.delete()` |
| **Q** | 12 | Uncertainty & Clarification (naming both ambiguous candidates rather than guessing) | LLM rubric verifying explicit clarification |

### Seven Independent Dimensions + Control Retention

EpiBench reports seven independent capability dimensions alongside a weighted composite score (missing dimensions are reported as `null` and excluded with weight renormalization; any track flagged `experimental` is excluded from composite scores):

1. **Factual Accuracy** (weight `0.15`): Track K (`static_recall`).
2. **Temporal Consistency** (weight `0.20`): Track R (`current`, `previous`, `as_of`, `count`) and Track F (`forecast_slip`, `forecast_realism`).
3. **Dependency Reasoning** (weight `0.15`): Track D (`revisit_set`, `still_standing`, `invalidation_why`).
4. **Constraint Compliance** (weight `0.10`): Track X (`hidden_constraint`).
5. **Provenance** (weight `0.15`): Track P (`false_memory`).
6. **Calibration** (weight `0.15`): Track I (`causal_calibration`) and Track Q (`ambiguity_clarification`).
7. **Privacy & Deletion** (weight `0.10`): Track E joint pass rate (`control_retained AND leak_free`).
8. **Retention Control** (reported independently, not double-counted in privacy): Track E non-deleted control fact retention.

---

## Benchmark Profiles & Depth Scaling (`--profile dev|hard`)

EpiBench provides two built-in evaluation profiles plus full CLI controls for deeper, higher-budget stress testing (`--filler-count`, `--distractor-mode neutral|same_domain|mixed`, `--distractor-file <path>`, `--token-budget`, `--scenarios-per-track`, and `--budget-usd`):

- **`--profile dev`** (`filler_count = 6`, `distractor_mode = "neutral"`, `token_budget = 1500`): Fast smoke and regression check (~`473` history tokens; fits completely inside a `1,500`-token budget).
- **`--profile hard`** (`filler_count = 120`, `distractor_mode = "mixed"`, `token_budget = 1500`): True budget-overflow and semantic-interference benchmark interleaving 60 same-domain confuser sessions and 60 neutral sessions (~`13,552` average full-context tokens vs `1,500`-token retrieval budget — only ~`11%` of history fits inside budget).

---

## Structured `FINAL ANSWER:` Protocol

To eliminate substring false positives, `ReaderLLM` (`READER_PROMPT_VERSION = "2.0.0"`) instructs the reader model to end every programmatic answer with a dedicated final line:

```text
FINAL ANSWER: <concise answer>
```

Programmatic scorers (`src/epibench/judging/programmatic.py`, `SCORER_VERSION = "2.0.0"`) enforce:
- **Mandatory parsing**: Missing `FINAL ANSWER:` lines fail scoring and increment `parse_failures`.
- **Token-preserving normalization (`string_match`)**: Applies NFKC normalization, currency/thousands normalization (`$1,500` == `1500 dollars`), preserves short tokens (`PostgreSQL 14` does not match `PostgreSQL 16`), and requires whole-token alias matches.
- **Exact polarity (`yes_no`)**: Accepts only exact `YES` or `NO` on the `FINAL ANSWER:` line.
- **Single integer (`integer_match`)**: Accepts exactly one integer (digit or number word) on the `FINAL ANSWER:` line, rejecting multi-number explanations or list numbering.
- **Exact alias set overlap (`set_overlap`)**: Parses semicolon-separated items against explicit per-scenario `gold_aliases` and `unaffected_aliases` (zero hard-coded pool keywords), requiring `F1 == 1.0`.
- **Whole-phrase contamination checks**: `check_forbidden_phrases` and `check_stale_answer` match at word boundaries.

For LLM-judged tracks (`X`, `I`, `P`, `Q`, `D-why`, `F-realism`, `E-direct`, `E-semantic`), `RubricJudge` defaults to a distinct model (`gemini-3.5-flash`) from `ReaderLLM` (`gemini-3.5-flash-lite`), enforces strict JSON parsing with one retry (`judge_error` logged on failure), and runs a 10% secondary judge audit reported with Cohen's kappa.

---

## Quickstart

### 1. Install in a Clean Virtual Environment

```bash
git clone https://github.com/HarunGokay/EpiBench
cd EpiBench
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Optional third-party baselines (`mem0ai`, `qdrant-client`, `google-genai`) are isolated under `pip install -e ".[baselines]"`.

### 2. Run Unit Tests, Offline Quickstart, and Offline Validity Gate (Zero API Key / Zero Network)

```bash
pytest
python examples/quickstart.py
epibench validate --profile hard --offline
```

### 3. Generate and Verify Deterministic Manifests

```bash
epibench manifest --profile dev --output manifests/dev_manifest.json
epibench manifest --profile hard --output manifests/hard_manifest.json
```

Committed `v2.1.0` manifests:
- [`manifests/hard_manifest.json`](manifests/hard_manifest.json) (`108` scenarios, `profile = hard`, `filler_count = 120`, `distractor_mode = mixed`, `manifest_sha256 = 58ed8c9d755dac4e5bdea8394f99f52a344b492bae3545444257f90c197543ba`)
- [`manifests/dev_manifest.json`](manifests/dev_manifest.json) (`108` scenarios, `profile = dev`, `filler_count = 6`, `distractor_mode = neutral`, `manifest_sha256 = be9fc0c37b630673cbb04852b39ed36b9764c661748fe3268fb974b2b902306a`)
- Historical `v2.0.0` dev manifest: [`manifests/dev_manifest_v2_0.json`](manifests/dev_manifest_v2_0.json) (`manifest_sha256 = 4d842cf9bd844a64793a5cd78cacf3757b0dca35141030827fd41bfd9ed6b9cb`)

### 4. Run a Live Evaluation (`--profile hard` Recommended)

`--budget-usd` is mandatory on every live runner to prevent unbounded API spend.

```bash
export GEMINI_API_KEY="your-api-key"
python -m epibench.runners.run_epibench \
  --profile hard \
  --tracks R,D,X,F,I,P,K,E,Q \
  --systems null_memory,oracle,recent_context,full_context_unbounded,vector_rag,hybrid_rag \
  --scenarios-per-track 12 \
  --seed 20261122 \
  --token-budget 1500 \
  --secondary-audit-rate 0.10 \
  --budget-usd 1.00 \
  --concurrency 4 \
  --output-dir results/hard_v2_1
```

For custom stress tests with larger budgets, pass `--filler-count 200 --distractor-mode same_domain`, supply a custom haystack via `--distractor-file path/to/fillers.json`, or sweep depths via `python -m epibench.runners.run_depth_benchmark --depths 6,40,120 --distractor-mode mixed --budget-usd 2.00`.

---

## Validity Gates (`null_memory` vs `oracle`)

Before freezing each release, every track is verified live against the mandatory validity gate (`oracle >= 95.0%` and `null_memory <= 15.0%` per track; Track E evaluated by joint scenario pass rate). All **9/9 tracks passed live** on both `v2.1.0` (`hard`) and `v2.0.0` (`dev`).

### `v2.1.0` Hard-Profile Validity Gate (`filler_count = 120`, `distractor_mode = mixed`)

- **Artifact**: [`results/validity/validity_hard_report.json`](results/validity/validity_hard_report.json)
- **Manifest SHA-256**: `58ed8c9d755dac4e5bdea8394f99f52a344b492bae3545444257f90c197543ba`
- **Command**: `epibench validate --profile hard --scenarios-per-track 12 --budget-usd 0.30 --output-dir results/validity_hard --concurrency 6`
- **Metered Spend**: `$0.027129 USD`

| Track | Scoring Metric | `oracle` (%) | `null_memory` (%) | Status |
| --- | --- | ---: | ---: | --- |
| **R** | `question_accuracy` | 100.00 | 0.00 | PASS |
| **D** | `question_accuracy` | 97.22 | 0.00 | PASS |
| **X** | `question_accuracy` | 100.00 | 0.00 | PASS |
| **F** | `question_accuracy` | 97.22 | 0.00 | PASS |
| **I** | `question_accuracy` | 100.00 | 0.00 | PASS |
| **P** | `question_accuracy` | 100.00 | 0.00 | PASS |
| **K** | `question_accuracy` | 100.00 | 0.00 | PASS |
| **E** | `joint_scenario_pass_rate` | 100.00 | 0.00 | PASS |
| **Q** | `question_accuracy` | 100.00 | 0.00 | PASS |

### `v2.0.0` Dev-Profile Validity Gate (`filler_count = 6`, `distractor_mode = neutral`)

- **Artifact**: [`results/validity/validity_report.json`](results/validity/validity_report.json)

| Track | Scoring Metric | `oracle` (%) | `null_memory` (%) | Status |
| --- | --- | ---: | ---: | --- |
| **R** | `question_accuracy` | 100.00 | 0.00 | PASS |
| **D** | `question_accuracy` | 100.00 | 0.00 | PASS |
| **X** | `question_accuracy` | 100.00 | 0.00 | PASS |
| **F** | `question_accuracy` | 97.22 | 0.00 | PASS |
| **I** | `question_accuracy` | 100.00 | 0.00 | PASS |
| **P** | `question_accuracy` | 100.00 | 0.00 | PASS |
| **K** | `question_accuracy` | 100.00 | 0.00 | PASS |
| **E** | `joint_scenario_pass_rate` | 100.00 | 0.00 | PASS |
| **Q** | `question_accuracy` | 100.00 | 0.00 | PASS |

---

## Measured Built-In Benchmark Results

### 1. Hard-Regime 9-Track Evaluation (`v2.1.0`, `--profile hard`, `filler_count = 120`, `mixed` distractors)

- **Artifact**: [`results/hard_v2_1/results.json`](results/hard_v2_1/results.json) (validated against [`src/epibench/schemas/results_schema.json`](src/epibench/schemas/results_schema.json))
- **Manifest SHA-256**: `58ed8c9d755dac4e5bdea8394f99f52a344b492bae3545444257f90c197543ba`
- **Sample Size**: `N = 108` scenarios (`12` per track, `216` questions per system across all 9 tracks; `1,296` total system-question evaluations; ~`13,552` average full-history tokens vs `token_budget = 1500`).
- **Models & Audit**: Reader `gemini-3.5-flash-lite`, Judge `gemini-3.5-flash`, `secondary_audit_rate = 0.10` (`53` audited decisions, `52/53` agreement = `98.11%`, Cohen's kappa = `0.9574`), `0` parse failures, `0` judge errors, `0` `as_of` violations, `0` unverified `as_of` calls.
- **Total Metered Spend**: `$0.438501 USD` (`$0.85` cap).
- **Command**: `python -m epibench.runners.run_epibench --profile hard --scenarios-per-track 12 --systems null_memory,oracle,recent_context,full_context_unbounded,vector_rag,hybrid_rag --budget-usd 0.85 --output-dir results/hard_v2_1 --concurrency 4`

| System | Composite (%) | Question Acc (%) | 95% Wilson CI (%) | Scenario Pass (%) | Avg Tokens | Avg Ret (ms) | Metered Spend (USD) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `null_memory` | 0.00 | 11.11 | [7.58, 16.00] | 0.00 | 0.0 | 0.00 | 0.010556 |
| `oracle` | 98.87 | 98.15 | [95.34, 99.28] | 96.30 | 115.6 | 0.02 | 0.017338 |
| `recent_context` | 3.10 | 17.13 | [12.69, 22.72] | 0.00 | 1434.0 | 0.15 | 0.039802 |
| `full_context_unbounded` | 96.49 | 95.83 | [92.27, 97.79] | 92.59 | 13552.2 | 0.28 | 0.283254 |
| `vector_rag` | 94.00 | 93.98 | [89.98, 96.45] | 89.81 | 1477.0 | 12.10 | 0.043570 |
| `hybrid_rag` | 96.22 | 96.30 | [92.86, 98.11] | 93.52 | 1480.5 | 15.63 | 0.043981 |

#### Hard-Regime Dimensional Breakdown (`results/hard_v2_1/results.json`)

| System | Factual (%) | Temporal (%) | Dependency (%) | Constraint (%) | Provenance (%) | Calibration (%) | Privacy (%) | Retention Ctrl (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `null_memory` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `oracle` | 100.00 | 96.43 | 97.22 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| `recent_context` | 0.00 | 15.48 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `full_context_unbounded` | 100.00 | 92.86 | 94.44 | 91.67 | 100.00 | 100.00 | 100.00 | 100.00 |
| `vector_rag` | 100.00 | 97.62 | 75.00 | 91.67 | 100.00 | 95.83 | 100.00 | 100.00 |
| `hybrid_rag` | 100.00 | 98.81 | 83.33 | 100.00 | 91.67 | 100.00 | 100.00 | 100.00 |

**Interpretation**: Under `--profile hard` (`~13,552` full history tokens vs `1,500`-token budget with 60 same-domain confusers):
- `recent_context` collapses to `17.13%` accuracy (`3.10%` composite, `0.00%` scenario pass rate), proving true budget overflow.
- `vector_rag` drops to `75.00%` on **Dependency Reasoning** (Track D) and `91.67%` on **Constraint Compliance** (Track X), pulling its scenario pass rate down to `89.81%`.
- `hybrid_rag` also degrades on **Dependency Reasoning** (`83.33%`) and **Provenance** (`91.67%`), because single-query top-k retrieval struggles to co-retrieve all scattered linked decisions amidst same-domain distractors.
- `full_context_unbounded` incurs `6.4x` higher reader spend (`$0.283` vs `$0.044`) and `9.2x` higher token load (`13,552` vs `1,481` tokens) while still dropping below `oracle` on Temporal (`92.86%`), Constraint (`91.67%`), and Dependency (`94.44%`).

---

### 2. Historical `v2.0.0` Dev Evaluation (`filler_count = 6`, `token_budget = 1500`)

- **Artifact**: [`results/dev_v2/results.json`](results/dev_v2/results.json) (evaluated under `GENERATOR_VERSION = "2.0.0"`, manifest [`manifests/dev_manifest_v2_0.json`](manifests/dev_manifest_v2_0.json), `manifest_sha256 = 4d842cf9bd844a64793a5cd78cacf3757b0dca35141030827fd41bfd9ed6b9cb`)
- **Models & Audit**: Reader `gemini-3.5-flash-lite`, Judge `gemini-3.5-flash`, `secondary_audit_rate = 0.10` (`53` audited decisions, `52/53` agreement = `98.11%`, Cohen's kappa = `0.9406`).
- **Total Metered Spend**: `$0.130999 USD` (`$2.00` cap).

| System | Composite (%) | Question Acc (%) | 95% Wilson CI (%) | Scenario Pass (%) | Avg Tokens | Avg Ret (ms) | Metered Spend (USD) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `null_memory` | 0.00 | 11.11 | [7.58, 16.00] | 0.00 | 0.0 | 0.00 | 0.010459 |
| `oracle` | 98.93 | 98.61 | [96.00, 99.53] | 97.22 | 115.5 | 0.02 | 0.017443 |
| `recent_context` | 98.45 | 97.69 | [94.70, 99.01] | 95.37 | 473.4 | 0.05 | 0.025536 |
| `full_context_unbounded` | 98.41 | 98.15 | [95.34, 99.28] | 96.30 | 473.4 | 0.04 | 0.025684 |
| `vector_rag` | 98.69 | 98.15 | [95.34, 99.28] | 96.30 | 494.3 | 4.59 | 0.026007 |
| `hybrid_rag` | 98.87 | 98.15 | [95.34, 99.28] | 96.30 | 494.3 | 4.50 | 0.025870 |

*Note on `null_memory` question accuracy (`11.11%`, `24/216`)*: those 24 passing questions are strictly Track E's empty-memory non-leak questions (`deletion_direct` and `deletion_semantic`); `null_memory` passes `0/12` Track E control-retention questions, giving `0.00%` Track E joint pass rate, `0.00%` scenario pass rate, and `0.00%` composite score.

#### `v2.0.0` Dev Dimensional Breakdown (`results/dev_v2/results.json`)

| System | Factual (%) | Temporal (%) | Dependency (%) | Constraint (%) | Provenance (%) | Calibration (%) | Privacy (%) | Retention Ctrl (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `null_memory` | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `oracle` | 100.00 | 98.81 | 94.44 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| `recent_context` | 100.00 | 96.43 | 94.44 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| `full_context_unbounded` | 100.00 | 98.81 | 94.44 | 100.00 | 100.00 | 95.83 | 100.00 | 100.00 |
| `vector_rag` | 100.00 | 97.62 | 94.44 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| `hybrid_rag` | 100.00 | 96.43 | 97.22 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |

At `filler_count = 6`, total conversation length averages `473.4` tokens—well below `token_budget = 1500`—so all non-null baselines fit 100% of history inside budget. Use `--profile hard` above for selective-retrieval benchmarking.

---

### 3. Depth Stress Test (`v2.0.0`, `filler_count = 6` vs `40`, `neutral` distractors)

- **Artifact**: [`results/depth_v2/results_depth.json`](results/depth_v2/results_depth.json)
- **Sample Size**: `N = 36` scenarios (`12` per track across Tracks `R`, `D`, `X`; `96` questions per system at each depth).
- **Total Metered Spend**: `$0.150411 USD` (`$0.75` cap).

| System | Depth 6 Acc (%) | Depth 6 95% CI (%) | Depth 6 Tokens | Depth 40 Acc (%) | Depth 40 95% CI (%) | Depth 40 Tokens | Depth 40 Scen Pass (%) |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: |
| `null_memory` | 0.00 | [0.00, 3.85] | 0.0 | 0.00 | [0.00, 3.85] | 0.0 | 0.00 |
| `oracle` | 97.92 | [92.72, 99.43] | 160.1 | 98.96 | [94.33, 99.82] | 160.1 | 97.22 |
| `recent_context` | 96.88 | [91.21, 98.93] | 500.0 | 61.46 | [51.46, 70.57] | 1397.2 | 25.00 |
| `full_context_unbounded` | 98.96 | [94.33, 99.82] | 500.0 | 93.75 | [87.03, 97.10] | 2428.5 | 83.33 |
| `vector_rag` | 96.88 | [91.21, 98.93] | 521.3 | 94.79 | [88.38, 97.76] | 1467.3 | 86.11 |
| `hybrid_rag` | 95.83 | [89.77, 98.37] | 521.3 | 96.88 | [91.21, 98.93] | 1467.3 | 91.67 |

---

### 4. Memory Efficiency & Planted-Fact Recall (`scales = 100, 1000` Turns, Local FastEmbed)

- **Artifact**: [`results/efficiency_v2.json`](results/efficiency_v2.json)
- **Embeddings**: Local FastEmbed (`BAAI/bge-small-en-v1.5`), zero LLM API calls, `token_budget = 1500`, 5 spaced factual needles planted across synthetic histories.
- **Command**: `python -m epibench.runners.run_efficiency_benchmark --scales 100,1000 --systems null_memory,oracle,recent_context,vector_rag,hybrid_rag --seed 20261122 --token-budget 1500 --output results/efficiency_v2.json`

| Turns | System | Fact Recall (%) | Ingest Time (s) | Avg Ret Latency (ms) | Avg Context Tokens |
| ---: | --- | ---: | ---: | ---: | ---: |
| 100 | `null_memory` | 0.0 | 0.0000 | 0.00 | 0.0 |
| 100 | `oracle` | 100.0 | 0.0000 | 0.01 | 209.0 |
| 100 | `recent_context` | 60.0 | 0.0000 | 0.03 | 1479.0 |
| 100 | `vector_rag` | 100.0 | 0.7985 | 4.95 | 1580.4 |
| 100 | `hybrid_rag` | 100.0 | 0.1748 | 5.35 | 1576.8 |
| 1000 | `null_memory` | 0.0 | 0.0000 | 0.00 | 0.0 |
| 1000 | `oracle` | 100.0 | 0.0000 | 0.01 | 209.0 |
| 1000 | `recent_context` | 20.0 | 0.0000 | 0.09 | 1494.0 |
| 1000 | `vector_rag` | 100.0 | 1.9802 | 23.83 | 1557.2 |
| 1000 | `hybrid_rag` | 100.0 | 2.1266 | 31.06 | 1574.8 |

---

## Adapter Interface (`ADAPTER_CONTRACT_VERSION = "2.0.0"`)

Implement `BaseMemoryAdapter` from `epibench.adapters.base` to evaluate an external memory system:

```python
from typing import Any, Dict, List, Optional
from epibench.adapters.base import BaseMemoryAdapter, IngestResult, RetrievalResult

class CustomMemoryAdapter(BaseMemoryAdapter):
    async def ingest(self, tenant_id: str, sessions: List[Dict[str, Any]]) -> IngestResult:
        """Append dated sessions incrementally (do not wipe prior state for tenant_id)."""
        ...

    async def retrieve(
        self,
        tenant_id: str,
        query: str,
        token_budget: int = 1500,
        as_of: Optional[str] = None,
    ) -> RetrievalResult:
        """Return bounded context respecting `as_of` (exclude sessions dated after `as_of`).
        Populate metadata['source_session_ids'] so the runner can audit `as_of` compliance.
        """
        ...

    async def delete(
        self,
        tenant_id: str,
        session_id: Optional[str] = None,
        target: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return {'supported': True, ...} when targeted/session erasure is implemented,
        or {'supported': False} otherwise.
        """
        ...

    async def cleanup(self, tenant_id: str) -> None:
        """Release all per-tenant state after scenario completion."""
        ...
```

Built-in adapters under `src/epibench/adapters/`:
- `null_memory`: returns empty context (lower-bound reference).
- `oracle`: returns only evidence sessions (tagged `is_evidence=True`, respecting `as_of` and `delete`; upper-bound reference).
- `recent_context` (aliased as `full_context`): packs the newest contiguous sessions in chronological order within `token_budget` using unified `count_tokens()`.
- `full_context_unbounded`: returns all eligible sessions without token truncation (unbounded ceiling reference).
- `vector_rag`: local FastEmbed (`BAAI/bge-small-en-v1.5`) dense retrieval with `as_of` filtering before ranking, strict separator-aware token budgeting, and targeted deletion.
- `hybrid_rag`: BM25 + dense vector reciprocal fusion with `as_of` filtering before ranking, strict separator-aware token budgeting, and targeted deletion.
- `mem0`: optional baseline (`pip install -e ".[baselines]"`); flags `internal_llm_cost_unmetered = True` because internal extraction LLM spend is not captured by EpiBench's `SpendTracker`.

---

## Interactive Environment (`epibench.environment`)

`EpiBenchEnvironment` and `WorldSimulator` (`src/epibench/environment/`) simulate an evolving organization where initial world state and every state mutation are emitted explicitly as dated multi-turn dialogue, and probes are generated strictly for events that occurred. Every generated probe is verified derivable from the emitted dialogue across 20 deterministic seeds (`tests/test_environment_v2.py`) and replays directly into any `BaseMemoryAdapter` via `env.replay_into_adapter(adapter, tenant_id)`.

```python
from epibench.environment import EpiBenchEnvironment

env = EpiBenchEnvironment(seed=42)
obs = env.reset(num_events=12)
while not env.is_done:
    obs = env.step()
probes = env.generate_probes()
```

---

## Known Limitations & Governance

1. **Public Finite Template Pools**: EpiBench's public `dev` split consists of 114 curated, deterministic templates (`12`–`16` per track). Because the templates are public and finite, the public split does not resist training-data contamination; verified contamination-free scoring requires supplying a separate private held-out pool (`split="held_out"` requires an explicit `custom_pools` mapping and raises `ValueError` otherwise).
2. **Same-Vendor Reader and Judge**: Although the default reader (`gemini-3.5-flash-lite`) and judge (`gemini-3.5-flash`) are distinct models and achieve `98.11%` secondary audit agreement (`kappa = 0.9574` on `hard_v2_1`, `kappa = 0.9406` on `dev_v2`), both models are from the same vendor (Google Gemini). Cross-vendor judge validation is recommended for formal publications.
3. **LLM-Judge Dependence**: Programmatic scoring covers Tracks `K`, `R`, `F-slip`, `D-revisit`, `D-stands`, and `E-control`, while Tracks `X`, `I`, `P`, `Q`, `D-why`, `F-realism`, `E-direct`, and `E-semantic` rely on rubric LLM judging.
4. **Short-Context Saturation on `--profile dev`**: Default `dev` scenarios average ~`473` tokens, which fits inside a `1,500`-token budget. Meaningful comparison of selective memory architectures requires evaluating under `--profile hard` (`120` mixed fillers, ~`13.6k` tokens) or custom `--filler-count` / `--distractor-mode` depth sweeps.
5. **Unmeasured Optional Baselines**: `mem0` is packaged as an optional extra (`[baselines]`) and is not included in the committed core `results/hard_v2_1/results.json` or `results/dev_v2/results.json` runs.
6. **Conflict-of-Interest Policy**: Any memory architecture developed by a benchmark maintainer must live outside the `epibench` repository, implement the public `BaseMemoryAdapter` interface without benchmark modifications, and be evaluated against the same frozen `SCORER_VERSION`, `GENERATOR_VERSION`, seeds, and manifests as external baselines.

---

## Roadmap

- **Procedural Entity & Timeline Generation**: Planned future upgrade to synthesize novel entity names, projects, and multi-hop dependency graphs dynamically across seeds while preserving exact computed ground truth.
- **Private Held-Out Evaluation Split**: Maintained separately from the public repository for contamination-resistant leaderboard verification.
- **Cross-Vendor Judge Ensembles**: Optional multi-vendor rubric judging (e.g., open-weights + frontier cross-vendor audits) alongside the built-in secondary audit kappa report.

---

## License

MIT — see [`LICENSE`](LICENSE).
