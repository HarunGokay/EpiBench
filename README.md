# EpiBench

**EpiBench** is an open-source, deterministic benchmark for evaluating **epistemic long-term memory architectures** across evolving multi-session histories (`GENERATOR_VERSION = "2.2.0"`, `SCORER_VERSION = "2.0.0"`, `READER_PROMPT_VERSION = "2.0.0"`, `ADAPTER_CONTRACT_VERSION = "2.0.0"`).

---

## What EpiBench Is

Rather than treating memory as static single-turn fact retrieval, EpiBench generates curated, deterministic multi-session world models with ground truth fixed by construction and SHA-256 verified manifests across **nine epistemic tracks**:

- **Track R — Contradiction & Belief Revision**: Tracks evolving values over time (`current`, `previous`, point-in-time `as_of`, and exact transition counts), testing supersession hygiene without leaking stale values.
- **Track D — Dependency Cascades**: Tracks operational decisions anchored on upstream assumptions and evaluates which downstream commitments must be revisited (and which unaffected commitments still stand) when an assumption breaks — including 3-hop zero-overlap transitive cascades under `--profile stress`.
- **Track X — Hidden Constraints**: Evaluates unprompted safety, medical, financial, legal, and physical boundary enforcement when the user asks a neutral planning question — including strictly zero-content-word-overlap cross-domain constraints under `--profile stress`.
- **Track F — Forecast Velocity & Slips**: Measures computed calendar slip counts, exact calendar slip days, and realism calibration from dated commitment, slippage, and completion milestones.
- **Track I — Causal Calibration & Portfolio Aggregation**: Separates clean longitudinal interventions from confounded, unimplemented, and single-observation anecdotal claims — scaling to 16-session longitudinal portfolios under `--profile stress`.
- **Track P — Provenance & False Memory Rejection**: Tests whether memory distinguishes user-authored ground truth from assistant hallucinations or retracted claims.
- **Track K — Static Recall Control**: Baseline recall of stable biographical and project facts buried across multi-session distractors.
- **Track E — Selective Forgetting & Deletion**: Verifies targeted erasure via `adapter.delete()` jointly across direct recall leaks, semantic paraphrase leaks, and retention of non-deleted control facts.
- **Track Q — Uncertainty & Clarification**: Verifies that systems surface ambiguous memorized candidates and ask for clarification rather than guessing.

Every scenario is deterministic from a fixed seed, validated by `validate_pools()` for zero gold-answer leakage into distractors, and paired with two bundled distractor pools: 30 neutral everyday sessions (`src/epibench/data/sample_distractors.json`) and 66 same-domain confuser sessions (`src/epibench/data/same_domain_distractors.json`).

---

## Why EpiBench Exists

Standard long-context and RAG benchmarks heavily favor simple top-K chunk retrieval because:
1. Each question's relevant evidence usually fits inside 1–4 short chunks (~150–300 tokens), allowing flat chunk RAG to dump all relevant turns verbatim inside a 1,500-token budget and rely on a strong reader LLM to reconcile contradictions on the fly.
2. Static retrieval metrics rarely penalize dumping superseded historical facts into current-state queries, missing multi-hop bridge sessions with zero lexical overlap to the query, or dropping standing safety constraints that share no keywords with a user's action prompt.
3. Evaluating memory architectures iteratively with live LLM judges is costly and slows down core retrieval and state-management development.

EpiBench addresses all three gaps through **three evaluation profiles** and a **zero-cost (`$0.00`) offline epistemic-hygiene evaluation suite**:

- **`--profile dev`** (`filler_count = 6`, `distractor_mode = "neutral"`, `token_budget = 1500`): Fast smoke and regression check.
- **`--profile hard`** (`filler_count = 120`, `distractor_mode = "mixed"`, `token_budget = 1500`): Budget-overflow and semantic-interference benchmark interleaving 60 same-domain confuser sessions and 60 neutral sessions (~13.6k history tokens vs a 1,500-token budget).
- **`--profile stress`** (alias `--profile epistemic_stress`; `filler_count = 240`, `distractor_mode = "mixed"`, `token_budget = 1500`): Epistemic-stress regime specifically designed to expose the structural limits of flat chunk RAG:
  - **Evidence-exceeds-budget chains (Tracks R & F)**: 28-step timestamped revision chains (~2,600+ tokens of relevant evidence on one entity) and 24-step forecast/slippage timelines (~2,300+ tokens) that physically cannot fit verbatim inside a 1,500-token budget.
  - **3-hop transitive cascades (Track D)**: 5-session dependency chains where the broken upstream premise shares zero content words with the terminal affected initiatives.
  - **Zero-overlap cross-domain constraints (Track X)**: 12 curated constraints verified to share `0` content words with the user's neutral action query.
  - **16-session longitudinal portfolio aggregation (Track I)**: 4 competing interventions (clean, confounded, unimplemented, anecdotal) spread across 16 sessions (~1,900+ tokens).

---

## How to Use EpiBench

### 1. Install

```bash
git clone https://github.com/HarunGokay/EpiBench
cd EpiBench
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Optional third-party baselines (`mem0ai`, `qdrant-client`, `google-genai`) are isolated under `pip install -e ".[baselines]"`.

### 2. Run the Offline Test Suite & Quickstart (`$0.00`, Zero API Keys)

```bash
pytest
python examples/quickstart.py
epibench validate --profile stress --offline
```

### 3. Run Zero-Cost (`$0.00`) Offline Epistemic-Hygiene Evaluation

Evaluate built-in or custom memory adapters directly on retrieval recall, current-state supersession precision (penalizing stale-value leakage), multi-hop bridge recall, zero-overlap constraint recall (at both 1,500 and tight 600-token budgets), overflow-chain coverage, signal-to-noise density, selective deletion hygiene, and strict `as_of` / token-budget compliance without calling any LLM API:

```bash
epibench eval-offline \
  --profile stress \
  --systems null_memory,oracle,recent_context,vector_rag,hybrid_rag \
  --scenarios-per-track 12 \
  --output results/offline_stress.json
```

Pass `--fake-embedder` for instant CI smoke checks without running local ONNX embeddings.

### 4. Generate and Verify Deterministic Manifests

```bash
epibench manifest --profile dev --output manifests/dev_manifest.json
epibench manifest --profile hard --output manifests/hard_manifest.json
epibench manifest --profile stress --output manifests/stress_manifest.json
```

Committed `v2.2.0` manifests:
- [`manifests/dev_manifest.json`](manifests/dev_manifest.json) (`108` scenarios, `profile = dev`, `filler_count = 6`, `distractor_mode = neutral`, `manifest_sha256 = 5fd9e91ff612bc41c9e6b928c2dd107594d3a91e8f6e214e5988b9d91e562574`)
- [`manifests/hard_manifest.json`](manifests/hard_manifest.json) (`108` scenarios, `profile = hard`, `filler_count = 120`, `distractor_mode = mixed`, `manifest_sha256 = ae2b97960d85c31cdbab067092262ae79e7caf5401fd871c45aa57c7b0291e9c`)
- [`manifests/stress_manifest.json`](manifests/stress_manifest.json) (`108` scenarios, `profile = stress`, `filler_count = 240`, `distractor_mode = mixed`, `manifest_sha256 = dc3cc6c6088c9a8b04a1e37434f2db0f20749cb60e8092c86e02b30c32fa660f`)

### 5. Run a Live End-to-End Evaluation (Optional, Requires API Key)

`--budget-usd` is mandatory on every live runner to enforce a strict hard spend cap:

```bash
export GEMINI_API_KEY="your-api-key"
python -m epibench.runners.run_epibench \
  --profile stress \
  --tracks R,D,X,F,I,P,K,E,Q \
  --systems null_memory,oracle,recent_context,vector_rag,hybrid_rag \
  --scenarios-per-track 12 \
  --seed 20261122 \
  --token-budget 1500 \
  --secondary-audit-rate 0.10 \
  --budget-usd 1.50 \
  --concurrency 4 \
  --output-dir results/stress_run
```

Programmatic tracks (`K`, `R`, `F-slip`, `D-revisit`, `D-stands`, `E-control`) use strict `FINAL ANSWER:` parsing (`src/epibench/judging/programmatic.py`), while open-ended tracks use `RubricJudge` with configurable secondary inter-rater audits (`--secondary-audit-rate`). You can also supply custom distractor pools via `--distractor-file path/to/fillers.json` or sweep distractor depths via `python -m epibench.runners.run_depth_benchmark`.

### 6. Evaluate a Custom Memory Architecture

Implement `BaseMemoryAdapter` from `epibench.adapters.base` (outside this repository to preserve benchmark independence):

```python
from typing import Any, Dict, List, Optional
from epibench.adapters.base import BaseMemoryAdapter, IngestResult, RetrievalResult

class CustomMemoryAdapter(BaseMemoryAdapter):
    async def ingest(self, tenant_id: str, sessions: List[Dict[str, Any]]) -> IngestResult:
        """Append dated sessions incrementally without wiping prior state for tenant_id."""
        ...

    async def retrieve(
        self,
        tenant_id: str,
        query: str,
        token_budget: int = 1500,
        as_of: Optional[str] = None,
    ) -> RetrievalResult:
        """Return bounded context respecting `as_of` (exclude sessions dated after `as_of`)
        and populate metadata['source_session_ids'] for provenance and as_of auditing.
        """
        ...

    async def delete(
        self,
        tenant_id: str,
        session_id: Optional[str] = None,
        target: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return {'supported': True, ...} when selective erasure is implemented,
        or {'supported': False} otherwise.
        """
        ...

    async def cleanup(self, tenant_id: str) -> None:
        """Release all per-tenant state after scenario completion."""
        ...
```

Pass `custom_adapters={"my_system": CustomMemoryAdapter("my_system")}` to either `run_offline_epistemic_eval(...)` (for `$0.00` offline epistemic-hygiene evaluation) or `EpiBenchRunner(...)` (for full live evaluation).

Built-in reference and baseline adapters under `src/epibench/adapters/`:
- `null_memory`: Empty-context lower bound.
- `oracle`: Unbounded `is_evidence=True` upper bound (respecting `as_of` and `delete`).
- `recent_context`: Newest contiguous sessions fitting inside `token_budget`.
- `full_context_unbounded`: All eligible sessions without truncation.
- `vector_rag`: Local FastEmbed (`BAAI/bge-small-en-v1.5`) dense retrieval with `as_of` pre-filtering, separator-aware budgeting, and targeted deletion.
- `hybrid_rag`: Okapi BM25 + dense vector reciprocal rank fusion with `as_of` pre-filtering, separator-aware budgeting, and targeted deletion.
- `mem0`: Optional baseline (`pip install -e ".[baselines]"`).

### 7. Interactive World Simulator (`epibench.environment`)

`EpiBenchEnvironment` and `WorldSimulator` (`src/epibench/environment/`) simulate an evolving multi-session world whose state transitions are emitted as dated dialogue and replayed directly into any `BaseMemoryAdapter`:

```python
from epibench.environment import EpiBenchEnvironment

env = EpiBenchEnvironment(seed=42)
obs = env.reset(num_events=12)
while not env.is_done:
    obs = env.step()
probes = env.generate_probes()
```

---

## Governance & License

- **Contamination resistance**: Public template pools (`split="dev"`) are deterministic and auditable. Calling `split="held_out"` requires supplying a separate private template pool via `custom_pools`.
- **Historical archives**: Pre-2.0.0 artifacts are archived under [`results_archive_invalid/`](results_archive_invalid/README.md) and must not be cited.
- **License**: MIT — see [`LICENSE`](LICENSE).
