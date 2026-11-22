# EpiBench

EpiBench is an empirical benchmark for evaluating whether AI agents can maintain a coherent, evolving understanding of the world over long periods of time. Rather than evaluating memory purely as static information retrieval, EpiBench evaluates whether an agent can maintain an evolving world model: remembering accurately, revising beliefs intelligently, reasoning across dependencies, and maintaining epistemic integrity across thousands of interactions.

```
       Evolving World State (Entities, Relationships, Constraints, Projects)
                                      │
            ┌─────────────────────────┴─────────────────────────┐
            ▼                                                   ▼
  Sequential Stream Interaction                       Ground-Truth State Graph
  (Updates, Reversals, Deletions)                     (Temporal Ledgers, DAGs)
            │                                                   │
            ▼                                                   ▼
     Memory Architecture                                Six-Dimensional
  (RAG / Mem0 / Graph / Cache) ───────── Retrieval ────────▶ Evaluation
                                                                │
     ┌──────────────────────┬──────────────────────┬────────────┴─────────┐
     ▼                      ▼                      ▼                      ▼
  Factual & Temporal    Dependency Cascades    Provenance & Trust    Privacy & Calibration
```

---

## Core Research Questions

Conventional memory benchmarks test single-turn recall over static contexts. EpiBench evaluates seven fundamental world-model capabilities:

1. **Long-Term Recall**: Can an agent remember facts and commitments stated months ago?
2. **Belief Revision**: Can it recognize when something it previously believed is no longer true, and identify superseded statements?
3. **Dependency Cascades**: Can it identify every downstream decision and commitment that must be reconsidered when an underlying assumption changes?
4. **Epistemic Provenance**: Can it distinguish user facts from assistant hallucinations, misleading summaries, and retracted third-party claims?
5. **Unprompted Constraints**: Can it respect previously stated boundaries and limits when answering unrelated planning queries without explicit reminder?
6. **Calibration & Uncertainty**: Can it recognize when information is ambiguous, contradictory, or missing, and ask clarifying questions instead of guessing?
7. **Verifiable Erasure**: When asked to delete sensitive information, can it ensure that the deleted fact is omitted across direct, semantic, and derived queries?

---

## The Nine Evaluated Tracks

EpiBench evaluates memory architectures across nine distinct capability tracks:

| Track | Category | Evaluated Capability | Scoring Method |
| --- | --- | --- | --- |
| **Track R** | Contradiction & Belief Revision | Resolving timeline updates, identifying current vs historical as-of states, tracking authoritative sources, and identifying superseded statements. | Substring, integer match, and rubric |
| **Track D** | Dependency Cascades | Propagating the downstream consequences of an invalidated assumption to all dependent decisions while leaving unaffected commitments standing. | Set overlap F1 and LLM rubric on invalidation rationale |
| **Track X** | Hidden Constraints | Respecting previously stated limits (medical, financial, architectural) when answering unprompted planning queries. | LLM rubric checking constraint enforcement |
| **Track F** | Forecast Velocity & Slips | Calculating cumulative schedule slip days across repeated delays and assessing whether aggressive new targets are realistic. | Integer matching on slip days and LLM rubric on realism |
| **Track I** | Causal Calibration | Distinguishing valid intervention comparisons from confounded, unimplemented, or anecdotal single-observation claims. | LLM rubric on causal claim strength |
| **Track P** | False Memory Injection & Provenance | Rebutting assistant hallucinations, unconfirmed summaries, and retracted user statements in favor of authentic user facts. | LLM rubric checking user fact retention and distractor exclusion |
| **Track K** | Static Recall Baseline | Exact factual recall of unchanging biographical and project facts across multi-session haystacks. | Substring and exact token match |
| **Track E** | Forgetting & Deletion | Verifying that explicitly erased credentials or private records are completely withheld from direct and semantic retrieval, while non-deleted context remains intact. | Distractor exclusion check and LLM rubric on refusal |
| **Track Q** | Uncertainty & Clarification | Detecting ambiguous entities, ambiguous relative dates, or unconfirmed conflicting claims, and generating clarifying questions rather than guessing ungrounded answers. | LLM rubric on clarification requests |

---

## Six Independently Reported Metrics

EpiBench avoids single aggregate scores. Every evaluation computes six independent dimensional scores alongside an optional composite score:

1. **Factual Accuracy**: Did the agent retrieve and state the correct established information?
2. **Temporal Consistency**: Did it understand when facts were valid and resolve supersessions accurately?
3. **Dependency Reasoning**: Did it identify downstream consequences when governing assumptions changed?
4. **Provenance**: Can it identify the origin and authority of information, rejecting ungrounded assistant hallucinations?
5. **Calibration**: Does it recognize uncertainty and ask clarifying questions rather than guessing?
6. **Privacy & Deletion**: Does it respect forgetting requests across direct and semantic queries without data leakage?

### Resource & Efficiency Metrics

Every run tracks operational efficiency:
- **Token Consumption**: Average input context tokens delivered per query.
- **Retrieval Latency**: Milliseconds per query retrieval step.
- **Memory Overhead**: Storage growth and ingestion wall-clock time per turn.
- **Cost per Successful Answer**: Total cumulative inference spend ($USD) divided by correct answers.
- **Depth Degradation**: Accuracy retention curve as conversational history scales (100 to 50,000+ turns).

---

## Interactive Environment & World Simulator

EpiBench includes an interactive environment module (`epibench.environment`) that simulates an evolving organization:

- **World Simulator (`WorldSimulator`)**: Procedurally generates entities (people, projects, budgets, deadlines, infrastructure), active dependencies, and an evolving timeline of events (promotions, deadline changes, budget revisions, dependency breaks, deletions, and rumors).
- **Interactive Environment (`EpiBenchEnvironment`)**: Exposes a step-based interaction loop (`reset`, `step`, `observe`) where memory systems receive conversation streams sequentially over time.
- **Ground-Truth Evaluation Engine**: Evaluates memory systems across an entire sequence of events rather than isolated query-answer pairs.

```python
from epibench.environment import EpiBenchEnvironment

env = EpiBenchEnvironment(seed=42)
obs = env.reset(num_events=20)

while not env.is_done:
    # Observe incoming multi-turn conversation event
    print(f"[{obs.timestamp}] Event: {obs.event_type}")
    # Ingest event turns into your memory store
    obs = env.step()

# Generate ground-truth probe questions derived from the world state graph
probes = env.generate_probes()
for probe in probes:
    print(f"[{probe.dimension}] Q: {probe.question}")
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/your-username/epibench.git
cd epibench
pip install -e .
```

### 2. Verify Installation with Unit Tests

```bash
pytest tests/
```

### 3. Run the In-Memory Quickstart

```bash
python examples/quickstart.py
```

### 4. Run Benchmark Scenarios with the LLM Reader

Set your Gemini API key (users supply their own API key; EpiBench incurs zero proprietary service fees):

```bash
export GEMINI_API_KEY="your-api-key"
python runners/run_epibench.py --tracks R,D,X,P,E,Q --systems vector_rag,hybrid_rag
```

### 5. Distractor Data & Offline Operation

EpiBench bundles 30 non-evidence filler sessions in `data/sample_distractors.json` so all scenarios run out of the box without external downloads.

To run depth stress tests using the complete LongMemEval distractor pool, download `longmemeval_s_cleaned.json` from the Hugging Face repository `xiaowu0162/longmemeval-cleaned` (MIT license) and place it at `data/benchmarks/longmemeval_s.json`.

---

## Adapter Interface

Implement `BaseMemoryAdapter` from `adapters/base.py` to evaluate your own memory architecture:

```python
from adapters.base import BaseMemoryAdapter, IngestResult, RetrievalResult

class CustomMemoryAdapter(BaseMemoryAdapter):
    async def ingest(self, tenant_id: str, sessions: list[dict]) -> IngestResult:
        # Ingest dated conversational sessions
        ...

    async def retrieve(
        self, tenant_id: str, query: str, token_budget: int = 1500, as_of: str | None = None
    ) -> RetrievalResult:
        # Retrieve bounded text context relevant to query as of a specific date
        ...

    async def delete(self, tenant_id: str, session_id: str | None = None, target: str | None = None) -> dict:
        # Optional: erase specific session or entity records
        ...
```

The repository includes baseline adapters in `adapters/`:
- `vector_rag`: dense embedding similarity search using FastEmbed and cosine ranking.
- `hybrid_rag`: BM25 keyword matching combined with dense vector retrieval.
- `mem0`: memory extraction and retrieval via the Mem0 library.
- `full_context`: raw dialogue turn window within the token budget.

---

## Memory Efficiency & Scale Research

EpiBench includes a dedicated scale runner (`runners/run_efficiency_benchmark.py`) to evaluate memory degradation as conversation history grows:

```bash
python runners/run_efficiency_benchmark.py --scales 100,1000,5000 --systems vector_rag,hybrid_rag,full_context
```

### Illustrative Scale Experiment Design

| Scale (Turns) | Full Context | Vector RAG | Mem0 | Hybrid RAG |
| --- | --- | --- | --- | --- |
| **100 turns** | High | High | High | High |
| **1,000 turns** | Degrades | High | High | High |
| **10,000 turns** | Context limit | Moderate | Moderate | Moderate |
| **50,000 turns** | Infeasible | Retrieval decay | Retrieval decay | Retrieval decay |

---

## Contamination Resistance: Evaluation Tiers

To prevent benchmark overfitting and training data leakage, EpiBench implements three evaluation tiers:

| Tier | Visibility | Purpose |
| --- | --- | --- |
| **Public** | Fully available in repo | Local development, adapter debugging, and smoke tests |
| **Validation** | Parameterized seeds | Hyperparameter tuning and model configuration |
| **Hidden** | Held-out private seeds | Official benchmark evaluation and verification |

Scenarios are procedurally generated from dynamic entity-relationship graphs. Varying seeds produces unique permutations of names, projects, timelines, and dependencies while preserving deterministic ground-truth answers.

---

## Pilot Smoke Test Results

Raw preliminary results from the initial pilot run are saved in [results/pilot/](results/pilot/). The pilot serves as an end-to-end smoke test (21 scenarios, 42 questions per system) using a single reader model (`gemini-3.5-flash-lite`, temperature 0) to verify runner execution.

### Questions Passed by Track (Raw Counts)

| Track | Questions | mem0 | vector_rag | hybrid_rag | full_context |
| --- | --- | --- | --- | --- | --- |
| **R: Revision** | 12 | 12/12 | 11/12 | 11/12 | 5/12 |
| **D: Dependency** | 9 | 6/9 | 6/9 | 2/9 | 6/9 |
| **X: Constraints** | 3 | 3/3 | 3/3 | 3/3 | 3/3 |
| **F: Forecast** | 9 | 6/9 | 4/9 | 2/9 | 2/9 |
| **I: Causality** | 3 | 2/3 | 3/3 | 3/3 | 1/3 |
| **P: Provenance** | 3 | 2/3 | 3/3 | 2/3 | 3/3 |
| **K: Recall** | 3 | 3/3 | 3/3 | 1/3 | 3/3 |

### Overall Smoke Test Totals

| System | Questions Passed | Scenarios Passed |
| --- | --- | --- |
| `mem0` | 34/42 | 15/21 |
| `vector_rag` | 33/42 | 14/21 |
| `hybrid_rag` | 24/42 | 11/21 |
| `full_context` | 23/42 | 10/21 |

The smoke test shows that standard retrieval baselines struggle particularly on dependency cascades (Track D), repeated forecasting slips (Track F), and unprompted constraints (Track X).

---

## Benchmark Credibility & EpiBench 2.0 Roadmap

The current pilot represents an initial prototype constrained by budget and compute availability. To establish full academic and empirical credibility, EpiBench is actively building towards the following targets:

### Current Prototype vs. Proposed EpiBench 2.0 Targets

| Component | Current Prototype | Proposed EpiBench 2.0 Target |
| --- | --- | --- |
| **Scenarios** | 21 | 500–1,000 |
| **Questions** | 42 / system | 5,000+ |
| **Evaluated Systems** | 4 baseline adapters | 15–20 architectures |
| **Reader Models** | 1 (`gemini-3.5-flash-lite`) | 5+ open and proprietary models |
| **Independent Evaluators** | 1 | 3+ cross-validating judges |
| **Repeated Runs** | 1 | 5+ seeded runs with confidence bounds |

### Credibility Commitments

Future major releases of EpiBench will adhere to standard benchmark governance:
- **Independent Maintainers**: Community-guided review and maintenance.
- **Blind Evaluation**: Standardized submission harnesses with blind testing of memory systems.
- **Preregistered Scoring**: Public criteria fixed before evaluation runs commence.
- **Rigorous Statistics**: Confidence intervals and McNemar significance testing on all comparisons.
- **Versioned & Immutable Snapshots**: Cryptographically hashed scenario releases.
- **Hidden Test Split**: An unreleased held-out split reserved exclusively for verified scoring.

**Core Objective**: A researcher should be able to reproduce all reported figures without relying on the benchmark author's implementation or judgment.

---

## Future Public Leaderboard Specification

Submissions to the planned public leaderboard will report performance across multiple independent dimensions:

### Illustrative Leaderboard Design

| System Architecture | Overall Composite | Factual Accuracy | Temporal Consistency | Dependency Cascades | Epistemic Provenance | Calibration & Clarity | Privacy Compliance | Cost / Query ($) | Context Tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Memory System A** | 89.2% | 96% | 94% | 87% | 91% | 85% | 100% | $0.0008 | 450 |
| **Memory System B** | 85.6% | 94% | 88% | 90% | 84% | 80% | 95% | $0.0012 | 680 |
| **Memory System C** | 79.4% | 91% | 82% | 75% | 78% | 74% | 88% | $0.0004 | 1,200 |
| **Memory System D** | 71.8% | 85% | 68% | 74% | 72% | 65% | 80% | $0.0002 | 1,480 |

### Submission Metadata Requirements

Every submission must document:
1. Memory architecture and indexing strategy.
2. Underlying embedding and language models.
3. Reader model used for question generation.
4. Token budget and retrieval limits.
5. Ingestion and query compute costs.

---

## License

MIT License. Copyright (c) 2026 Harun Gokay.
