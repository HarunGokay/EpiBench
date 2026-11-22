# EpiBench

EpiBench is a benchmark for evaluating long-term conversational memory systems. While LongMemEval covers static recall and knowledge updates, EpiBench tests state revision, dependency invalidation, unprompted constraints, forecasting slips, causal calibration, and provenance.

## Status

Early pilot. Small sample, one reader model, results are preliminary.

## Quickstart

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/your-username/epibench.git
cd epibench
pip install -e .
```

Run the quickstart script to generate a scenario and test an in-memory dictionary adapter:

```bash
python examples/quickstart.py
```

To run benchmarks with the LLM reader or rubric judge, set your Gemini API key:

```bash
export GEMINI_API_KEY="your-api-key"
python runners/run_epibench.py --tracks R D X --systems vector_rag hybrid_rag
```

Distractor data:
EpiBench bundles 30 filler sessions in `data/sample_distractors.json` so scenarios run without downloading external files. To run large depth tests with the full LongMemEval distractor set, download `longmemeval_s_cleaned.json` from the Hugging Face repository `xiaowu0162/longmemeval-cleaned` (upstream project is released under the MIT license) and save it to `data/benchmarks/longmemeval_s.json`.

## The Seven Tracks

| Track | What it checks | How gold answers are computed |
| --- | --- | --- |
| R: Revision & Recency | Returning current values and point-in-time states | Substring and integer matching on updated values |
| D: Dependency Propagation | Tracking decisions broken when assumptions change | Set overlap F1 and LLM rubric on invalidation reasons |
| X: Hidden Constraints | Respecting previously stated limits without reminder | LLM rubric checking constraint enforcement |
| F: Forecast Velocity | Calculating project delays and schedule realism | Integer matching on slip days and LLM rubric on realism |
| I: Causal Calibration | Distinguishing valid interventions from confounds | LLM rubric on causal claim strength |
| P: Provenance | Keeping user facts and rejecting false assistant claims | LLM rubric checking user fact retention and distractor exclusion |
| K: Recall Baseline | Direct factual recall across sessions | Substring matching on static user facts |

## Adapter Interface

Implement `BaseMemoryAdapter` from `adapters/base.py` to test your own memory system:

```python
from adapters.base import BaseMemoryAdapter, IngestResult, RetrievalResult

class MyMemoryAdapter(BaseMemoryAdapter):
    async def ingest(self, tenant_id: str, sessions: list[dict]) -> IngestResult:
        # Ingest sessions into your memory store
        ...

    async def retrieve(
        self, tenant_id: str, query: str, token_budget: int = 1500, as_of: str | None = None
    ) -> RetrievalResult:
        # Return retrieved text context and token count
        ...
```

Subclass `BaseMemoryAdapter`, implement `ingest` and `retrieve`, and pass your adapter instance to `EpiBenchRunner`.

## Systems Included

The repository includes four baseline adapters in `adapters/`:
- `vector_rag`: dense embedding similarity search using FastEmbed and in-memory cosine ranking.
- `hybrid_rag`: BM25 keyword matching combined with dense vector retrieval.
- `mem0`: memory extraction and recall via the Mem0 library.
- `full_context`: baseline returning raw conversation turns within the token budget.

## Pilot Results

Raw results are saved in [results/pilot/](results/pilot/). The pilot is a smoke test across 21 scenarios and 42 questions per system using a single reader model (`gemini-3.5-flash-lite`, temperature 0). It confirms that the evaluation harness runs end to end and indicates that tracks D, X, F, and P present challenges for standard retrieval baselines.

### Questions Passed by Track

| Track | Questions | mem0 | vector_rag | hybrid_rag | full_context |
| --- | --- | --- | --- | --- | --- |
| R: Revision | 12 | 12/12 | 11/12 | 11/12 | 5/12 |
| D: Dependency | 9 | 6/9 | 6/9 | 2/9 | 6/9 |
| X: Constraints | 3 | 3/3 | 3/3 | 3/3 | 3/3 |
| F: Forecast | 9 | 6/9 | 4/9 | 2/9 | 2/9 |
| I: Causality | 3 | 2/3 | 3/3 | 3/3 | 1/3 |
| P: Provenance | 3 | 2/3 | 3/3 | 2/3 | 3/3 |
| K: Recall | 3 | 3/3 | 3/3 | 1/3 | 3/3 |

### Overall Totals

| System | Questions Passed | Scenarios Passed |
| --- | --- | --- |
| mem0 | 34/42 | 15/21 |
| vector_rag | 33/42 | 14/21 |
| hybrid_rag | 24/42 | 11/21 |
| full_context | 23/42 | 10/21 |

## Not Implemented

Track Q (conversational clarification before acting), Track E (post-erasure verification across database tables), and Tier 3 (interactive multi-turn knowledge exploration) are ideas, not code.

## Known Problems

See [LIMITS.md](LIMITS.md) for limitations regarding sample size, reader dependency, and scoring methods.

## License

MIT. Copyright (c) 2026 Harun Gokay.
