# Changelog

> **IMPORTANT — PRE-2.0.0 RESULTS INVALIDITY NOTICE**
>
> All benchmark results produced prior to `SCORER_VERSION 2.0.0` (archived in
> [`results_archive_invalid/`](results_archive_invalid/README.md)) were generated
> with flawed programmatic scorers, unverified deletion replay, ignored `as_of`
> cutoffs, oldest-first context truncation, and stale scenario generators.
> **Those pre-2.0.0 numbers are invalid and must not be cited or compared against
> `2.0.0` runs.**

---

## 2.2.0 (Epistemic-Stress Profile & Zero-Cost Offline Evaluation Suite)

Version `2.2.0` upgrades scenario generation (`GENERATOR_VERSION = "2.2.0"`) with the `--profile stress` (alias `epistemic_stress`) regime and a zero-cost (`$0.00`) offline epistemic-hygiene runner while keeping `SCORER_VERSION = "2.0.0"`, `READER_PROMPT_VERSION = "2.0.0"`, and `ADAPTER_CONTRACT_VERSION = "2.0.0"` frozen.

### Highlights
- **`--profile stress` (alias `epistemic_stress`)**: Adds a 240-filler mixed distractor regime (`~28k+` history tokens vs `1,500`-token budget) with four epistemic upgrades engineered to expose the structural limits of flat chunk-based RAG:
  1. **Evidence-Exceeds-Budget Revision & Forecast Chains (Tracks R & F)**: Expands Track R to 28 scattered timestamped revisions (`~2,600+` tokens of pure evidence on a single entity) and Track F to 24 dated schedule commitments/slips/checkpoints (`~2,300+` tokens) so raw evidence physically overflows a `1,500`-token budget.
  2. **3-Hop Transitive Dependency Cascades (Track D)**: Adds 12 curated 5-session 3-hop dependency chains (`s_hop1_premise -> s_hop2_bridge -> s_hop3_terminal` + `s_unaffected` + `s_hop4_trigger`) with verified zero content-word overlap between the breaking trigger and terminal affected initiatives.
  3. **Zero-Overlap Cross-Domain Constraints (Track X)**: Adds 12 curated cross-domain safety, legal, medical, and physical constraints verified by `validate_pools()` to share `0` content words with the neutral user action query.
  4. **16-Session Longitudinal Portfolio Aggregation (Track I)**: Adds 12 curated 16-session longitudinal portfolios (`~1,900+` tokens of relevant experiment records) spanning clean, confounded, unimplemented, and single-observation interventions.
- **Zero-Cost (`$0.00`) Offline Epistemic-Hygiene Suite (`epibench eval-offline`)**: Added `src/epibench/runners/offline_eval.py` measuring gold-session recall, current-state supersession precision (`1 - stale_leak_rate`), multi-hop bridge/terminal recall, zero-overlap constraint recall at both 1,500 and 600-token budgets, overflow-chain coverage, signal-to-noise density, selective deletion hygiene, and strict `as_of` / token-budget compliance with zero LLM API spend.
- **Manifests & Documentation**: Archived `manifests/hard_manifest_v2_1.json`, committed `v2.2.0` `manifests/dev_manifest.json`, `manifests/hard_manifest.json`, and `manifests/stress_manifest.json`, and streamlined `README.md` to focus strictly on what EpiBench is, why it exists, and how to use it (removing embedded performance tables).

---

## 2.1.0 (Hard-Regime Profile & Same-Domain Distractors)

Version `2.1.0` upgrades scenario generation (`GENERATOR_VERSION = "2.1.0"`) while keeping `SCORER_VERSION = "2.0.0"`, `READER_PROMPT_VERSION = "2.0.0"`, and `ADAPTER_CONTRACT_VERSION = "2.0.0"` frozen.

### Highlights
- **66 Curated Same-Domain Confuser Distractors**: Added `src/epibench/data/same_domain_distractors.json` covering software engineering, operations, travel/logistics, personal finance, health/fitness, home renovation, legal/compliance, and education/career, validated inside `validate_pools()` for zero entity-gold contamination.
- **Distractor Modes & Custom Haystacks**: Added `--distractor-mode neutral|same_domain|mixed` and `--distractor-file <path>` across `DistractorPool`, `ScenarioGenerator`, `generate_manifest`, `epibench manifest`, `epibench validate`, `run_epibench.py`, and `run_depth_benchmark.py`.
- **`--profile hard` Budget-Overflow Preset**: Added `BENCHMARK_PROFILES` (`dev` and `hard`). `--profile hard` uses `filler_count = 120` with `distractor_mode = "mixed"` (60 same-domain + 60 neutral fillers, ~`13,552` average history tokens vs `token_budget = 1500` so only ~`11%` of history fits inside budget).
- **Hardened Track X & Track F Clarity**: Reduced mean content-word lexical overlap on Track X (`HIDDEN_CONSTRAINT_POOLS`) to `0.42` tokens per template (8/12 templates at zero overlap) and added explicit schedule durations (`19-day`, `5-day`, `13-day`) to Track F's three on-time control templates (`f_09`–`f_11`) caught during the mandatory pre-freeze live validity gate.
- **Strict Separator-Aware RAG Token Budgeting**: Updated `VectorRAGAdapter` and `HybridRAGAdapter` to account for `"\n\n---\n\n"` joiner tokens so retrieved contexts strictly never exceed `token_budget`.
- **Committed Frozen `v2.1.0` Artifacts**:
  - Archived `manifests/dev_manifest_v2_0.json` for historical `dev_v2` provenance and committed `v2.1.0` `manifests/dev_manifest.json` and `manifests/hard_manifest.json`.
  - Passed live `epibench validate --profile hard` across all 9/9 tracks (`results/validity/validity_hard_report.json`, Oracle `97.22%`–`100.00%`, Null `0.00%`).
  - Committed full live 9-track hard-regime baseline evaluation (`results/hard_v2_1/results.json`, `N = 108` scenarios / `216` questions per system across 6 built-in architectures; `recent_context` collapses to `17.13%` accuracy, `vector_rag` drops to `75.00%` on Track D Dependency Reasoning, `hybrid_rag` drops to `83.33%` on Track D).

---

## 2.0.0 (Integrity Freeze & Complete Overhaul)

Version `2.0.0` introduces an audited, standalone, deterministic evaluation suite
with `SCORER_VERSION = "2.0.0"`, `GENERATOR_VERSION = "2.0.0"`,
`READER_PROMPT_VERSION = "2.0.0"`, and `ADAPTER_CONTRACT_VERSION = "2.0.0"`.

### Packaging & Dependencies (`P1`–`P5`)
- **`P1`**: Standardized on a `src/epibench/` layout under a single lowercase package
  name (`epibench`), removed the legacy `sys.modules` alias hack, replaced all
  legacy CamelCase imports, and exposed the `epibench` CLI entry point.
- **`P2`**: Added standalone `epibench.embedding.EmbeddingService` backed by local
  FastEmbed (`BAAI/bge-small-en-v1.5`), eliminating external private-module imports
  from `vector_rag` and `hybrid_rag`.
- **`P3`**: Declared `httpx` and `jsonschema` in `pyproject.toml` and
  `requirements.txt`, removed unused `rank-bm25`, and moved optional third-party
  baselines (`mem0ai`, `qdrant-client`, `google-genai`) into `[project.optional-dependencies].baselines`.
- **`P4`**: Updated `epibench.runners.run_efficiency_benchmark` to use the canonical
  `epibench` package imports and `session_date` session keys.
- **`P5`**: Removed hard-coded personal spending limits from `metering.py`, made
  `--budget-usd` an explicit required CLI cap, and replaced the hard-coded commit
  hash with runtime `get_git_commit_sha()`.

### Programmatic Scorers & Reader Protocol (`S1`–`S5`)
- **`S1`**: Replaced substring/short-token-dropping matching with NFKC, currency,
  and thousands-separator normalization that preserves short tokens (`14` vs `16`)
  and requires whole-token alias matching inside a mandatory `FINAL ANSWER:` line.
- **`S2`**: Added strict `score_yes_no` requiring exact `YES` or `NO` on the
  `FINAL ANSWER:` line so negations such as `"No longer stands"` cannot pass `"YES"`.
- **`S3`**: Rewrote `score_integer_match` to accept exactly one integer (digit or
  number word) on the `FINAL ANSWER:` line, rejecting multi-number explanations and
  list numbering.
- **`S4`**: Removed all hard-coded pool keywords from `score_set_overlap`; Track D
  now scores semicolon-delimited answers strictly against per-scenario `gold_aliases`
  and `unaffected_aliases` with `F1 == 1.0` required and zero phantom penalties on
  unaffected explanations.
- **`S5`**: Replaced lenient contamination checks with whole-phrase boundary
  matching (`check_forbidden_phrases`, `check_stale_answer`).

### Scenario Generator & Ground Truth (`G1`–`G5`)
- **`G1`**: Track F slip days are now computed directly from milestone dates via
  `compute_forecast_gold()` (fixing the 2024 leap-year Titan slip from 52 to 53 days).
- **`G2`**: Fixed Track R pool item 6 (`r_06_project_deadline`) to have three
  distinct dates (`December 15`, `January 15`, `February 15`).
- **`G3`**: Added centralized `allocate_filler_dates()` guaranteeing that no filler
  or evidence session ever postdates `question_date` at any `filler_count`.
- **`G4`**: Replaced colliding arithmetic seed offsets with deterministic SHA-256
  `derive_seed()`.
- **`G5`**: Expanded all 9 curated tracks to 12–16 templates each (114 templates
  total) with explicit `gold_aliases` and `forbidden_phrases`, enforced startup
  `validate_pools()`, added `epibench manifest` with committed
  `manifests/dev_manifest.json`, and guarded `split="held_out"` to require external
  `custom_pools`. Updated documentation to state honestly that pools are curated,
  deterministic, templated world models rather than procedurally generated names.

### Adapters & Runner (`A1`–`A6`)
- **`A1`**: Enforced strict `as_of` cutoff filtering across every built-in adapter
  before ranking or windowing (`parse_as_of_cutoff`, `filter_sessions_as_of`),
  verified by runner-level `source_session_ids` compliance auditing.
- **`A2`**: Replaced oldest-first truncation with `RecentContextAdapter` (packing
  the newest contiguous sessions in chronological order within the unified token
  budget; `full_context` kept as an alias) plus `FullContextUnboundedAdapter` as an
  unbounded reference ceiling.
- **`A3`**: Implemented chronological deletion replay in `EpiBenchRunner`: Track E
  ingests sessions up to the deletion boundary, invokes `adapter.delete(tenant_id,
  target=...)`, records `delete_unsupported` if unsupported, and evaluates Track E
  jointly (`control_retained AND leak_free`).
- **`A4`**: Passed the true `scenario.question_date` to `ReaderLLM` rather than
  substituting `as_of`, and enabled Track R `as_of` retrieval filtering.
- **`A5`**: Added `NullMemoryAdapter` (lower bound) and `OracleAdapter` (upper
  bound) reference baselines plus deterministic offline `FakeEmbedder`, `FakeReader`,
  and `FakeJudge`.
- **`A6`**: Removed global `openai` monkeypatching from `Mem0Adapter`, attached
  `session_date`/`session_id` metadata for `as_of` filtering and targeted deletion,
  and flagged `internal_llm_cost_unmetered=True` honestly.

### Metrics, Judging & Statistics (`M1`–`M5`)
- **`M1`**: Missing dimensions now report `null` instead of `100%`, and composite
  scores renormalize weights over evaluated non-experimental dimensions only.
- **`M2`**: Added a seventh independent dimension (`constraint_compliance`) for
  Track X and separated Track E's non-deleted control check into `retention_control`.
- **`M3`**: Added async-safe per-system spend attribution (`CURRENT_SYSTEM_VAR`)
  so `cost_per_successful_answer_usd` reflects real metered reader/judge/adapter spend.
- **`M4`**: Added `compute_pairwise_comparisons()` (exact McNemar test, paired
  scenario bootstrap `B=10,000`, Wilson 95% CIs, Holm-Bonferroni correction) and
  JSON Schema validation (`src/epibench/schemas/results_schema.json`).
- **`M5`**: Defaulted the rubric judge (`gemini-3.5-flash`) to a distinct model
  from the reader (`gemini-3.5-flash-lite`), added strict JSON parsing with retry
  and `judge_error` accounting, enabled 10% secondary judge auditing with Cohen's
  kappa reporting, and added transient HTTP 429/5xx retry backoff.

### Validity, Environment, Runners & Documentation (`V1`–`V4`)
- **`V1`**: Added mandatory `epibench validate` gate (`oracle >= 95%`,
  `null_memory <= 15%` per track) and passed all 9 tracks live before the integrity
  freeze (`results/validity/validity_report.json`).
- **`V2`**: Rewrote `epibench.environment` (`WorldSimulator`, `EpiBenchEnvironment`)
  so every probed fact is emitted explicitly in dialogue turns and passes a 20-seed
  answerability suite with `replay_into_adapter()`.
- **`V3`**: Upgraded `run_depth_benchmark.py` (default `scenarios_per_track=30`,
  Wilson 95% CIs, reference bounds) and `run_efficiency_benchmark.py` (planted
  factual needle recall alongside latency and token metrics).
- **`V4`**: Archived all pre-2.0.0 results to `results_archive_invalid/`, committed
  schema-validated `2.0.0` built-in benchmark and efficiency artifacts, rewrote
  `README.md` with zero placeholder tables or unbacked claims, and added
  `scripts/check_readme_claims.py` and GitHub Actions CI.

---

## Pre-2.0.0 (Archived / Invalid)
- Initial prototype releases prior to `SCORER_VERSION 2.0.0`. Archived results live
  in `results_archive_invalid/` for provenance only.
