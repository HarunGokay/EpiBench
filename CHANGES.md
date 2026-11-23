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
