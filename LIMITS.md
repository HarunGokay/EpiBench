# Known Limitations

- The pilot run is small, with 21 scenarios and 42 questions per system.
- All runs use one reader model (gemini-3.5-flash-lite at temperature 0), so scores reflect both retrieval quality and reader capability.
- Scenarios are generated from templates and follow structured patterns rather than organic conversation flows.
- Distractor sessions come from LongMemEval, which uses simulated multi-session chat dialogues rather than natural human chat logs.
- The relational graph memory adapter was built by the benchmark author.
- Five question types rely on an LLM judge using gemini-3.5-flash-lite: Track D why explanations, Track X constraint enforcement, Track F timeline realism, Track I causal calibration, and Track P provenance checks. All other questions are scored with deterministic code checkers (exact string, integer, or set overlap).
- The retrieval context cap is fixed at 1500 tokens. This favors systems that compress or select information over systems that dump full history into context.
