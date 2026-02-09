# Add A New Benchmark

This repository evaluates memory architectures by running an benchmark end-to-end, but routing *all* model calls through `src/llm_orchestrator.py` so that memory strategies can intercept and transform the conversation context.

This doc explains the minimal integration surface area needed to add a new benchmark under `benchmarks/` while:

- keeping benchmark-specific logic isolated
- reusing the existing experiment matrix (`enabled_models` x `enabled_memory_methods`)
- producing results in the same directory layout as existing runs

The existing reference integration is ComplexFuncBench:

- Benchmark package: `benchmarks/complex_func_bench/`
- Entrypoint: `cfb_run_eval.py`

## Integration Contract

Your benchmark runner must do two things:

1) Build the benchmark's conversation state (messages + tool definitions).
2) Call the LLM through `LLMOrchestrator.generate_with_memory_applied(...)`.

The orchestrator is responsible for:

- applying memory strategy transforms via `src/memory_processing.py`
- selecting the correct model from `model_config.toml`
- sending requests via LiteLLM

In practice, you typically implement a small benchmark-specific model adapter, similar to:

- `benchmarks/complex_func_bench/models/sap_gpt.py`

Key constraint:

- Do not call `litellm.completion(...)` directly inside the benchmark.
- Route benchmark inference calls through the orchestrator, so memory strategies are evaluated.

## Entrypoint Pattern

Use `cfb_run_eval.py` as the template. Keep the entrypoint benchmark-specific (one file per benchmark). This avoids coupling between unrelated benchmarks and keeps dependencies isolated.

## Config Changes

This project has two config files:

- `config.toml` (experiment + memory strategies)
- `model_config.toml` (model registry)

To add a new benchmark, prefer adding a dedicated config block under `config.toml`, while reusing global knobs:

- reuse:
  - `experiment_name`
  - `enabled_models`
  - `enabled_memory_methods`
  - `benchmark_sample_size`
  - `selected_test_cases`
  - `compact_threshold`
- benchmark-specific:
  - input file path(s)
  - benchmark category selection
  - evaluator settings

Example (illustrative):

```toml
benchmark_sample_size = 50
enabled_models = ["gpt-4-1"]
enabled_memory_methods = ["no_strategy"]

[benchmarks.my_benchmark]
input_file = "benchmarks/my_benchmark/data/dataset.jsonl"
use_official_scoring = true
```

If you add new config fields, update the Pydantic models in:

- `src/utils/config.py`

## Official Scoring

If the benchmark provides an official scorer use that as it is.

- generate outputs in the exact format the scorer expects
- call the scorer after inference completes
- parse the scorer output into:
  - a per-case result object (store the raw scorer output)
  - an aggregate metrics object (for `metrics_*.json`)

Avoid reimplementing benchmark metrics unless necessary.

## Tests (TDD)

Add tests before implementing non-trivial benchmark logic.

Suggested tests:

- dataset loader parses and produces stable `id`s
- selected id filtering behaves correctly (reusable logic)
- scorer output parsing is stable across versions

All tests live under `tests/` and should be runnable via:

- `uv run pytest`

Each test must include a detailed docstring explaining intent and expected behavior.

## Logging And Results

Persist two artifacts per (model, memory, benchmark) run:

- raw per-case outputs (benchmark-native + orchestrator metadata)
- aggregate metrics json

Recommended file naming convention (match existing style):

- `results/<benchmark>/<experiment>/<timestamp>/<memory>/<model>/<benchmark>_<model>_<memory>_<timestamp>.json`
- `results/<benchmark>/<experiment>/<timestamp>/<memory>/<model>/metrics_<model>_<memory>_<timestamp>.json`

## Common Pitfalls

- Tool schema mismatch: benchmarks may use different tool/function formats.
- Role mismatch: some benchmarks use custom roles (e.g., `observation`).
- Session resets: call `orchestrator.reset_session()` per benchmark case to avoid cross-case leakage in memory strategies.
- Scorer assumptions: official scorers often assume a specific output structure and deterministic ordering.
