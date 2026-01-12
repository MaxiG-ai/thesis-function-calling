# Refactor Recommendations (LLM/AI Coding Principles)

Date: 2026-01-12

This document summarizes **high-level** refactor suggestions to better align this repo with practical LLM/AI coding principles: clarity of interfaces, deterministic behavior, strong observability, safe config handling, and reproducible experiments.

It intentionally focuses on **changes that materially improve reliability, maintainability, or evaluation validity**. If something is already fine, it’s called out as such.

---

## What “LLM/AI coding principles” means here

For an LLM + tool-calling + evaluation harness, the principles that usually matter most:

- **Single, explicit “LLM boundary”**: one module owns request shaping, retries, provider quirks, and telemetry.
- **Schema-first messages**: strong invariants for “messages/tools/tool results” to avoid silent API invalid payloads.
- **Deterministic experiments**: configs are complete, runs are reproducible, outputs are structured and versionable.
- **Observability by default**: structured logging + tracing; no `print()` in library code.
- **Separation of concerns**: evaluation vs orchestration vs memory strategies vs UI tools.
- **Minimize duplication & legacy paths**: one canonical implementation; legacy kept only if it’s actively needed and clearly isolated.

---

## What looks already good (don’t change just to change)

- **Clear conceptual separation**: `src/llm_orchestrator.py` + `src/memory_processing.py` is a good backbone.
- **Strategy modules are isolated**: `src/strategies/*` is the right direction.
- **Tests exist and are meaningful**: especially the ACE test suite.
- **Config-driven experiment matrix**: `config.toml` + `model_config.toml` is a solid pattern.

---

## High priority

### 1) Fix trace splitting semantics (docstring + clarity)

Why it matters: memory strategies are only as good as the “conversation slice” they operate on.

-- `src/utils/split_trace.py` has `process_and_split_trace_user()` whose behavior is fine for ComplexFuncBench,
  but the docstring was misleading:
  - It returns `messages_after_last_user = messages[last_user_index + 1:]`.
  - ComplexFuncBench typically has a single user message at the start; the remainder is system/assistant/tool.
- `src/strategies/progressive_summarization/prog_sum.py` uses it as:
  - `user_query, conversation_history = process_and_split_trace_user(messages)`
  - then summarizes `conversation_history`.

Recommendation:
- Update docstrings / naming so the contract is unambiguous.

### 2) Collapse/remove the legacy SAP GPT path

Why it matters: duplicate model adapters create drift, inconsistent behavior, and non-reproducible evals.

- `benchmarks/complex_func_bench/models/legacy_sap_gpt.py`:
  - uses `sys.path` hacks,
  - hardcodes an OpenAI client pointing to `http://localhost:4000/v1`,
  - contains placeholder API key behavior,
  - duplicates behavior already covered by the orchestrator-driven path.
- `benchmarks/complex_func_bench/runner/legacy_sap_gpt_runner.py` is the only place importing it.

Recommendation:
- If it’s not actively used: delete both legacy files.

### 3) Eliminate `print()` from non-demo code paths

Why it matters: `print()` breaks structured logging and makes experiment runs harder to debug and compare.

Observed examples:
- `src/utils/config.py` prints ValidationError output.
- multiple benchmark model adapters print exceptions.

Recommendation:
- For library/middleware code (`src/**`): replace `print()` with `src/utils/logger.py`.
- For benchmark code: keep prints only in explicit demo tools (e.g., `benchmarks/dummy_bench/`), otherwise use the same logger.

### 4) Make the Orchestrator the single “LLM boundary” (and enforce message/tool schema)

Why it matters: tool-calling payload validity is fragile; schema drift causes silent evaluation artifacts.

Observed:
- The orchestrator already centralizes LiteLLM + memory processing.
- Some benchmark paths still do their own “LLM calling” concerns (retries, printing errors, mixed message formats like `{"role": "observation"}`).

Recommendation:
- Ensure every model path routes through the orchestrator (or is explicitly flagged “legacy”).
- Introduce a small “message normalization” layer (even if it’s just a function) that:
  - validates roles,
  - ensures tool call / tool output pairing,
  - enforces `tool_call_id` consistency,
  - rejects unsupported roles early.

### 5) Reduce risk in config handling (keys, endpoints, reproducibility)

Why it matters: experiments are hard to reproduce if configs are partly “in code” and partly “in env”.

Observed:
- `model_config.toml` contains `api_key = "THINKTANK"` and fixed URLs.

Recommendation:
- Prefer `api_key = "${ENV_VAR_NAME}"` style indirection (or leave blank and require env vars), and document required env vars.
- Keep `example_config.toml` as the reference with placeholders.

---

## Medium priority

### 1) Token counting should be model-aware and reflect tool payloads

Why it matters: memory strategies trigger on token count (`compact_threshold`) and may trigger incorrectly.

Observed:
- `src/utils/token_count.py` always uses `tiktoken.encoding_for_model("gpt-4.1")`.
- It counts only string `content` fields; tool calls / arguments may be ignored.

Recommendation:
- Make token counting accept a model name (from active model config).
- Ensure counting includes:
  - tool call `arguments`,
  - tool output payloads,
  - system prefixes injected by strategies.

### 2) Progressive summarization prompt loading should be path-safe and configurable

Observed:
- `src/strategies/progressive_summarization/prog_sum.py` reads a hardcoded relative path.

Recommendation:
- Use `pathlib.Path(__file__)` relative resolution or treat the prompt path as a config value.
- Consider making the summarization prompt part of the strategy config (already hinted in `config.toml`).

### 3) Remove or implement unused/placeholder state

Observed:
- `MemoryProcessor.processed_message_ids` isn’t used.
- `LLMOrchestrator.full_trace` is initialized but unused.

Recommendation:
- If you want full trace retention, implement it end-to-end (and test it).
- Otherwise delete the variables to reduce mental overhead.

### 4) Separate “apps/tools” from “library code”

Observed:
- `analyze_traces_for_memory.py` is a Streamlit UI app sitting at repo root.

Recommendation:
- Move UI/analysis utilities into a clear folder (e.g., `apps/` or `tools/`).
- Keep `src/` importable as the library.

---

## Low priority

### 1) Rename entrypoints for clarity

Suggested renames (only if you find yourself confused when running things):
- `cfb_run_eval.py` → `run_cfb_eval.py`
- `run_baseline.py` → `run_cfb_baseline.py`
- `analyze_traces_for_memory.py` → `apps/trace_viewer_streamlit.py`

### 2) Docs consolidation

Observed:
- `docs/` contains multiple “fix_*” and refactor summaries.

Recommendation:
- If these are still useful, group by purpose:
  - `docs/design/` (what the system is)
  - `docs/implementation/` (how it works)
  - `docs/notes/` (one-off fixes)
- Otherwise, delete stale docs that no longer reflect current behavior.

### 3) Keep artifacts out of the repo by default

Recommendation:
- Ensure runtime outputs are written under `results/` (already) but not committed.
- Avoid committing `__pycache__/` and local envs.

---

## Suggested target structure (high-level)

This is a “direction”, not a required change:

- `src/`
  - `llm_orchestrator.py` (single boundary)
  - `memory_processing.py`
  - `strategies/` (pure-ish strategy functions)
  - `utils/` (logging, config, token counting, trace splitting)
- `benchmarks/`
  - benchmark clients and adapters (thin; no provider logic)
- `apps/` (or `tools/`)
  - Streamlit trace viewer, offline analysis scripts
- `scripts/`
  - runnable entrypoints (`run_cfb_eval.py`, etc.)

---

## Next steps (practical)

If you want the smallest set of changes that provides the biggest payoff:

1. Fix and test the trace-splitting contract (unblocks trustworthy summarization).
2. Delete or quarantine the legacy SAP GPT adapter path.
3. Replace `print()` in `src/` with the project logger.
4. Make token counting model-aware and include tool payloads.

