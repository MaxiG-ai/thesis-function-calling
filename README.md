# thesis-function-calling

The purpose of this repository is to evaluate the impact of different **Memory architectures** on Function Calling performance.

This project operates as a **Middleware Proxy** between standard benchmarks and LLM providers. It intercepts the conversation history sent by the benchmark client, transforms it using specific memory strategies (e.g., summarization, RAG selection), and forwards the optimized context to the model.

## Components

The repository consists of three core components: Benchmarks (Clients), Memory Middleware (Proxy Logic), and Models (Providers).

### Memory Techniques (Middleware)

The core thesis work involves implementing `Context Transformation` functions. These functions take the full conversation history `List[Message]` provided by the benchmark and return an optimized `List[Message]`.

#### Baselines

* **Full Context:** No modification (Control group).
* **Truncation:** Mechanically cutting context after $N$ tokens (Baseline for failure).

#### Implemented Techniques

* **Tool Output Summarization (Semantic Compression):**
  * Interprets massive JSON `tool_outputs`.
  * Replaces raw data with concise, semantic natural language summaries before forwarding to the LLM.
  * *Hypothesis:* Reduces token usage and distraction without losing semantic resolution.
* **RAG-based Turn Selection:**
  * Uses the current `user_query` to embed and retrieve only the most relevant past `(ToolCall, ToolOutput)` pairs from the history.
  * *Constraint:* Maintains atomic consistency (never separates a Call from its Result) to preserve API validity.
* **Progressive Summarization:**
  * Periodically condenses older conversation turns into a running "state summary" injected into the System Prompt.

### Models & Providers

The Middleware acts as a standard OpenAI-compatible API. The Benchmarks point to this proxy, unaware that their requests are being intercepted and optimized.

#### Local Proxy Architecture

All LLM requests flow through a **local proxy at `localhost:3030`** that serves as the routing layer:

1. **LiteLLM Integration:** The `LLMOrchestrator` uses LiteLLM to send requests to `localhost:3030/v1` (configured in `model_config.toml`)
2. **Proxy Router:** The proxy at port 3030 routes requests to actual LLM providers (OpenAI, Ollama, SAP AI Core, etc.)
3. **Provider Flexibility:** This abstraction allows switching between providers without changing benchmark code

**Configuration Flow:**
* `model_config.toml` - Defines models with `api_base = "http://localhost:3030/v1"`
* `config.toml` - Selects which models to test via `enabled_models` list

To add/edit models the following files need to be changed:

* `model_config.toml` - Define model configurations and provider endpoints
* `config.toml` - Select models in enabled model list

### Benchmarks (Clients)

The benchmarks manage the execution loop and state. They send the accumulating history to the proxy at every turn.

#### ComplexFuncBench (z.ai)

* **Focus:** Complex and repeated function calls
* **Source:** [GitHub](https://github.com/zai-org/ComplexFuncBench)
* **Integration Status:** Fully integrated with custom middleware injection

**Historic Changes to `benchmarks/complex_func_bench`:**
The ComplexFuncBench benchmark was integrated into this repository with the following modifications:
* **Added `SAPGPTRunner`:** A custom runner that injects the `LLMOrchestrator` into the benchmark execution flow
* **Orchestrator Integration:** Modified `FunctionCallSAPGPT` model class to accept an optional `orchestrator` parameter, enabling memory processing while preserving original benchmark code
* **Preserved Original Structure:** All original benchmark files remain intact; thesis-specific modifications are isolated in custom runner classes
* **Legacy Support:** Original runners (e.g., `gpt_runner.py`) are preserved for comparison

#### NestFul (IBM)

* **Focus:** Nested function calling and multi-step reasoning.
* **Source:** [GitHub](https://github.com/IBM/NESTFUL)

#### MCP-Bench (Accenture)

* **Focus:** Complex tasks using Model Context Protocol (MCP) servers and cross-tool coordination.
* **Source:** [GitHub](https://github.com/Accenture/mcp-bench)

## Implementation Details

### Core Components

#### LLMOrchestrator

**Purpose:** Central manager for all LLM interactions with integrated memory processing.

**Key Responsibilities:**
* **Configuration Management:** Loads and manages experiment config (`config.toml`) and model registry (`model_config.toml`)
* **Memory Integration:** Instantiates and delegates to `MemoryProcessor` for context optimization
* **LiteLLM Coordination:** Handles all LLM API calls through LiteLLM with proper error handling and retries
* **Context Switching:** Provides `set_active_context(model, memory)` to dynamically switch configurations during benchmark runs
* **Observability:** Integrates with Weave for comprehensive tracing of all LLM calls and memory operations

**Usage Example:**
```python
orchestrator = LLMOrchestrator()
orchestrator.set_active_context("gpt-4-1", "memory_bank")
response = orchestrator.generate_with_memory_applied(messages, tools=tools)
```

#### MemoryProcessor

**Purpose:** Applies memory strategies to optimize conversation context before LLM calls.

**Key Responsibilities:**
* **Strategy Application:** Executes the configured memory strategy (truncation, progressive summarization, memory bank)
* **Context Segmentation:** Splits messages into system prompts, archived context, and working memory
* **Stateful Summarization:** Maintains running summaries for progressive summarization strategy
* **Loop Detection:** Identifies and prevents infinite conversation loops
* **Metrics Tracking:** Reports compression ratios and token usage via Weave

**Supported Strategies:**
* **Truncation:** Naive baseline that keeps only system prompt + recent N messages
* **Progressive Summarization:** Periodically condenses older turns into a running summary
* **Memory Bank:** RAG-based retrieval of relevant past interactions using embeddings

### Config

A central `config.toml` defines the active Memory Strategy and the target Model Provider.

#### Running Specific Test Cases

You can run the benchmark on specific test cases by setting the `selected_test_cases` option in `config.toml`:

```toml
# Run only specific test cases by their IDs
selected_test_cases = ["Car-Rental-0", "Car-Rental-1", "Travel-5"]
```

When `selected_test_cases` is set:
- Only the specified test cases will be executed
- The `benchmark_sample_size` option is ignored
- This provides minimal overhead as only the selected cases are loaded and processed

To run all test cases or use random sampling, comment out or remove the `selected_test_cases` line.

### Middleware Logic

The implementation handles **Context Transformation** through direct orchestrator injection:

1. **Intercept:** Benchmark runner calls `orchestrator.generate_with_memory_applied(messages, tools)`
2. **Identify:** `MemoryProcessor` segments messages into system prompt, archived context, and working memory
3. **Transform:** Apply the active Memory Technique (e.g., compress via summarization or RAG selection)
4. **Forward:** Send the optimized payload to the LLM provider via LiteLLM through `localhost:3030`
5. **Return:** Return the LLM response to the benchmark runner

> **Note:** All transformations strictly preserve `tool_call_id` integrity to ensure the Benchmark's execution loop remains valid.

## Getting Started

### Prerequisites

1. **Start the Local Proxy (Port 3030):** 
   Ensure a local proxy is running at `localhost:3030` that routes LiteLLM requests to LLM providers.

2. **(Optional) Run SAP AI Core Proxy:**
   ```bash
   cd dev/sap
   sap
   ```

### Running Benchmark Evaluations

#### ComplexFuncBench Evaluation

Execute the full benchmark with memory strategies:

```bash
python cfb_run_eval.py
```

**What This Does:**
1. Loads configuration from `config.toml` (models, memory strategies, sample size)
2. Initializes the `LLMOrchestrator` with model registry from `model_config.toml`
3. Iterates through the Cartesian product of `enabled_models` × `enabled_memory_methods`
4. For each configuration:
   - Creates a `SAPGPTRunner` with orchestrator injection
   - Processes benchmark cases with memory transformation
   - Evaluates function calling accuracy and response quality
   - Saves results to `results/{experiment_name}/{timestamp}/{memory}/{model}/`
5. Aggregates metrics and logs to Weave for tracking

**Configuration:**

Edit `config.toml` to control the experiment:
```toml
experiment_name = "initial_tests"
benchmark_sample_size = 10  # Number of test cases (or full dataset)

enabled_models = ["gpt-4-1-mini"]
enabled_memory_methods = ["truncation", "progressive_summarization"]

[memory_strategies.progressive_summarization]
type = "progressive_summarization"
auto_compact_threshold = 10000
summarizer_model = "gpt-4-1-mini"
```

**Results:**
* Raw results: `results/{experiment_name}/{timestamp}/{memory}/{model}/result.json`
* Aggregated metrics: `results/{experiment_name}/{timestamp}/{memory}/{model}/metrics.json`
* Weave traces: View in Weave UI for detailed call tracing
