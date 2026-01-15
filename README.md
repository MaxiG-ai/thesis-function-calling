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

LLM logic is abstracted using [Apantli](https://github.com/pborenstein/apantli).
The Middleware acts as a standard OpenAI-compatible API. The Benchmarks point to this proxy, unaware that their requests are being intercepted and optimized.

To add/edit models the following files need to be changed:

* `/dev/apantli/config.yaml` or wherever the apantli server is installed
* `model_config.toml` important to send to AICore Proxy
* `config.toml` use model in enabled model list

### Benchmarks (Clients)

The benchmarks manage the execution loop and state. They send the accumulating history to the proxy at every turn.

#### ComplexFuncBench (z.ai)

* **Focus:** Complex and repeated function calls
* **Source:** [GitHub](https://github.com/zai-org/ComplexFuncBench)

#### NestFul (IBM)

* **Focus:** Nested function calling and multi-step reasoning.
* **Source:** [GitHub](https://github.com/IBM/NESTFUL)

#### MCP-Bench (Accenture)

* **Focus:** Complex tasks using Model Context Protocol (MCP) servers and cross-tool coordination.
* **Source:** [GitHub](https://github.com/Accenture/mcp-bench)

## Implementation Details

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

The proxy implementation handles the **Stateless Transformation**:

1. **Intercept:** Receive `POST /chat/completions` with `messages=[]`.
2. **Identify:** Isolate the `System`, `User`, and `History` (Assistant/Tool pairs).
3. **Transform:** Apply the active Memory Technique (e.g., compress `ToolOutput` messages).
4. **Forward:** Send the optimized payload to the real LLM via LiteLLM.
5. **Return:** Stream the response back to the Benchmark.

> **Note:** All transformations strictly preserve `tool_call_id` integrity to ensure the Benchmark's execution loop remains valid.

## Getting Started

### Environment Variables for Langfuse Tracing

This project uses [Langfuse](https://langfuse.com/) for observability and tracing of LLM calls. Configure the following environment variables:

```bash
export LANGFUSE_PUBLIC_KEY="your-public-key"
export LANGFUSE_SECRET_KEY="your-secret-key"
export LANGFUSE_HOST="https://cloud.langfuse.com"  # Optional, defaults to cloud.langfuse.com
```

### Running the Evaluation

1. Run local SAP AI Core Proxy `cd dev/sap` and run `sap`
2. Run cost orchestration `cd dev/apantli`, activate env and run `apantli --port 4000`.
