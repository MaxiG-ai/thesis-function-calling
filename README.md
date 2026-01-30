# thesis-function-calling

Evaluating the impact of different **Memory Architectures** on LLM Function Calling performance.

This project intercepts conversation history from benchmarks, applies memory compression strategies, and forwards optimized context to LLM providers via SAP AI Core Proxy.

## Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│    Benchmark    │────►│   Memory Processor   │────►│  SAP AI Proxy   │
│  (CFB Dataset)  │     │  (Context Transform) │     │   (LiteLLM)     │
└─────────────────┘     └──────────────────────┘     └─────────────────┘
```

**Components:**
- **LLMOrchestrator** (`src/llm_orchestrator.py`) - Central LLM management with memory integration
- **MemoryProcessor** (`src/memory_processing.py`) - Dispatches to active memory strategy
- **Strategies** (`src/strategies/`) - Individual compression implementations

## Memory Strategies

### Baselines

| Strategy | Type Key | Description |
|----------|----------|-------------|
| **Full Context** | `no_strategy` | Passthrough, no modification (control group) |
| **Truncation** | `truncation` | Keeps user query + last tool interaction, discards middle history |

### Implemented Techniques

| Strategy | Type Key | Description |
|----------|----------|-------------|
| **Progressive Summarization** | `progressive_summarization` | LLM periodically condenses history into a running summary |
| **Memory Bank** | `memory_bank` | Dual-store (vector + key-value) retrieval over past interactions. See [docs/memorybank.md](docs/memorybank.md) |
| **ACE** | `ace` | Agentic Context Engineering with Generator → Reflector → Curator learning cycle |

## Configuration

### config.toml

```toml
# Experiment settings
experiment_name = "my_experiment"
benchmark_sample_size = 3                    # Number of test cases (null for full)
compact_threshold = 6000                     # Tokens before compression activates

# Select models and strategies to test (Cartesian product)
enabled_models = ["gpt-4-1"]
enabled_memory_methods = ["memory_bank"]

# Optional: run specific test cases only
selected_test_cases = ["Car-Rental-131", "Cross-131"]
```

### model_config.toml

Model registry with LiteLLM names. All models route through SAP AI Core Proxy at `localhost`.

## Running

```bash
# Run evaluation
uv run python cfb_run_eval.py

# Run tests
uv run pytest
```

**Output:** Results saved to `results/<experiment_name>/<timestamp>/<memory>/<model>/`

## Benchmark

Currently integrated: **ComplexFuncBench** (CFB)
- Focus: Complex and repeated function calls
- Data: `benchmarks/complex_func_bench/data/ComplexFuncBench.jsonl`
- Source: [GitHub](https://github.com/zai-org/ComplexFuncBench)

## Project Structure

```
├── cfb_run_eval.py              # Main evaluation entry point
├── config.toml                  # Experiment configuration
├── model_config.toml            # Model registry
├── src/
│   ├── llm_orchestrator.py      # Central LLM + memory management
│   ├── memory_processing.py     # Strategy dispatch
│   └── strategies/              # Memory strategy implementations
│       ├── truncation/
│       ├── progressive_summarization/
│       ├── memory_bank/
│       └── ace/
├── benchmarks/
│   └── complex_func_bench/      # CFB benchmark integration
├── tests/                       # Test suite
└── docs/                        # Detailed documentation
```
