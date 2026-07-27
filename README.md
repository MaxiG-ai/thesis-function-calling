# thesis-function-calling

Evaluating the impact of different **Memory Architectures** on LLM Function Calling performance. Based on related package [MemOrch](https://github.com/MaxiG-ai/memory-orchestrator), this repo provides the integration and tests to run different Benchmarks and thus test the memory strategies implemented.

## Prerequesits

This benchmark is designed to be evaluated on a Linux GPU-Cluster. In the case of the data for submission for AAAI, this was a 8*H200 GPU cluster on openSUSE, which took ~ 1 week for all configurations. 

## Benchmark: ComplexFuncBench

- Focus: Complex and repeated function calls
- Data: `benchmarks/complex_func_bench/data/ComplexFuncBench.jsonl`
- Source: [GitHub](https://github.com/zai-org/ComplexFuncBench)

## Project Structure

```bash
├── benchmarks/
│   └── complex_func_bench/      # CFB benchmark integration
├── docs/                        # Documentation
├── tests/                       # Test suite
├── tools/                       # Development tools
│   └── trace_viewer.py          # Local trace inspection UI
├── config.toml                  # Experiment configuration
├── cfb_run_eval.py              # Main evaluation entry point
└── run_baseline.py              # Run evaluation multiple times for statistical validity
```

## Tools

## Testing

### Model Availability Report Test

Run the integration test below to check all models from `configs/model_config.toml` and print a pass/fail report in the terminal:

```bash
uv run pytest tests/integration/test_model_registry_report.py -q
```

This test performs real model calls and fails if one or more configured models are not callable.

### Trace Viewer

A web interface for inspecting experiment traces. Replaces cloud-based solutions for offline trace analysis.

```bash
uv run python tools/trace_viewer.py
```

Then open http://localhost:8080 in your browser.

**Features:**
- Browse experiments and timestamps from results directory
- View conversation traces as a chat interface
- Side-by-side model/strategy comparison
- Search cases by ID
- JSON inspector for debugging individual messages

**Environment variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `TRACE_VIEWER_PORT` | `8080` | Port to run the server on |
| `TRACE_VIEWER_RESULTS_ROOT` | `results/cfb` | Path to results directory |
