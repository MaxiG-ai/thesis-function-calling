# thesis-function-calling

Evaluating the impact of different **Memory Architectures** on LLM Function Calling performance. Based on related package [MemOrch](https://github.com/MaxiG-ai/memory-orchestrator), this repo provides the integration and tests to run different Benchmarks and thus test the memory strategies implemented.

## Benchmark: ComplexFuncBench

Currently integrated: **ComplexFuncBench** (CFB)

- Focus: Complex and repeated function calls
- Data: `benchmarks/complex_func_bench/data/ComplexFuncBench.jsonl`
- Source: [GitHub](https://github.com/zai-org/ComplexFuncBench)

## Project Structure

```bash
├── benchmarks/
│   └── complex_func_bench/      # CFB benchmark integration
├── docs/                        # Documentation
├── tests/                       # Test suite
├── config.toml                  # Experiment configuration
├── cfb_run_eval.py              # Main evaluation entry point
└── run_baseline.py                       # Run the evaluation multiple times to mitigate undeterministic results.
```
