import logging
from src.llm_orchestrator import LLMOrchestrator
from benchmarks.dummy_bench.client import DummyBenchmark

# Setup simple console logging
logging.basicConfig(level=logging.INFO)


def main():
    # 1. Initialize Orchestrator (Reads config.toml)
    orchestrator = LLMOrchestrator()

    # 2. Run for each enabled model in the config
    for model_key in orchestrator.cfg.enabled_models:
        for memory_key in orchestrator.cfg.enabled_memory_methods:
            # Hotswap context
            orchestrator.set_active_context(model_key, memory_key)

            # Initialize and Run Benchmark
            bench = DummyBenchmark(orchestrator)
            bench.run()

            print("\n" + "=" * 50 + "\n")


if __name__ == "__main__":
    main()
