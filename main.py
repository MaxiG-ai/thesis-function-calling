from src.llm_orchestrator import LLMOrchestrator

def main():
    print("Hello from thesis-function-calling!")

# Conceptual example of your experiment runner
orchestrator = LLMOrchestrator()

if __name__ == "__main__":
    main()
    
    for model in orchestrator.cfg.enabled_models:
        for memory in orchestrator.cfg.enabled_memory_methods:
            # 1. Switch the Proxy State
            orchestrator.set_active_context(model, memory)

            # 2. Run the Benchmark (which hits your Proxy API)
            # run_benchmark_process(...)
