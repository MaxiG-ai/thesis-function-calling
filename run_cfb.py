import json
import os
import sys
import copy
import random
import logging
from datetime import datetime
from src.utils.logger import get_logger

# Ensure we can find the src module and benchmark modules
sys.path.append(os.getcwd())
cfb_path = os.path.join(os.getcwd(), "benchmarks", "complex_func_bench")
sys.path.append(cfb_path)

# Import custom logger

logger = get_logger("CFB_Runner")

# Import your Orchestrator
try:
    from src.llm_orchestrator import LLMOrchestrator
except ImportError:
    logger.error("❌ Could not import LLMOrchestrator. Run this script from the root of the repository.")
    sys.exit(1)

# Import CFB Adapter and Utils
try:
    from benchmarks.complex_func_bench.orchestrator_runner import OrchestratorRunner
    # We rename this to avoid confusion with the standard logging module
    from benchmarks.complex_func_bench.utils.logger import Logger as FileLogger 
except ImportError as e:
    logger.error(f"❌ Failed to import CFB modules: {e}")
    sys.exit(1)

def load_jsonl(path):
    if not os.path.exists(path):
        logger.error(f"Dataset not found at {path}")
        return []
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def main():
    logger.info("Initializing ComplexFuncBench integration...")

    # 2. Initialize Orchestrator
    try:
        orchestrator = LLMOrchestrator()
    except Exception as e:
        logger.critical(f"Failed to initialize Orchestrator: {e}")
        return

    # 3. Load Dataset
    data_path = os.path.join(cfb_path, "data", "ComplexFuncBench.jsonl")
    dataset = load_jsonl(data_path)
    
    if not dataset:
        logger.warning("No data loaded. Exiting.")
        return

    # Optional: Test with a subset
    sample_size = orchestrator.cfg.benchmark_sample_size
    if sample_size:
        dataset = random.sample(dataset, sample_size)
    logger.info(f"Loaded {len(dataset)} evaluation samples.")

    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    # 4. Iterate Configurations
    for model in orchestrator.cfg.enabled_models:
        for memory in orchestrator.cfg.enabled_memory_methods:
            logger.info(f"🚀 Starting Run | Model: {model} | Memory: {memory}")

            try:
                orchestrator.set_active_context(model, memory)
            except Exception as e:
                logger.error(f"Failed to switch context: {e}")
                continue

            # Setup a file logger specifically for the CFB internal runner
            # (The runner expects a logger instance that handles file writing)
            log_dir = os.path.join("results", orchestrator.cfg.experiment_name, run_timestamp, memory, model)
            os.makedirs(log_dir, exist_ok=True)
            
            # This 'runner_logger' is passed to the benchmark class
            runner_logger = FileLogger(f"runner_{run_timestamp}", os.path.join(log_dir, f"cfb_{model}_{memory}_{run_timestamp}.log"), level = logging.DEBUG)
            
            # Create Dummy Args for the Runner (it expects an object with attributes)
            class RunnerArgs:
                def __init__(self):
                    self.log_dir = log_dir
            
            runner = OrchestratorRunner(RunnerArgs(), runner_logger,orchestrator)
            
            results = []
            success_count = 0
            
            # 5. Execution Loop
            for i, case in enumerate(dataset):
                case_id = case.get('id', i)
                
                # Use debug for per-turn info to avoid cluttering console
                logger.info("###" * 20) 
                logger.info(f"Processing Case {case_id}...") 
                
                try:
                    # Run the case (deepcopy to prevent mutation of original dataset)
                    convs, msg, turns, correct_calls = runner.run(copy.deepcopy(case))
                    
                    status = "Success" if msg == "Success." else "Failed"
                    if status == "Success":
                        success_count += 1
                    
                    results.append({
                        "id": case_id,
                        "status": status,
                        # added to showcase memory impact
                        "memory_method": memory,
                        "message": msg,
                        "turns": turns,
                        "correct_calls": correct_calls
                    })
                    
                except Exception as e:
                    logger.error(f"❌ Exception on Case {case_id}: {e}")

            # 6. Report Results
            pass_rate = (success_count / len(dataset)) * 100
            logger.info(f"🏁 Completed {model}/{memory}")
            logger.info(f"📊 Result: {success_count}/{len(dataset)} ({pass_rate:.1f}%) passed")

            # Save Results
            result_file = os.path.join("results", orchestrator.cfg.experiment_name, run_timestamp, memory, model, f"cfb_{model}_{memory}_{run_timestamp}.json")
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"💾 Results saved to {result_file}")

if __name__ == "__main__":
    main()
