import json
import os
import sys
import copy
import random
import logging
from datetime import datetime
from benchmarks.complex_func_bench.utils.runner.response_runner import RespEvalRunner
from benchmarks.complex_func_bench.utils.utils import load_json
from collections import defaultdict 
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


def basic_metric(result_dir):
    results = load_json(result_dir)
    domain_success = defaultdict(int)
    domain_turn_count = defaultdict(lambda: [0, 0])
    domain_call_count = defaultdict(lambda: [0, 0])
    complete_score_count = defaultdict(lambda: [0, 0])
    correct_score_count = defaultdict(lambda: [0, 0])
    if results is None:
        logger.error(f"Failed to load results from {result_dir}")
        return
    for result in results:
        domain = result['id'].rsplit("-", 1)[0]
        if result['message'] == "Success.":
            domain_success[domain] += 1
        domain_turn_count[domain][0] += result['count_dict']['success_turn_num']
        domain_turn_count[domain][1] += result['count_dict']['total_turn_num']

        domain_call_count[domain][0] += result['count_dict']['correct_call_num']
        domain_call_count[domain][1] += result['count_dict']['total_call_num']

        if result["resp_eval"] is None:
            continue

        if result["resp_eval"]['complete']['score'] in {0, 1, 2}:
            complete_score_count[domain][0] += result["resp_eval"]['complete']['score']
            complete_score_count[domain][1] += 1
        
        if result["resp_eval"]['correct']['score'] in {0, 1, 2}:
            correct_score_count[domain][0] += result["resp_eval"]['correct']['score']
            correct_score_count[domain][1] += 1

    domain_success_rate = {k: v / 150 * 100 if k != "Cross" else v / 400 * 100 for k, v in domain_success.items()}
    domain_turn_acc = {k: v[0] / v[1] * 100 if v[1] != 0 else 0 for k, v in domain_turn_count.items()}
    domain_call_acc = {k: v[0] / v[1] * 100 if v[1] != 0 else 0 for k, v in domain_call_count.items()}

    overall_success = sum(domain_success.values()) / 1000 * 100
    overall_call_acc = sum([v[0] for v in domain_call_count.values()]) / sum([v[1] for v in domain_call_count.values()]) * 100

    complete_score, complete_total = 0, 0
    for k, v in complete_score_count.items():
        complete_score += v[0]
        complete_total += v[1]
    complete_score_avg = complete_score / complete_total if complete_total != 0 else 0

    correct_score, correct_total = 0, 0
    for k, v in correct_score_count.items():
        correct_score += v[0]
        correct_total += v[1]  
    correct_score_avg = correct_score / correct_total if correct_total != 0 else 0

    
    logger.info(f"🎯 Domain Success Rate: {domain_success_rate}")
    logger.info(f"🎯 Domain Turn Accuracy: {domain_turn_acc}")
    logger.info(f"🎯 Domain Call Accuracy: {domain_call_acc}")
    logger.info(f"🎯 Overall Success Rate: {overall_success}")
    logger.info(f"🎯 Overall Call Accuracy: {overall_call_acc}")
    logger.info(f"🎯 Complete Score: {complete_score_avg}")
    logger.info(f"🎯 Correct Score: {correct_score_avg}")

    # Save metrics to a summary file
    summary_path = os.path.join(os.path.dirname(result_dir), "summary_metrics.json")
    summary_metrics = {
        "domain_success_rate": domain_success_rate,
        "domain_turn_acc": domain_turn_acc,
        "domain_call_acc": domain_call_acc,
        "overall_success": overall_success,
        "overall_call_acc": overall_call_acc,
        "complete_score_avg": complete_score_avg,
        "correct_score_avg": correct_score_avg
    }
    with open(summary_path, 'w') as f:
        json.dump(summary_metrics, f, indent=2)
    logging.info(f"Summary metrics saved to {summary_path}")


def process_single_case(runner, case, resp_eval_runner):
    """
    Process a single test case through the runner and evaluate the response.
    
    Args:
        runner: OrchestratorRunner instance
        case: Test case dictionary from the dataset
        resp_eval_runner: RespEvalRunner instance for response evaluation
        
    Returns:
        Dictionary with case results including metrics and evaluation scores
    """
    case_id = case.get('id', 'unknown')
    
    logger.info(f"Test Example {case_id}")
    logger.info(f"Query: {case['conversations'][0]['content']}")
    
    # Extract ground truth metrics from the case
    ground_truth_turn_count = 0
    ground_truth_call_count = 0
    for turn in case['conversations']:
        if turn['role'] == "assistant" and "function_call" in turn:
            ground_truth_turn_count += 1
            ground_truth_call_count += len(turn["function_call"])
    
    # Run the case through the orchestrator
    try:
        convs, message, success_turn_num, correct_call_num = runner.run(copy.deepcopy(case))
    except Exception as e:
        logger.error(f"Exception during case execution: {e}")
        return None
    
    # Check for API errors
    if isinstance(message, dict) and message.get("error_type") == "unknown_error":
        logger.error(f"API error on case {case_id}: {message}")
        return None
    
    # Count actual turns with function calls in generated conversation
    actual_turn_count = 0
    for turn in convs:
        if turn['role'] == "assistant" and "function_call" in turn:
            actual_turn_count += 1
    
    # Evaluate final response quality if available
    resp_eval_result = None
    if convs and convs[-1]['role'] == "assistant" and "content" in convs[-1]:
        gen_response = convs[-1]['content']
        if gen_response and resp_eval_runner:
            resp_eval_result = resp_eval_runner.run(case, gen_response)
    
    logger.info(f"Message: {message}")
    logger.info(f"Success turn num = {success_turn_num}")
    logger.info("-" * 100)
    
    # Build comprehensive result
    result = {
        "id": case_id,
        "gen_convs": convs,
        "message": message,
        "count_dict": {
            "success_turn_num": success_turn_num,
            "total_turn_num": ground_truth_turn_count,
            "correct_call_num": correct_call_num,
            "total_call_num": ground_truth_call_count,
            "real_turn_num": actual_turn_count
        },
        "resp_eval": resp_eval_result
    }
    
    return result

def run_configuration(orchestrator, dataset, model, memory, run_timestamp, resp_eval_runner=None):
    """
    Execute a complete benchmark run for a single model/memory configuration.
    
    Args:
        orchestrator: LLMOrchestrator instance
        dataset: List of test cases
        model: Model name string
        memory: Memory method name string
        run_timestamp: Timestamp string for this run
        resp_eval_runner: Optional RespEvalRunner for response evaluation
        
    Returns:
        Dictionary with run statistics
    """
    logger.info(f"🚀 Starting Run | Model: {model} | Memory: {memory}")
    
    # Set active context for this configuration
    try:
        orchestrator.set_active_context(model, memory)
    except Exception as e:
        logger.error(f"Failed to switch context: {e}")
        return None
    
    # Setup directory structure
    log_dir = os.path.join("results", orchestrator.cfg.experiment_name, run_timestamp, memory, model)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create file logger for the CFB runner
    runner_logger = FileLogger(
        f"runner_{run_timestamp}", 
        os.path.join(log_dir, f"cfb_{model}_{memory}_{run_timestamp}.log"), 
        level=logging.DEBUG
    )
    
    # Create runner with dummy args
    class RunnerArgs:
        def __init__(self, log_dir):
            self.log_dir = log_dir
    
    runner = OrchestratorRunner(RunnerArgs(log_dir), runner_logger, orchestrator)
    
    # Process all cases
    results = []
    success_count = 0
    
    for i, case in enumerate(dataset):
        case_id = case.get('id', i)
        
        logger.info("###" * 20)
        logger.info(f"Processing Case {case_id}...")
        
        try:
            result = process_single_case(runner, case, resp_eval_runner)
            
            if result is None:
                logger.warning(f"Case {case_id} returned None (likely API error)")
                continue
            
            # Determine success status
            status = "Success" if result['message'] == "Success." else "Failed"
            if status == "Success":
                success_count += 1
            
            # Append to results with additional metadata
            result['status'] = status
            result['memory_method'] = memory
            results.append(result)
            
        except Exception as e:
            logger.error(f"❌ Exception on Case {case_id}: {e}")
    
    # Calculate and report statistics
    pass_rate = (success_count / len(dataset)) * 100 if dataset else 0
    logger.info(f"🏁 Completed {model}/{memory}")
    logger.info(f"📊 Result: {success_count}/{len(dataset)} ({pass_rate:.1f}%) passed")
    
    # Save results to JSON
    result_file = os.path.join(log_dir, f"cfb_{model}_{memory}_{run_timestamp}.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"💾 Results saved to {result_file}")
    
    # Calculate detailed metrics
    logger.info("🧮 Calculating detailed metrics...")
    basic_metric(result_file)
    
    return {
        "model": model,
        "memory": memory,
        "total_cases": len(dataset),
        "success_count": success_count,
        "pass_rate": pass_rate
    }

def main():
    """
    Main orchestration function for running ComplexFuncBench evaluations.
    Coordinates dataset loading, configuration iteration, and result aggregation.
    """
    logger.info("Initializing ComplexFuncBench integration...")

    # Initialize Orchestrator
    try:
        orchestrator = LLMOrchestrator()
    except Exception as e:
        logger.critical(f"Failed to initialize Orchestrator: {e}")
        return

    # Load Dataset
    data_path = os.path.join(cfb_path, "data", "ComplexFuncBench.jsonl")
    dataset = load_jsonl(data_path)
    
    if not dataset:
        logger.warning("No data loaded. Exiting.")
        return

    # Optional: Sample subset for testing
    sample_size = orchestrator.cfg.benchmark_sample_size
    if sample_size:
        dataset = random.sample(dataset, sample_size)
    logger.info(f"Loaded {len(dataset)} evaluation samples.")

    # Initialize Response Evaluation Runner (reused across all configurations)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Create a temporary log dir for the response evaluator
    temp_log_dir = os.path.join("results", orchestrator.cfg.experiment_name, run_timestamp, "temp")
    os.makedirs(temp_log_dir, exist_ok=True)
    
    class RespEvalArgs:
        def __init__(self, log_dir):
            self.log_dir = log_dir
    
    resp_eval_runner = RespEvalRunner(args=RespEvalArgs(temp_log_dir), logger=logger)
    
    # Track overall statistics
    all_run_stats = []

    # Iterate over all model/memory configurations
    for model in orchestrator.cfg.enabled_models:
        for memory in orchestrator.cfg.enabled_memory_methods:
            stats = run_configuration(
                orchestrator=orchestrator,
                dataset=dataset,
                model=model,
                memory=memory,
                run_timestamp=run_timestamp,
                resp_eval_runner=resp_eval_runner
            )
            
            if stats:
                all_run_stats.append(stats)
    
    # Final summary
    logger.info("=" * 100)
    logger.info("🎉 All configurations completed!")
    logger.info("=" * 100)
    
    for stats in all_run_stats:
        logger.info(
            f"{stats['model']}/{stats['memory']}: "
            f"{stats['success_count']}/{stats['total_cases']} "
            f"({stats['pass_rate']:.1f}%)"
        )


if __name__ == "__main__":
    main()
