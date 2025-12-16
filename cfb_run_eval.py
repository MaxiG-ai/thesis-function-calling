
import json
import os
import sys
import copy
import random
import logging
import weave
import tomllib
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict
# Import custom logger
from src.utils.logger import get_logger
logger = get_logger("CFB_Runner")

# Import Orchestrator
try:
    from src.llm_orchestrator import LLMOrchestrator
except ImportError as e:
    logger.error("❌ Could not import LLMOrchestrator. Run this script from the root of the repository.")
    logger.error(f"ImportError: {e}")
    sys.exit(1)

# Import CFB components
try:
    from benchmarks.complex_func_bench.runner.sap_gpt_runner import SAPGPTRunner
    from benchmarks.complex_func_bench.utils.logger import Logger as FileLogger
    from benchmarks.complex_func_bench.runner.response_runner import RespEvalRunner
    from benchmarks.complex_func_bench.utils.utils import load_json
except ImportError as e:
    logger.error(f"❌ Failed to import CFB modules: {e}")
    sys.exit(1)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def initialize_response_evaluator(log_dir: str) -> RespEvalRunner:
    """Initialize the response quality evaluator."""
    class RespEvalArgs:
        def __init__(self, log_dir):
            self.log_dir = log_dir
    
    return RespEvalRunner(args=RespEvalArgs(log_dir), logger=logger)


def setup_directories(experiment_name: str, run_timestamp: str, model: str, memory: str) -> str:
    """Create directory structure for results."""
    log_dir = os.path.join("results", experiment_name, run_timestamp, memory, model)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def create_runner(log_dir: str, orchestrator: LLMOrchestrator) -> SAPGPTRunner:
    """Create a CFB runner instance with orchestrator integration."""
    class RunnerArgs:
        def __init__(self, log_dir):
            self.log_dir = log_dir
    
    # TODO: Can this be replace by the default logger?
    runner_logger = FileLogger(
        f"runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
        os.path.join(log_dir, "cfb_runner.log"), 
        level=logging.ERROR
    )
    for handler in runner_logger.logger.handlers:
        handler.setLevel(logging.ERROR)
    
    # This routes all benchmark LLM calls through orchestrator with memory processing
    runner = SAPGPTRunner(
        model_name=orchestrator.active_model_key, 
        args=RunnerArgs(log_dir), 
        logger=runner_logger,
        orchestrator=orchestrator
    )
    
    return runner


def extract_ground_truth_metrics(case: Dict) -> Dict[str, int]:
    """Extract ground truth metrics from a test case."""
    turn_count = 0
    call_count = 0
    
    for turn in case['conversations']:
        if turn['role'] == "assistant" and "function_call" in turn:
            turn_count += 1
            call_count += len(turn["function_call"])
    
    return {
        "turn_count": turn_count,
        "call_count": call_count
    }


def extract_actual_metrics(convs: List[Dict]) -> Dict[str, int]:
    """Extract actual metrics from generated conversation."""
    turn_count = 0
    
    for turn in convs:
        if turn['role'] == "assistant" and "function_call" in turn:
            turn_count += 1
    
    return {
        "turn_count": turn_count
    }


def format_result_for_wandb(result: Dict) -> Dict:
    """
    Convert CFB result format to wandb-friendly format.
    
    This is a helper to transform the backwards-compatible result structure
    into a cleaner format for wandb logging.
    """
    wandb_result = {
        "case_id": result['id'],
        "status": result.get('status', 'unknown'),
        "success": result['message'] == "Success.",
        "message": result['message'],
    }
    
    # Add count metrics
    count_dict = result.get('count_dict', {})
    if count_dict:
        total_turns = count_dict.get('total_turn_num', 1)
        total_calls = count_dict.get('total_call_num', 1)
        
        wandb_result.update({
            "turn_accuracy": count_dict.get('success_turn_num', 0) / total_turns if total_turns > 0 else 0,
            "call_accuracy": count_dict.get('correct_call_num', 0) / total_calls if total_calls > 0 else 0,
            "success_turns": count_dict.get('success_turn_num', 0),
            "total_turns": total_turns,
            "correct_calls": count_dict.get('correct_call_num', 0),
            "total_calls": total_calls,
        })
    
    # Add response evaluation scores if available
    resp_eval = result.get('resp_eval')
    if resp_eval:
        wandb_result.update({
            "response_complete_score": resp_eval.get('complete', {}).get('score', None),
            "response_correct_score": resp_eval.get('correct', {}).get('score', None),
        })
    
    # Extract domain from case ID (e.g., "Travel-001" -> "Travel")
    domain = result['id'].rsplit("-", 1)[0]
    wandb_result['domain'] = domain
    
    return wandb_result


def scrub_eval_args(inputs: Dict) -> Dict:
    """
    Filter out technical objects and redundant data from Weave logs.
    Used in postprocess_inputs for evaluate_single_case.
    """
    scrubbed = inputs.copy()
    
    # Remove technical objects that clutter logs
    keys_to_remove = ["orchestrator", "resp_eval_runner", "log_dir"]
    for key in keys_to_remove:
        if key in scrubbed:
            del scrubbed[key]
            
    # Simplify the 'case' object to avoid logging full conversation history at the root level
    if "case" in scrubbed and isinstance(scrubbed["case"], dict):
        # Only keep the ID and domain, remove the heavy 'conversations' list
        # This forces you to look at the child 'generate' trace for the actual messages
        scrubbed["case"] = {
            "id": scrubbed["case"].get("id"),
            "domain": scrubbed["case"].get("id", "").split("-")[0]
        }
        
    return scrubbed

# ============================================================================
# CORE EVALUATION FUNCTIONS
# ============================================================================
@weave.op(
        postprocess_inputs=scrub_eval_args
)
def evaluate_single_case(
    case: Dict,
    orchestrator: LLMOrchestrator,
    resp_eval_runner: RespEvalRunner,
) -> Dict:
    """
    Evaluate a single test case.
    This function processes one case through the CFB benchmark runner.
    
    Args:
        case: Test case dictionary from the dataset
        orchestrator: LLM Orchestrator instance
        resp_eval_runner: Response quality evaluator
        log_dir: Directory for logs
        
    Returns:
        Result dictionary in backwards-compatible CFB format
    """
    case_id = case.get('id', 'unknown')
    
    # Set the trace name
    weave.require_current_call().display_name = f"{case_id}_{orchestrator.active_model_key}_{orchestrator.active_memory_key}"
    
    # Create runner for this case with orchestrator injection
    runner = create_runner(log_dir=orchestrator.cfg.results_dir, orchestrator=orchestrator)
    
    # Extract ground truth metrics
    ground_truth = extract_ground_truth_metrics(case)
    
    # Execute the case (runner.run internally calls orchestrator.generate multiple times)
    try:
        convs, message, success_turn_num, correct_call_num = runner.run(copy.deepcopy(case))
    except Exception as e:
        logger.error(f"❌ Exception on case {case_id}: {e}")
        raise
    
    # Check for API errors
    if isinstance(message, dict) and message.get("error_type") == "unknown_error":
        logger.error(f"❌ API error on case {case_id}: {message}")
        raise RuntimeError("API Error encountered during case execution.")
    
    # Extract actual metrics
    actual = extract_actual_metrics(convs)
    
    # Evaluate response quality if available
    resp_eval = None
    if convs and convs[-1].get('role') == 'assistant' and 'content' in convs[-1]:
        final_response = convs[-1]['content']
        if final_response and resp_eval_runner:
            resp_eval = resp_eval_runner.run(case, final_response)
    
    # Build result in backwards-compatible format
    result = {
        "id": case_id,
        "gen_convs": convs,
        "message": message,
        "count_dict": {
            "success_turn_num": success_turn_num,
            "total_turn_num": ground_truth['turn_count'],
            "correct_call_num": correct_call_num,
            "total_call_num": ground_truth['call_count'],
            "real_turn_num": actual['turn_count']
        },
        "resp_eval": resp_eval,
        "status": "Success" if message == "Success." else "Failed"
    }
    
    return result


def calculate_metrics(results: List[Dict]) -> Dict:
    """
    Calculate aggregate metrics from evaluation results.
    
    This is a pure function that takes results and computes statistics.
    Refactored from the original basic_metric function.
    
    Args:
        results: List of result dictionaries
        
    Returns:
        Dictionary of computed metrics
    """
    if not results:
        logger.warning("⚠️ No results to calculate metrics from")
        return {}
    
    # Initialize accumulators
    domain_success = defaultdict(int)
    domain_turn_count = defaultdict(lambda: [0, 0])
    domain_call_count = defaultdict(lambda: [0, 0])
    complete_score_count = defaultdict(lambda: [0, 0])
    correct_score_count = defaultdict(lambda: [0, 0])
    
    # Aggregate metrics
    for result in results:
        domain = result['id'].rsplit("-", 1)[0]
        
        if result['message'] == "Success.":
            domain_success[domain] += 1
        
        count_dict = result['count_dict']
        domain_turn_count[domain][0] += count_dict['success_turn_num']
        domain_turn_count[domain][1] += count_dict['total_turn_num']
        domain_call_count[domain][0] += count_dict['correct_call_num']
        domain_call_count[domain][1] += count_dict['total_call_num']
        
        # Response evaluation scores
        resp_eval = result.get("resp_eval")
        if resp_eval:
            complete_score = resp_eval.get('complete', {}).get('score')
            if complete_score in {0, 1, 2}:
                complete_score_count[domain][0] += complete_score
                complete_score_count[domain][1] += 1
            
            correct_score = resp_eval.get('correct', {}).get('score')
            if correct_score in {0, 1, 2}:
                correct_score_count[domain][0] += correct_score
                correct_score_count[domain][1] += 1
    
    # Calculate rates and averages
    domain_success_rate = {
        k: v / 150 * 100 if k != "Cross" else v / 400 * 100 
        for k, v in domain_success.items()
    }
    domain_turn_acc = {
        k: v[0] / v[1] * 100 if v[1] != 0 else 0 
        for k, v in domain_turn_count.items()
    }
    domain_call_acc = {
        k: v[0] / v[1] * 100 if v[1] != 0 else 0 
        for k, v in domain_call_count.items()
    }
    
    overall_success = sum(domain_success.values()) / len(results) * 100
    
    total_correct_calls = sum([v[0] for v in domain_call_count.values()])
    total_calls = sum([v[1] for v in domain_call_count.values()])
    overall_call_acc = total_correct_calls / total_calls * 100 if total_calls > 0 else 0
    
    # Calculate average scores
    complete_score_sum = sum([v[0] for v in complete_score_count.values()])
    complete_score_total = sum([v[1] for v in complete_score_count.values()])
    complete_score_avg = complete_score_sum / complete_score_total if complete_score_total > 0 else 0
    
    correct_score_sum = sum([v[0] for v in correct_score_count.values()])
    correct_score_total = sum([v[1] for v in correct_score_count.values()])
    correct_score_avg = correct_score_sum / correct_score_total if correct_score_total > 0 else 0
    
    # Build metrics dictionary
    metrics = {
        "domain_success_rate": domain_success_rate,
        "domain_turn_acc": domain_turn_acc,
        "domain_call_acc": domain_call_acc,
        "overall_success": overall_success,
        "overall_call_acc": overall_call_acc,
        "complete_score_avg": complete_score_avg,
        "correct_score_avg": correct_score_avg
    }
    
    return metrics


def save_results(
    results: List[Dict],
    metrics: Dict,
    model: str,
    memory: str,
    log_dir: str,
    run_timestamp: str
):
    """Save results and metrics to disk."""
    # Save detailed results
    result_file = os.path.join(log_dir, f"cfb_{model}_{memory}_{run_timestamp}.json")
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"💾 Results saved to {result_file}")
    
    # Save metrics summary
    metrics_file = os.path.join(log_dir, f"metrics_{model}_{memory}_{run_timestamp}.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"📊 Metrics saved to {metrics_file}")


# ============================================================================
# CONFIGURATION RUNNER
# ============================================================================

def run_single_configuration(
    orchestrator: LLMOrchestrator,
    dataset: List[Dict],
    model: str,
    memory: str,
    run_timestamp: str,
    resp_eval_runner: RespEvalRunner
) -> Optional[Dict]:
    """
    Run evaluation for a single model/memory configuration.
    
    This function:
    1. Sets the active context in the orchestrator
    2. Processes all test cases
    3. Calculates and logs metrics to wandb
    4. Saves results to disk
    
    Args:
        orchestrator: LLM Orchestrator instance
        dataset: List of test cases
        model: Model identifier
        memory: Memory method identifier
        run_timestamp: Timestamp string for this run
        resp_eval_runner: Response quality evaluator
        
    Returns:
        Summary statistics dictionary, or None if failed
    """
    logger.info(f"🚀 Starting evaluation: {model}/{memory}")
    
    # Set active context
    try:
        orchestrator.set_active_context(model, memory)
    except Exception as e:
        logger.error(f"❌ Failed to switch context: {e}")
        return None
    
    # Initialize weave evaluation logger for this configuration
    eval_logger = weave.EvaluationLogger(
        name=f"Eval_{model}_{memory}",
        model=model,
        dataset="ComplexFuncBench",
        eval_attributes={"memory_method": memory, "config": orchestrator.get_exp_config()},
        scorers=["success", "turn_accuracy", "call_accuracy", "response_complete", "response_correct"],
    )
    
    # Process all cases
    results = []
    success_count = 0
    
    for i, case in enumerate(dataset):
        case_id = case.get('id', i)
        logger.info(f"Processing case {i+1}/{len(dataset)}: {case_id}")
        
        try:

            with weave.attributes({"case_id": case_id, "memory_method": memory, "model": model}):
                result = evaluate_single_case(
                    case=case,
                    orchestrator=orchestrator,
                    resp_eval_runner=resp_eval_runner,
                )
            
            # Track success
            if result['message'] == "Success.":
                success_count += 1
            
            # Add metadata
            result['memory_method'] = memory
            results.append(result)
            
            # Log case prediction to wandb
            wandb_data = format_result_for_wandb(result)
            with eval_logger.log_prediction(
                inputs={"case_id": case_id, "domain": wandb_data['domain']},
                output={"status": wandb_data['status'], "message": wandb_data['message']}
            ) as pred:
                # Log scores for this prediction
                pred.log_score("success", 1.0 if wandb_data['success'] else 0.0)
                pred.log_score("turn_accuracy", wandb_data.get('turn_accuracy', 0.0))
                pred.log_score("call_accuracy", wandb_data.get('call_accuracy', 0.0))
                
                if wandb_data.get('response_complete_score') is not None:
                    pred.log_score("response_complete", wandb_data['response_complete_score'])
                if wandb_data.get('response_correct_score') is not None:
                    pred.log_score("response_correct", wandb_data['response_correct_score'])
            
        except Exception as e:
            logger.error(f"❌ Failed on case {case_id}: {e}")
            # Continue with remaining cases
            continue
    
    # Calculate aggregate metrics
    logger.info("🧮 Calculating aggregate metrics...")
    metrics = calculate_metrics(results)

    # Setup directories
    log_dir = setup_directories(
        orchestrator.cfg.experiment_name,
        run_timestamp,
        model,
        memory
    )
    
    # Save results to disk
    save_results(results, metrics, model, memory, log_dir, run_timestamp)
    
    # Build summary
    summary = {
        "model": model,
        "memory": memory,
        "total_cases": len(dataset),
        "success_count": success_count,
        "pass_rate": (success_count / len(dataset)) * 100 if dataset else 0,
        **metrics
    }
    
    # Log summary to wandb
    eval_logger.log_summary(summary)
    
    logger.info(f"✅ Completed evaluation: {model}/{memory}")
    
    return summary


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def main(experiment_name=None):
    """
    Main orchestration function for ComplexFuncBench evaluation.
    
    This function:
    1. Initializes wandb tracking
    2. Loads the orchestrator and dataset
    3. Iterates through all model/memory configurations
    4. Aggregates and reports final results
    """

    logger.info("=" * 80)
    logger.info("ComplexFuncBench Evaluation")
    logger.info("=" * 80)
    
    # Initialize orchestrator
    orchestrator = LLMOrchestrator()
    
    # Initialize wandb for the entire experiment
    if experiment_name:
        weave.init(experiment_name)
    else:
        weave.init(orchestrator.cfg.experiment_name)
    logger.info(f"📊 Weave initialized: {orchestrator.cfg.experiment_name}")
    
    # Load dataset
    data_path = os.path.join("benchmarks", "complex_func_bench", "data", "ComplexFuncBench.jsonl")
    dataset = load_json(data_path)
    
    if not dataset:
        logger.error("❌ No data loaded. Exiting.")
        return
    
    # Sample subset if configured
    sample_size = orchestrator.cfg.benchmark_sample_size
    if sample_size is not None and sample_size > 0:
        if sample_size > len(dataset):
            logger.warning(
                f"⚠️ Sample size {sample_size} exceeds dataset size {len(dataset)}, "
                "using full dataset"
            )
        else:
            random.seed(42)
            dataset = random.sample(dataset, sample_size)
            logger.info(f"📊 Sampled {sample_size} cases from dataset")
            
    # Initialize response evaluator (shared across all configurations)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    temp_log_dir = os.path.join("results", orchestrator.cfg.experiment_name, run_timestamp, "temp")
    os.makedirs(temp_log_dir, exist_ok=True)
    resp_eval_runner = initialize_response_evaluator(temp_log_dir)
    
    # Track all run statistics
    all_summaries = []
    
    for model in orchestrator.cfg.enabled_models:
        for memory in orchestrator.cfg.enabled_memory_methods:

            # Run one of the cross product results memory - model            
            summary = run_single_configuration(
                orchestrator=orchestrator,
                dataset=dataset,
                model=model,
                memory=memory,
                run_timestamp=run_timestamp,
                resp_eval_runner=resp_eval_runner
            )
            
            if summary:
                all_summaries.append(summary)
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("🎉 All configurations completed!")
    logger.info("=" * 80)
    
    for summary in all_summaries:
        logger.info(
            f"{summary['model']}/{summary['memory']}: "
            f"{summary['success_count']}/{summary['total_cases']} "
            f"({summary['pass_rate']:.1f}%) - "
            f"Overall Success: {summary.get('overall_success', 0):.1f}%"
        )


if __name__ == "__main__":
    # import experiment name from toml config if available
    experiment_name = tomllib.load(open("config.toml", "rb")).get("experiment_name")
    if experiment_name:
        main(experiment_name=experiment_name)
    else:
        main(experiment_name="No_Experiment_Name")