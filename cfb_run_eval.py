import gc
import json
import os
import copy
import random
import logging
import weave
import tomllib
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict

from memorch.utils import model_load_lock
from memorch.utils.logger import get_logger
from memorch.llm_orchestrator import LLMOrchestrator

from benchmarks.complex_func_bench.runner.sap_gpt_runner import SAPGPTRunner
from benchmarks.complex_func_bench.utils.logger import Logger as FileLogger
from benchmarks.complex_func_bench.runner.response_runner import RespEvalRunner
from benchmarks.complex_func_bench.utils.compare_method import CompareFC
from benchmarks.complex_func_bench.utils.utils import load_json

logger = get_logger("CFB_Runner")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def initialize_response_evaluator(log_dir: str) -> RespEvalRunner:
    """Initialize the response quality evaluator."""

    class RespEvalArgs:
        def __init__(self, log_dir):
            self.log_dir = log_dir

    return RespEvalRunner(args=RespEvalArgs(log_dir), logger=logger)


def setup_directories(
    experiment_name: str,
    run_timestamp: str,
    model: str,
    memory: str,
    compact_threshold: Optional[int] = None,
    haystack_threshold: Optional[int] = None,
) -> str:
    """Create directory structure for results.

    Directory layout includes both compact and haystack threshold segments
    so results are organized by all experimental axes.
    """
    # Include compact threshold in path for threshold-sensitive strategies
    threshold_segment = (
        f"threshold_{compact_threshold}"
        if compact_threshold is not None
        else "no_threshold"
    )
    # Include haystack threshold to separate NIAH experiment levels
    haystack_segment = (
        f"haystack_{haystack_threshold}"
        if haystack_threshold is not None
        else "no_haystack"
    )
    log_dir = os.path.join(
        "results",
        "cfb",
        experiment_name,
        run_timestamp,
        memory,
        threshold_segment,
        haystack_segment,
        model,
    )
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def create_runner(
    log_dir: str, orchestrator: LLMOrchestrator, compare_class=None
) -> SAPGPTRunner:
    """Create a CFB runner instance with orchestrator integration.

    Args:
        log_dir: Directory for logs
        orchestrator: LLM Orchestrator instance
        compare_class: Optional pre-built CompareFC to reuse (avoids reloading FlagModel)
    """

    class RunnerArgs:
        def __init__(self, log_dir):
            self.log_dir = log_dir

    # TODO: Can this be replace by the default logger?
    runner_logger = FileLogger(
        f"runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        os.path.join(log_dir, "cfb_runner.log"),
        level=logging.ERROR,
    )
    for handler in runner_logger.logger.handlers:
        handler.setLevel(logging.ERROR)

    # This routes all benchmark LLM calls through orchestrator with memory processing
    runner = SAPGPTRunner(
        model_name=orchestrator.active_model_key,
        args=RunnerArgs(log_dir),
        logger=runner_logger,
        orchestrator=orchestrator,
        compare_class=compare_class,
    )

    return runner


def load_haystack_dataset(
    haystack_threshold: Optional[int],
    input_file: str = os.path.join(
        "benchmarks", "complex_func_bench", "data", "ComplexFuncBench.jsonl"
    ),
) -> List[Dict]:
    """Load the appropriate dataset for a haystack threshold.

    For baseline (haystack_threshold=None), loads the original dataset at input_file.
    For a specific threshold, loads the pre-generated haystack_{threshold}.jsonl file
    from the same directory as input_file.

    Args:
        haystack_threshold: Token target for haystack context, or None for baseline.
        input_file: Path to the original dataset file (from config.input_file).
            Haystack files are expected in the same directory.

    Returns:
        List of case dicts, optionally with haystack_messages.

    Raises:
        FileNotFoundError: If the required data file does not exist.
    """
    if haystack_threshold is None:
        # Baseline: use original dataset without haystack augmentation
        file_path = input_file
    else:
        # Haystack files live alongside the original dataset
        data_dir = os.path.dirname(input_file)
        file_path = os.path.join(data_dir, f"haystack_{haystack_threshold}.jsonl")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}. "
            f"Run the haystack generation script first."
        )

    return load_json(file_path)


def extract_ground_truth_metrics(case: Dict) -> Dict[str, int]:
    """Extract ground truth metrics from a test case."""
    turn_count = 0
    call_count = 0

    for turn in case["conversations"]:
        if turn["role"] == "assistant" and "function_call" in turn:
            turn_count += 1
            call_count += len(turn["function_call"])

    return {"turn_count": turn_count, "call_count": call_count}


def extract_actual_metrics(convs: List[Dict]) -> Dict[str, int]:
    """Extract actual metrics from generated conversation."""
    turn_count = 0

    for turn in convs:
        if turn["role"] == "assistant" and "function_call" in turn:
            turn_count += 1

    return {"turn_count": turn_count}


def format_result_for_wandb(result: Dict) -> Dict:
    """
    Convert CFB result format to wandb-friendly format.

    This is a helper to transform the backwards-compatible result structure
    into a cleaner format for wandb logging.
    """
    wandb_result = {
        "case_id": result["id"],
        "status": result.get("status", "unknown"),
        "success": result["message"] == "Success.",
        "message": result["message"],
    }

    # Add count metrics
    count_dict = result.get("count_dict", {})
    if count_dict:
        total_turns = count_dict.get("total_turn_num", 1)
        total_calls = count_dict.get("total_call_num", 1)

        wandb_result.update(
            {
                "turn_accuracy": count_dict.get("success_turn_num", 0) / total_turns
                if total_turns > 0
                else 0,
                "call_accuracy": count_dict.get("correct_call_num", 0) / total_calls
                if total_calls > 0
                else 0,
                "success_turns": count_dict.get("success_turn_num", 0),
                "total_turns": total_turns,
                "correct_calls": count_dict.get("correct_call_num", 0),
                "total_calls": total_calls,
            }
        )

    # Add response evaluation scores if available
    resp_eval = result.get("resp_eval")
    if resp_eval:
        wandb_result.update(
            {
                "response_complete_score": resp_eval.get("complete", {}).get(
                    "score", None
                ),
                "response_correct_score": resp_eval.get("correct", {}).get(
                    "score", None
                ),
            }
        )

    # Extract domain from case ID (e.g., "Travel-001" -> "Travel")
    domain = result["id"].rsplit("-", 1)[0]
    wandb_result["domain"] = domain

    return wandb_result


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
        domain = result["id"].rsplit("-", 1)[0]

        if result["message"] == "Success.":
            domain_success[domain] += 1

        count_dict = result["count_dict"]
        domain_turn_count[domain][0] += count_dict["success_turn_num"]
        domain_turn_count[domain][1] += count_dict["total_turn_num"]
        domain_call_count[domain][0] += count_dict["correct_call_num"]
        domain_call_count[domain][1] += count_dict["total_call_num"]

        # Response evaluation scores
        resp_eval = result.get("resp_eval")
        if resp_eval:
            complete_score = resp_eval.get("complete", {}).get("score")
            if complete_score in {0, 1, 2}:
                complete_score_count[domain][0] += complete_score
                complete_score_count[domain][1] += 1

            correct_score = resp_eval.get("correct", {}).get("score")
            if correct_score in {0, 1, 2}:
                correct_score_count[domain][0] += correct_score
                correct_score_count[domain][1] += 1

    # Calculate rates and averages
    domain_success_rate = {
        k: v / 150 * 100 if k != "Cross" else v / 400 * 100
        for k, v in domain_success.items()
    }
    domain_turn_acc = {
        k: v[0] / v[1] * 100 if v[1] != 0 else 0 for k, v in domain_turn_count.items()
    }
    domain_call_acc = {
        k: v[0] / v[1] * 100 if v[1] != 0 else 0 for k, v in domain_call_count.items()
    }

    overall_success = sum(domain_success.values()) / len(results) * 100

    total_correct_calls = sum([v[0] for v in domain_call_count.values()])
    total_calls = sum([v[1] for v in domain_call_count.values()])
    overall_call_acc = total_correct_calls / total_calls * 100 if total_calls > 0 else 0

    # Calculate average scores
    complete_score_sum = sum([v[0] for v in complete_score_count.values()])
    complete_score_total = sum([v[1] for v in complete_score_count.values()])
    complete_score_avg = (
        complete_score_sum / complete_score_total if complete_score_total > 0 else 0
    )

    correct_score_sum = sum([v[0] for v in correct_score_count.values()])
    correct_score_total = sum([v[1] for v in correct_score_count.values()])
    correct_score_avg = (
        correct_score_sum / correct_score_total if correct_score_total > 0 else 0
    )

    # Build metrics dictionary
    metrics = {
        "domain_success_rate": domain_success_rate,
        "domain_turn_acc": domain_turn_acc,
        "domain_call_acc": domain_call_acc,
        "overall_success": overall_success,
        "overall_call_acc": overall_call_acc,
        "complete_score_avg": complete_score_avg,
        "correct_score_avg": correct_score_avg,
    }

    return metrics


def save_results(
    results: List[Dict],
    metrics: Dict,
    model: str,
    memory: str,
    log_dir: str,
    run_timestamp: str,
    compressed_traces: Optional[List[Dict]] = None,
):
    """Save results, metrics, and optionally compressed traces to disk."""
    # Save detailed results
    result_file = os.path.join(log_dir, f"cfb_{model}_{memory}_{run_timestamp}.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"💾 Results saved to {result_file}")

    # Save metrics summary
    metrics_file = os.path.join(
        log_dir, f"metrics_{model}_{memory}_{run_timestamp}.json"
    )
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"📊 Metrics saved to {metrics_file}")

    # Save compressed traces (memory-processed messages) if provided
    if compressed_traces:
        compressed_file = os.path.join(
            log_dir, f"compressed_{model}_{memory}_{run_timestamp}.json"
        )
        with open(compressed_file, "w") as f:
            json.dump(compressed_traces, f, indent=2)
        logger.info(f"🧠 Compressed traces saved to {compressed_file}")


def scrub_trace_args(inputs: Dict) -> Dict:
    """
    Filter out technical objects and redundant data from Weave logs.
    Used in postprocess_inputs for evaluate_single_case.
    """
    scrubbed = inputs.copy()

    # Remove technical objects that clutter logs
    keys_to_remove = ["orchestrator", "resp_eval_runner", "log_dir", "compare_class"]
    for key in keys_to_remove:
        if key in scrubbed:
            del scrubbed[key]

    # Simplify the 'case' object to avoid logging full conversation history at the root level
    if "case" in scrubbed and isinstance(scrubbed["case"], dict):
        # Only keep the ID and domain, remove the heavy 'conversations' list
        # This forces you to look at the child 'generate' trace for the actual messages
        scrubbed["case"] = {
            "id": scrubbed["case"].get("id"),
            "domain": scrubbed["case"].get("id", "").split("-")[0],
        }

    return scrubbed


@weave.op(postprocess_inputs=scrub_trace_args, enable_code_capture=False)
def evaluate_single_case(
    case: Dict,
    orchestrator: LLMOrchestrator,
    resp_eval_runner: RespEvalRunner,
    compare_class=None,
) -> tuple:
    """
    Evaluate a single test case.
    This function processes one case through the CFB benchmark runner.

    Args:
        case: Test case dictionary from the dataset
        orchestrator: LLM Orchestrator instance
        resp_eval_runner: Response quality evaluator
        compare_class: Optional pre-built CompareFC to reuse across cases

    Returns:
        Result dictionary in backwards-compatible CFB format
    """
    case_id = case.get("id", "unknown")

    # Set the trace name
    weave.require_current_call().display_name = (
        f"{case_id}_{orchestrator.active_model_key}_{orchestrator.active_memory_key}"
    )

    # Create runner for this case with orchestrator injection
    runner = create_runner(
        log_dir=orchestrator.cfg.results_dir,
        orchestrator=orchestrator,
        compare_class=compare_class,
    )

    # Extract ground truth metrics
    ground_truth = extract_ground_truth_metrics(case)

    # Execute the case (runner.run internally calls orchestrator.generate multiple times)
    try:
        convs, message, success_turn_num, correct_call_num = runner.run(
            copy.deepcopy(case)
        )
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
    if convs and convs[-1].get("role") == "assistant" and "content" in convs[-1]:
        final_response = convs[-1]["content"]
        if final_response and resp_eval_runner:
            resp_eval = resp_eval_runner.run(case, final_response)

    # Collect compressed trace from orchestrator (memory-processed messages)
    compressed_trace = orchestrator.get_compressed_trace_as_dicts()

    # Build result in backwards-compatible format
    result = {
        "id": case_id,
        "gen_convs": convs,
        "message": message,
        "count_dict": {
            "success_turn_num": success_turn_num,
            "total_turn_num": ground_truth["turn_count"],
            "correct_call_num": correct_call_num,
            "total_call_num": ground_truth["call_count"],
            "real_turn_num": actual["turn_count"],
        },
        "resp_eval": resp_eval,
        "status": "Success" if message == "Success." else "Failed",
    }

    # Return both result and compressed trace (compressed trace saved separately)
    return result, compressed_trace


def batch_log_to_weave(eval_logger, collected_results: List[Dict]):
    """Log all collected results to Weave after case processing completes.

    Decouples Weave network calls from the benchmark loop so they don't add
    latency to each case and so thread-safety is simpler.

    Args:
        eval_logger: Weave EvaluationLogger instance
        collected_results: List of formatted result dicts (from format_result_for_wandb)
    """
    for wandb_data in collected_results:
        with eval_logger.log_prediction(
            inputs={"case_id": wandb_data["case_id"], "domain": wandb_data["domain"]},
            output={
                "status": wandb_data["status"],
                "message": wandb_data["message"],
            },
        ) as pred:
            pred.log_score("success", 1.0 if wandb_data["success"] else 0.0)
            pred.log_score("turn_accuracy", wandb_data.get("turn_accuracy", 0.0))
            pred.log_score("call_accuracy", wandb_data.get("call_accuracy", 0.0))

            if wandb_data.get("response_complete_score") is not None:
                pred.log_score(
                    "response_complete", wandb_data["response_complete_score"]
                )
            if wandb_data.get("response_correct_score") is not None:
                pred.log_score("response_correct", wandb_data["response_correct_score"])


def run_single_configuration(
    orchestrator: LLMOrchestrator,
    dataset: List[Dict],
    model: str,
    memory: str,
    run_timestamp: str,
    resp_eval_runner: RespEvalRunner,
    compact_threshold: Optional[int] = None,
    haystack_threshold: Optional[int] = None,
) -> Optional[Dict]:
    """
    Run evaluation for a single model/memory/threshold configuration.

    This function:
    1. Sets the active context in the orchestrator (model, memory, threshold)
    2. Creates a shared CompareFC (FlagModel loaded once, reused across cases)
    3. Processes all test cases
    4. Batch-logs results to Weave (after case processing, not inline)
    5. Saves results to disk

    Args:
        orchestrator: LLM Orchestrator instance
        dataset: List of test cases (may include haystack_messages)
        model: Model identifier
        memory: Memory method identifier
        run_timestamp: Timestamp string for this run
        resp_eval_runner: Response quality evaluator
        compact_threshold: Token threshold for this run. None for threshold-insensitive
            strategies (ace, memory_bank, no_strategy); a value from
            config.compact_thresholds for truncation and progressive_summarization.
        haystack_threshold: NIAH haystack token target for this run. None for baseline
            (no distractor context injected).

    Returns:
        Summary statistics dictionary, or None if failed
    """
    threshold_label = (
        f"threshold={compact_threshold}"
        if compact_threshold is not None
        else "no_threshold"
    )
    haystack_label = (
        f"haystack={haystack_threshold}"
        if haystack_threshold is not None
        else "no_haystack"
    )
    logger.info(
        f"🚀 Starting evaluation: {model}/{memory}/{threshold_label}/{haystack_label}"
    )

    # Set active context (including threshold for this run)
    try:
        orchestrator.set_active_context(
            model, memory, compact_threshold=compact_threshold
        )
    except Exception as e:
        logger.error(f"❌ Failed to switch context: {e}")
        return None

    # Create a shared CompareFC for this configuration.
    # This loads FlagModel (BAAI/bge-large-en-v1.5, ~1.3GB) once instead of per case.
    # Per-case mutable state (free_functions, error_message) is reset at the start of
    # each runner.run() call, so reuse is safe.
    compare_class_args = type("Args", (), {"log_dir": orchestrator.cfg.results_dir})()
    with model_load_lock:
        shared_compare_class = CompareFC(compare_class_args, logger)
    logger.info("🔧 Shared CompareFC created (FlagModel loaded once for this config)")

    # Name this evaluation run including thresholds so wandb runs are distinguishable
    compact_suffix = f"_t{compact_threshold}" if compact_threshold is not None else ""
    haystack_suffix = (
        f"_h{haystack_threshold}" if haystack_threshold is not None else ""
    )
    eval_name = f"Eval_{model}_{memory}{compact_suffix}{haystack_suffix}"

    # Initialize weave evaluation logger for this configuration
    # Note: experiment config is provided at the global level via weave.init()
    eval_logger = weave.EvaluationLogger(
        name=eval_name,
        dataset="ComplexFuncBench",
        scorers=[
            "success",
            "turn_accuracy",
            "call_accuracy",
            "response_complete",
            "response_correct",
        ],
    )

    # Process all cases -- collect results locally for batch Weave logging
    results = []
    compressed_traces = []  # Separate list for memory-processed messages
    collected_wandb_data = []  # Collected for batch Weave logging after loop
    success_count = 0

    for i, case in enumerate(dataset):
        orchestrator.reset_session()
        case_id = case.get("id", i)
        logger.info(f"Processing case {i + 1}/{len(dataset)}: {case_id}")

        try:
            result, compressed_trace = evaluate_single_case(
                case=case,
                orchestrator=orchestrator,
                resp_eval_runner=resp_eval_runner,
                compare_class=shared_compare_class,
            )

            # Track success
            if result["message"] == "Success.":
                success_count += 1

            # Add metadata
            result["memory_method"] = memory
            results.append(result)

            # Collect compressed trace with case ID for reference
            compressed_traces.append(
                {
                    "id": case_id,
                    "memory_method": memory,
                    "compressed_trace": compressed_trace,
                }
            )

            # Collect formatted result for batch Weave logging (no network calls here)
            collected_wandb_data.append(format_result_for_wandb(result))

        except Exception as e:
            logger.error(f"❌ Failed on case {case_id}: {e}")
            # Continue with remaining cases
            continue

    # Release shared CompareFC and its FlagModel GPU memory
    del shared_compare_class
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("🧹 Shared CompareFC released, GPU memory freed")

    # Batch-log all predictions to Weave (after case processing)
    logger.info("📡 Batch-logging predictions to Weave...")
    batch_log_to_weave(eval_logger, collected_wandb_data)

    # Calculate aggregate metrics
    logger.info("🧮 Calculating aggregate metrics...")
    metrics = calculate_metrics(results)

    # Setup directories (threshold included in path for threshold-sensitive strategies)
    log_dir = setup_directories(
        orchestrator.cfg.experiment_name,
        run_timestamp,
        model,
        memory,
        compact_threshold=compact_threshold,
        haystack_threshold=haystack_threshold,
    )

    # Save results to disk (including compressed traces)
    save_results(
        results, metrics, model, memory, log_dir, run_timestamp, compressed_traces
    )

    # Log summary to wandb; include compact_threshold and haystack_threshold
    # so runs are distinguishable
    eval_logger.log_summary(
        {
            "model": model,
            "memory": memory,
            "compact_threshold": compact_threshold,
            "haystack_threshold": haystack_threshold,
            "total_cases": len(dataset),
            "success_count": success_count,
            "pass_rate": (success_count / len(dataset)) * 100 if dataset else 0,
            **metrics,
        }
    )

    logger.info(
        f"✅ Completed evaluation: {model}/{memory}/{threshold_label}/{haystack_label}"
    )


def run_model_configs(
    model: str,
    memory_methods: List[str],
    orchestrator: LLMOrchestrator,
    run_timestamp: str,
    resp_eval_runner: RespEvalRunner,
    selected_ids: Optional[List[str]] = None,
    threshold_sensitive: Optional[set] = None,
):
    """Run all memory method configurations for a single model.

    Execution order:
    - Memory methods iterate sequentially (shared gpt-4-1-mini refinement endpoint)
    - Compact thresholds iterate sequentially within each memory method
    - Haystack thresholds (including baseline=None) run in PARALLEL within each
      (memory, compact_threshold) combination. Each parallel thread gets its own
      LLMOrchestrator instance and loads its own dataset via load_haystack_dataset().

    Args:
        model: Model identifier
        memory_methods: List of memory method keys to evaluate
        orchestrator: LLM Orchestrator instance (used for config reading;
            parallel threads create their own instances)
        run_timestamp: Timestamp string for this run
        resp_eval_runner: Response quality evaluator (thread-safe, shared)
        selected_ids: Optional list of case IDs to filter to. Applied after
            loading each dataset. None means use full dataset.
        threshold_sensitive: Set of strategy types that require threshold sweeps
    """
    if threshold_sensitive is None:
        threshold_sensitive = {"truncation", "progressive_summarization"}

    # Build the list of haystack thresholds: baseline (None) + configured values
    haystack_values = [None] + (orchestrator.cfg.haystack_thresholds or [])

    # Resolve input_file from config for dataset loading
    input_file = orchestrator.cfg.input_file

    for memory in memory_methods:
        strategy_type = orchestrator.cfg.memory_strategies[memory].type

        # Only threshold-sensitive strategies produce one run per threshold value;
        # all others (ace, memory_bank, no_strategy) do a single run with no threshold.
        if strategy_type in threshold_sensitive:
            compact_thresholds = orchestrator.cfg.compact_thresholds
        else:
            compact_thresholds = [None]

        for compact_threshold in compact_thresholds:
            if len(haystack_values) > 1:
                # Parallel haystack execution: each haystack_threshold gets its own
                # orchestrator and dataset to avoid shared mutable state.
                logger.info(
                    f"🔀 Running {len(haystack_values)} haystack thresholds in "
                    f"parallel for {model}/{memory}/compact={compact_threshold}"
                )
                with ThreadPoolExecutor(max_workers=len(haystack_values)) as executor:
                    futures = {}
                    for ht in haystack_values:
                        # Each thread needs its own orchestrator (mutable per-session
                        # state) and loads its own dataset for this haystack level
                        thread_orchestrator = LLMOrchestrator()
                        dataset = load_haystack_dataset(ht, input_file=input_file)
                        if selected_ids is not None:
                            dataset = [
                                c for c in dataset if c.get("id") in selected_ids
                            ]
                        future = executor.submit(
                            run_single_configuration,
                            orchestrator=thread_orchestrator,
                            dataset=dataset,
                            model=model,
                            memory=memory,
                            run_timestamp=run_timestamp,
                            resp_eval_runner=resp_eval_runner,
                            compact_threshold=compact_threshold,
                            haystack_threshold=ht,
                        )
                        futures[future] = ht

                    # Collect results and report any errors
                    for future in as_completed(futures):
                        ht = futures[future]
                        ht_label = f"haystack={ht}" if ht is not None else "baseline"
                        try:
                            future.result()
                            logger.info(f"✅ {ht_label} completed")
                        except Exception as e:
                            logger.error(f"❌ {ht_label} failed: {e}")
            else:
                # No haystack thresholds configured — single baseline run
                dataset = load_haystack_dataset(None, input_file=input_file)
                if selected_ids is not None:
                    dataset = [c for c in dataset if c.get("id") in selected_ids]
                run_single_configuration(
                    orchestrator=orchestrator,
                    dataset=dataset,
                    model=model,
                    memory=memory,
                    run_timestamp=run_timestamp,
                    resp_eval_runner=resp_eval_runner,
                    compact_threshold=compact_threshold,
                    haystack_threshold=None,
                )


def main(experiment_name=None):
    """
    Main orchestration function for ComplexFuncBench evaluation.

    This function:
    1. Initializes wandb tracking
    2. Loads the orchestrator and determines which case IDs to evaluate
    3. Runs models sequentially (parallelism is at the haystack threshold level)
    4. Within each model, memory methods and compact thresholds run sequentially
    5. Haystack thresholds run in parallel (each thread gets its own orchestrator + dataset)
    """
    # Initialize orchestrator (used for config reading; parallel threads create their own)
    orchestrator = LLMOrchestrator()
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Initialize weave for the entire experiment and attach experiment-level metadata
    # (use a clear key name and include a run timestamp so traces can be correlated)
    weave.init(
        project_name=experiment_name or orchestrator.cfg.experiment_name,
        global_attributes={
            "experiment_config": orchestrator.get_exp_config(),
            "run_timestamp": run_timestamp,
        },
        settings={"implicitly_patch_integrations": False},
    )
    logger.info(
        f"📊 Weave initialized with global attributes: {orchestrator.cfg.experiment_name}"
    )

    # Determine which case IDs to run. Datasets are loaded per-haystack-threshold
    # inside run_model_configs, but filtering/sampling is decided here once.
    selected_ids = None  # None = use full dataset (no filtering)
    input_file = orchestrator.cfg.input_file
    selected_test_cases = orchestrator.cfg.selected_test_cases
    if selected_test_cases:
        # Explicit case IDs from config — validate they exist in the original dataset
        all_cases = load_json(input_file)
        all_ids = {case.get("id") for case in all_cases}
        missing = set(selected_test_cases) - all_ids
        if missing:
            logger.error(f"❌ Test case IDs not found in dataset: {missing}")
            return
        selected_ids = list(selected_test_cases)
        logger.info(
            f"🎯 Will filter to {len(selected_ids)} specific test case(s): {selected_ids}"
        )
    else:
        # Sample subset if configured (only when not using specific test cases)
        sample_size = orchestrator.cfg.benchmark_sample_size
        if sample_size is not None and sample_size > 0:
            all_cases = load_json(input_file)
            if sample_size > len(all_cases):
                logger.warning(
                    f"⚠️ Sample size {sample_size} exceeds dataset size {len(all_cases)}, "
                    "using full dataset"
                )
            else:
                random.seed(42)
                sampled = random.sample(all_cases, sample_size)
                selected_ids = [case.get("id") for case in sampled]
                logger.info(f"📊 Sampled {sample_size} case IDs from dataset")

    # Initialize response evaluator (shared across all configurations -- thread-safe)
    temp_log_dir = os.path.join(
        "results", "cfb", orchestrator.cfg.experiment_name, run_timestamp, "temp"
    )
    os.makedirs(temp_log_dir, exist_ok=True)
    resp_eval_runner = initialize_response_evaluator(temp_log_dir)

    enabled_models = orchestrator.cfg.enabled_models
    memory_methods = orchestrator.cfg.enabled_memory_methods

    # Models run sequentially; parallelism is at the haystack threshold level
    # inside run_model_configs. This avoids rate-limit spikes from multiple
    # models hitting the same API concurrently.
    for model in enabled_models:
        logger.info(f"🚀 Starting model: {model}")
        run_model_configs(
            model=model,
            memory_methods=memory_methods,
            orchestrator=orchestrator,
            run_timestamp=run_timestamp,
            resp_eval_runner=resp_eval_runner,
            selected_ids=selected_ids,
        )
        logger.info(f"✅ Model '{model}' completed all configurations")

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("🎉 All configurations completed!")
    logger.info("=" * 80)


if __name__ == "__main__":
    # import experiment name from toml config if available
    experiment_name = tomllib.load(open("config.toml", "rb")).get("experiment_name")
    if experiment_name:
        main(experiment_name=experiment_name)
    else:
        main(experiment_name="No_Experiment_Name")
