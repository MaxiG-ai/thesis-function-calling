import gc
import json
import os
import copy
import random
import logging
import shutil
import sys
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

from memorch.utils import model_load_lock
from memorch.utils.logger import get_logger
from memorch.utils.config import load_configs
from memorch.llm_orchestrator import LLMOrchestrator

from benchmarks.complex_func_bench.runner.sap_gpt_runner import SAPGPTRunner
from benchmarks.complex_func_bench.utils.logger import Logger as FileLogger
from benchmarks.complex_func_bench.runner.response_runner import RespEvalRunner
from benchmarks.complex_func_bench.utils.compare_method import CompareFC
from benchmarks.complex_func_bench.utils.utils import load_json
from tools.progress_dashboard import ConfigKey, ExperimentProgress

logger = get_logger("CFB_Runner")

CONFIGS_DIR = Path("configs")
MODEL_CONFIG_NAME = "model_config.toml"
BASELINE_HAYSTACK_SENTINEL = 0

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def normalize_experiment_config_name(experiment_config: str) -> str:
    """Normalize a CLI selector to a config file stem."""
    config_name = Path(experiment_config).name
    if config_name.endswith("_config.toml"):
        return config_name[: -len("_config.toml")]
    if config_name.endswith(".toml"):
        return config_name[: -len(".toml")]
    return experiment_config


def resolve_config_paths(
    experiment_config: str,
    config_dir: Path = CONFIGS_DIR,
) -> tuple[Path, Path]:
    """Resolve experiment and model config paths under the configs directory."""
    experiment_stem = normalize_experiment_config_name(experiment_config)
    experiment_config_path = config_dir / f"{experiment_stem}_config.toml"
    model_config_path = config_dir / MODEL_CONFIG_NAME

    if not experiment_config_path.exists():
        raise FileNotFoundError(
            f"Experiment config not found: {experiment_config_path}"
        )
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")

    return experiment_config_path, model_config_path


def load_runtime_config(
    experiment_config: str,
    config_dir: Path = CONFIGS_DIR,
):
    """Load the experiment-specific config and the shared model config."""
    experiment_config_path, model_config_path = resolve_config_paths(
        experiment_config,
        config_dir=config_dir,
    )
    config = load_configs(str(experiment_config_path), str(model_config_path))
    return config, experiment_config_path, model_config_path


def build_log_path(logs_dir: Path | str, experiment_name: str, run_timestamp: str) -> Path:
    """Build a log path that includes the experiment name."""
    safe_experiment_name = experiment_name.replace(os.sep, "_").replace(" ", "_")
    return Path(logs_dir) / f"experiment_run_{safe_experiment_name}_{run_timestamp}.log"


def save_config_snapshot(
    run_dir: Path | str,
    experiment_config_path: Path,
    model_config_path: Path,
) -> None:
    """Copy the exact TOML inputs used for a run into the run directory."""
    snapshot_dir = Path(run_dir) / "used_configs"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(experiment_config_path, snapshot_dir / experiment_config_path.name)
    shutil.copy2(model_config_path, snapshot_dir / model_config_path.name)


def initialize_response_evaluator(log_dir: str, config) -> RespEvalRunner:
    """Initialize the response quality evaluator."""

    class RespEvalArgs:
        def __init__(self, log_dir, config):
            self.log_dir = log_dir
            self.config = config

    return RespEvalRunner(args=RespEvalArgs(log_dir, config), logger=logger)


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
        args=RunnerArgs(log_dir),
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


def filter_dataset_by_ids(
    dataset: List[Dict], selected_ids: Optional[List[str]]
) -> List[Dict]:
    """Filter dataset by selected case IDs."""
    if selected_ids is None:
        return dataset
    selected_id_set = set(selected_ids)
    return [case for case in dataset if case.get("id") in selected_id_set]


def normalize_haystack_threshold(haystack_threshold: int) -> Optional[int]:
    """Map config-level sentinel values to runtime haystack thresholds."""
    return None if haystack_threshold == BASELINE_HAYSTACK_SENTINEL else haystack_threshold


def validate_haystack_thresholds_config(haystack_thresholds: Optional[List[int]]) -> List[int]:
    """Validate explicit haystack config and return the ordered threshold list.

    The config must explicitly include baseline via 0, for example:
    haystack_thresholds = [0, 15000, 50000]
    """
    if not haystack_thresholds:
        raise ValueError(
            "haystack_thresholds must be explicitly configured and include baseline 0. "
            "Example: haystack_thresholds = [0, 15000]"
        )

    if BASELINE_HAYSTACK_SENTINEL not in haystack_thresholds:
        raise ValueError(
            "haystack_thresholds must include baseline 0. "
            "Example: haystack_thresholds = [0, 15000]"
        )

    return list(haystack_thresholds)


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


def run_single_configuration(
    orchestrator: LLMOrchestrator,
    dataset: List[Dict],
    model: str,
    memory: str,
    run_timestamp: str,
    resp_eval_runner: RespEvalRunner,
    compact_threshold: Optional[int] = None,
    haystack_threshold: Optional[int] = None,
    shared_compare_class=None,
    progress: Optional[ExperimentProgress] = None,
) -> Optional[Dict]:
    """
    Run evaluation for a single model/memory/threshold configuration.

    This function:
    1. Sets the active context in the orchestrator (model, memory, threshold)
    2. Creates a shared CompareFC (FlagModel loaded once, reused across cases)
    3. Processes all test cases, updating the live progress dashboard per case
    4. Saves results to disk

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
        shared_compare_class: Optional pre-built CompareFC instance. When provided,
            skips internal CompareFC creation (avoids redundant FlagModel loading).
            Each thread must receive its own instance since CompareFC has mutable
            per-case state (free_functions, error_message).
        progress: Optional live dashboard instance. When provided, progress is
            reported per-case and per-configuration.

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

    # Register configuration in dashboard.
    config_key = ConfigKey(
        model=model,
        memory=memory,
        compact_threshold=compact_threshold,
        haystack_threshold=haystack_threshold,
    )
    if progress is not None:
        progress.start_configuration(config_key, total_cases=len(dataset))

    # Set active run context.
    try:
        orchestrator.set_active_context(
            model, memory, compact_threshold=compact_threshold
        )
    except Exception as e:
        logger.error(f"❌ Failed to switch context: {e}")
        return None

    # Build or reuse CompareFC for this configuration.
    if shared_compare_class is None:
        compare_class_args = type(
            "Args", (), {"log_dir": orchestrator.cfg.results_dir, "config": orchestrator.cfg}
        )()
        with model_load_lock:
            shared_compare_class = CompareFC(compare_class_args, logger)
        logger.info("🔧 CompareFC created (FlagModel loaded for this config)")
    else:
        logger.info("🔧 Using pre-built CompareFC (FlagModel shared)")

    # Process all cases.
    results = []
    compressed_traces = []  # Memory-processed messages.

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

            case_success = result["message"] == "Success."

            if progress is not None:
                progress.complete_case(
                    config_key, case_id=str(case_id), success=case_success
                )

            result["memory_method"] = memory
            results.append(result)

            compressed_traces.append(
                {
                    "id": case_id,
                    "memory_method": memory,
                    "compressed_trace": compressed_trace,
                }
            )

        except Exception as e:
            logger.error(f"❌ Failed on case {case_id}: {e}")
            if progress is not None:
                progress.complete_case(config_key, case_id=str(case_id), success=False)
            continue

    # Release CompareFC and CUDA cache.
    del shared_compare_class
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("🧹 Shared CompareFC released, GPU memory freed")

    logger.info("🧮 Calculating aggregate metrics...")
    metrics = calculate_metrics(results)

    log_dir = setup_directories(
        orchestrator.cfg.experiment_name,
        run_timestamp,
        model,
        memory,
        compact_threshold=compact_threshold,
        haystack_threshold=haystack_threshold,
    )

    save_results(
        results, metrics, model, memory, log_dir, run_timestamp, compressed_traces
    )

    if progress is not None:
        progress.complete_configuration(config_key, metrics=metrics)

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
    progress: Optional[ExperimentProgress] = None,
):
    """Run all memory method configurations for a single model.

        Execution order:
    - Memory methods iterate sequentially (shared gpt-4-1-mini refinement endpoint)
    - Compact thresholds iterate sequentially within each memory method
        - Haystack thresholds (baseline explicitly configured as 0) run in PARALLEL within each
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
        progress: Optional live dashboard instance passed through to each
            run_single_configuration call.
    """
    if threshold_sensitive is None:
        threshold_sensitive = {"truncation", "progressive_summarization"}

    # Build haystack threshold list from config.
    haystack_values = validate_haystack_thresholds_config(
        orchestrator.cfg.haystack_thresholds
    )

    # Resolve input_file from config for dataset loading
    input_file = orchestrator.cfg.input_file

    # Reuse immutable config for thread-specific orchestrators.
    shared_config = orchestrator.cfg

    for memory in memory_methods:
        strategy_type = orchestrator.cfg.memory_strategies[memory].type
        force_sequential_haystack = strategy_type == "memory_bank"

        # Only threshold-sensitive strategies produce one run per threshold value;
        # all others (ace, memory_bank, no_strategy) do a single run with no threshold.
        if strategy_type in threshold_sensitive:
            compact_thresholds = orchestrator.cfg.compact_thresholds
        else:
            compact_thresholds = [None]

        for compact_threshold in compact_thresholds:
            runtime_haystack_values = [
                normalize_haystack_threshold(ht) for ht in haystack_values
            ]

            if len(haystack_values) > 1 and not force_sequential_haystack:
                # Pre-create per-thread resources sequentially.
                logger.info(
                    f"🔀 Preparing {len(haystack_values)} haystack thresholds "
                    f"for {model}/{memory}/compact={compact_threshold}"
                )

                # Pre-load datasets and orchestrators.
                thread_orchestrators = {
                    runtime_ht: LLMOrchestrator(config=shared_config)
                    for runtime_ht in runtime_haystack_values
                }
                preloaded_datasets = {
                    runtime_ht: filter_dataset_by_ids(
                        load_haystack_dataset(runtime_ht, input_file=input_file),
                        selected_ids,
                    )
                    for runtime_ht in runtime_haystack_values
                }

                compare_class_args = type(
                    "Args", (), {"log_dir": shared_config.results_dir, "config": shared_config}
                )()
                thread_compare_classes = {}
                for runtime_ht in runtime_haystack_values:
                    with model_load_lock:
                        thread_compare_classes[runtime_ht] = CompareFC(
                            compare_class_args, logger
                        )
                logger.info(
                    f"🔧 Pre-created {len(haystack_values)} orchestrators, "
                    f"datasets, and CompareFC instances"
                )

                with ThreadPoolExecutor(max_workers=len(haystack_values)) as executor:
                    futures = {}
                    for runtime_ht in runtime_haystack_values:
                        future = executor.submit(
                            run_single_configuration,
                            orchestrator=thread_orchestrators[runtime_ht],
                            dataset=preloaded_datasets[runtime_ht],
                            model=model,
                            memory=memory,
                            run_timestamp=run_timestamp,
                            resp_eval_runner=resp_eval_runner,
                            compact_threshold=compact_threshold,
                            haystack_threshold=runtime_ht,
                            shared_compare_class=thread_compare_classes[runtime_ht],
                            progress=progress,
                        )
                        futures[future] = runtime_ht

                    for future in as_completed(futures):
                        haystack_threshold = futures[future]
                        ht_label = (
                            f"haystack={haystack_threshold}"
                            if haystack_threshold is not None
                            else "baseline"
                        )
                        try:
                            future.result()
                            logger.info(f"✅ {ht_label} completed")
                        except Exception as e:
                            logger.error(f"❌ {ht_label} failed: {e}")

                del thread_compare_classes
                del thread_orchestrators
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                logger.info("🧹 Thread resources released, GPU memory freed")
            else:
                # Run haystack thresholds sequentially.
                for runtime_ht in runtime_haystack_values:
                    dataset = filter_dataset_by_ids(
                        load_haystack_dataset(
                            runtime_ht,
                            input_file=input_file,
                        ),
                        selected_ids,
                    )
                    run_single_configuration(
                        orchestrator=orchestrator,
                        dataset=dataset,
                        model=model,
                        memory=memory,
                        run_timestamp=run_timestamp,
                        resp_eval_runner=resp_eval_runner,
                        compact_threshold=compact_threshold,
                        haystack_threshold=runtime_ht,
                        progress=progress,
                    )


def main(experiment_config: str):
    """
    Main orchestration function for ComplexFuncBench evaluation.

    This function:
    1. Starts the live progress dashboard
    2. Loads the orchestrator and determines which case IDs to evaluate
    3. Runs models sequentially (parallelism is at the haystack threshold level)
    4. Within each model, memory methods and compact thresholds run sequentially
    5. Haystack thresholds run in parallel (each thread gets its own orchestrator + dataset)
    """
    config, experiment_config_path, model_config_path = load_runtime_config(
        experiment_config
    )

    # Initialize base orchestrator.
    orchestrator = LLMOrchestrator(config=config)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Route logs to timestamped file.
    os.makedirs("logs", exist_ok=True)
    log_path = build_log_path(
        logs_dir="logs",
        experiment_name=orchestrator.cfg.experiment_name,
        run_timestamp=run_timestamp,
    )

    run_dir = (
        Path(orchestrator.cfg.results_dir)
        / "cfb"
        / orchestrator.cfg.experiment_name
        / run_timestamp
    )
    save_config_snapshot(run_dir, experiment_config_path, model_config_path)

    print(f"📝 Logs will be written to: {log_path}")
    print(f"📊 Starting live progress dashboard...\n")

    # Configure root logger to write to file.
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    )

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)

    # Configure all existing loggers.
    for name in list(logging.Logger.manager.loggerDict):
        if isinstance(logging.Logger.manager.loggerDict[name], logging.Logger):
            log_obj = logging.getLogger(name)
            log_obj.handlers.clear()
            log_obj.addHandler(file_handler)
            log_obj.propagate = False

    # explicitly setting httpx and litellm to silent
    logging.getLogger("litellm").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    # Start live dashboard.
    progress = ExperimentProgress()
    progress.start_experiment()

    try:
        logger.info(f"Experiment run started: {run_timestamp}")
        logger.info(f"Logs written to: {log_path}")
        logger.info(
            f"Saved config snapshot from {experiment_config_path} and {model_config_path}"
        )
        logger.info(
            f"📊 Progress dashboard started for: {orchestrator.cfg.experiment_name}"
        )

        # Resolve optional case filtering/sampling.
        selected_ids = None  # None = use full dataset (no filtering)
        input_file = orchestrator.cfg.input_file
        selected_test_cases = orchestrator.cfg.selected_test_cases
        if selected_test_cases:
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

        # Initialize shared response evaluator.
        temp_log_dir = os.path.join(
            "results", "cfb", orchestrator.cfg.experiment_name, run_timestamp, "temp"
        )
        os.makedirs(temp_log_dir, exist_ok=True)
        resp_eval_runner = initialize_response_evaluator(temp_log_dir, orchestrator.cfg)

        enabled_models = orchestrator.cfg.enabled_models
        memory_methods = orchestrator.cfg.enabled_memory_methods

        # Run models sequentially.
        for model in enabled_models:
            logger.info(f"🚀 Starting model: {model}")
            run_model_configs(
                model=model,
                memory_methods=memory_methods,
                orchestrator=orchestrator,
                run_timestamp=run_timestamp,
                resp_eval_runner=resp_eval_runner,
                selected_ids=selected_ids,
                progress=progress,
            )
            logger.info(f"✅ Model '{model}' completed all configurations")

        progress.finish_experiment()
        logger.info("=" * 80)
        logger.info("🎉 All configurations completed!")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.info("\n⚠️ Experiment interrupted by user (Ctrl+C)")
        progress._cleanup_alternate_screen()
        raise
    except Exception as e:
        logger.error(f"❌ Experiment failed with error: {e}")
        progress._cleanup_alternate_screen()
        raise


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: uv run cfb_run_eval.py <experiment_config_name>\n"
            "Example: uv run cfb_run_eval.py full_haystack_run_claude"
        )

    main(experiment_config=sys.argv[1])
