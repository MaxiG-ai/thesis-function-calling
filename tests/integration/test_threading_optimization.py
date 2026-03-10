"""
Tests for the threading optimization refactor in cfb_run_eval.

These tests verify that the parallelization changes eliminate the
RuntimeError caused by concurrent LLMOrchestrator construction, and
that pre-creation of resources (orchestrators, datasets, CompareFC)
outside the ThreadPoolExecutor works correctly.

Root cause:
    LLMOrchestrator.__init__ -> load_configs() -> set_global_log_level()
    iterates logging.root.manager.loggerDict. When multiple threads
    construct LLMOrchestrator concurrently, concurrent get_logger() calls
    mutate the same dict, causing RuntimeError. The fix pre-creates all
    per-thread resources sequentially before submitting to threads.

Test categories:
1. Pre-creation of orchestrators using shared config
2. Pre-loading of datasets before thread submission
3. End-to-end run_model_configs with pre-created resources
4. CompareFC sharing via optional parameter
"""

import json
import pytest
from unittest.mock import MagicMock, patch, call
from typing import List, Dict, Optional

import cfb_run_eval
from memorch.utils.config import ExperimentConfig, MemoryDef, ModelDef, load_configs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_exp_config(
    compact_thresholds: List[int],
    haystack_thresholds: Optional[List[int]] = None,
    strategy_type: str = "truncation",
) -> ExperimentConfig:
    """Build a minimal ExperimentConfig for testing."""
    memory_def = MemoryDef(type=strategy_type)
    model_def = ModelDef(
        litellm_name="openai/gpt-test",
        context_window=128000,
        provider="test",
    )
    config_dict = {
        "experiment_name": "test_threading",
        "results_dir": "results",
        "log_dir": "logs",
        "logging_level": "WARNING",
        "weave_logging": False,
        "input_file": "benchmarks/complex_func_bench/data/ComplexFuncBench.jsonl",
        "enabled_models": ["gpt-test"],
        "enabled_memory_methods": [strategy_type],
        "compact_thresholds": compact_thresholds,
        "memory_strategies": {strategy_type: memory_def},
        "model_registry": {"gpt-test": model_def},
    }
    if haystack_thresholds is not None:
        config_dict["haystack_thresholds"] = haystack_thresholds
    return ExperimentConfig.model_validate(config_dict)


# ---------------------------------------------------------------------------
# 1. Pre-creation of orchestrators using shared config
# ---------------------------------------------------------------------------


def test_orchestrator_accepts_preloaded_config():
    """
    LLMOrchestrator must accept a pre-loaded ExperimentConfig via the
    config parameter, skipping file I/O and set_global_log_level().
    This is the foundation for thread-safe parallel creation: load_configs
    is called once, and the resulting config is shared with all threads.
    """
    from memorch.llm_orchestrator import LLMOrchestrator

    cfg = load_configs("config.toml", "model_config.toml")

    # Patch load_configs to verify it is NOT called when config is passed
    with patch("memorch.llm_orchestrator.load_configs") as mock_load:
        orch = LLMOrchestrator(config=cfg)
        mock_load.assert_not_called()

    assert orch.cfg is cfg


def test_run_model_configs_does_not_construct_orchestrators_in_threads():
    """
    After the fix, run_model_configs must NOT call LLMOrchestrator() inside
    executor.submit() lambdas. All orchestrators should be pre-created
    sequentially before the ThreadPoolExecutor. This test verifies that
    no LLMOrchestrator() construction occurs inside thread workers.

    We patch LLMOrchestrator to track where it's called from and verify
    all calls happen on the main thread (i.e. before thread submission).
    """
    import threading

    cfg = _make_exp_config(
        compact_thresholds=[5000],
        haystack_thresholds=[20000, 40000],
        strategy_type="truncation",
    )

    main_thread = threading.current_thread()
    construction_threads = []

    original_init = cfb_run_eval.LLMOrchestrator.__init__

    def tracking_init(self, *args, **kwargs):
        construction_threads.append(threading.current_thread())
        # Use a mock cfg to avoid actual init
        self.cfg = cfg
        self.memory_processor = MagicMock()
        self.active_model_key = cfg.enabled_models[0]
        self.active_memory_key = cfg.enabled_memory_methods[0]
        self.active_compact_threshold = None
        self._compressed_trace_buffer = []
        self._trace_step_counter = 0
        self.last_compressed_view = None

    orchestrator = MagicMock()
    orchestrator.cfg = cfg

    with (
        patch.object(cfb_run_eval, "run_single_configuration"),
        patch.object(cfb_run_eval, "load_haystack_dataset", return_value=[]),
        patch.object(
            cfb_run_eval.LLMOrchestrator,
            "__init__",
            tracking_init,
        ),
    ):
        cfb_run_eval.run_model_configs(
            model="gpt-test",
            memory_methods=["truncation"],
            orchestrator=orchestrator,
            run_timestamp="20260305_0000",
            resp_eval_runner=MagicMock(),
            selected_ids=None,
        )

    # All orchestrator constructions must happen on main thread
    for t in construction_threads:
        assert t == main_thread, (
            f"LLMOrchestrator was constructed on thread {t.name}, "
            f"expected main thread {main_thread.name}"
        )


# ---------------------------------------------------------------------------
# 2. Pre-loading of datasets before thread submission
# ---------------------------------------------------------------------------


def test_datasets_preloaded_before_thread_submission():
    """
    Datasets for each haystack threshold must be loaded sequentially before
    the ThreadPoolExecutor block, not inside each thread. This is verified
    by patching load_haystack_dataset and checking it's called before
    run_single_configuration (which runs in threads).

    The ordering contract: all load_haystack_dataset calls happen first,
    then all run_single_configuration calls happen.
    """
    cfg = _make_exp_config(
        compact_thresholds=[5000],
        haystack_thresholds=[20000],
        strategy_type="truncation",
    )

    call_order = []

    def tracked_load(haystack_threshold, input_file=None):
        call_order.append(("load", haystack_threshold))
        return [{"id": "test-1", "conversations": [], "functions": []}]

    def tracked_run(**kwargs):
        call_order.append(("run", kwargs.get("haystack_threshold")))

    orchestrator = MagicMock()
    orchestrator.cfg = cfg

    with (
        patch.object(cfb_run_eval, "run_single_configuration", side_effect=tracked_run),
        patch.object(cfb_run_eval, "load_haystack_dataset", side_effect=tracked_load),
        patch("memorch.llm_orchestrator.LLMOrchestrator", return_value=MagicMock()),
    ):
        cfb_run_eval.run_model_configs(
            model="gpt-test",
            memory_methods=["truncation"],
            orchestrator=orchestrator,
            run_timestamp="20260305_0000",
            resp_eval_runner=MagicMock(),
            selected_ids=None,
        )

    # All loads must precede all runs
    load_indices = [i for i, (op, _) in enumerate(call_order) if op == "load"]
    run_indices = [i for i, (op, _) in enumerate(call_order) if op == "run"]

    assert len(load_indices) > 0, "load_haystack_dataset was never called"
    assert len(run_indices) > 0, "run_single_configuration was never called"
    assert max(load_indices) < min(run_indices), (
        f"Some dataset loads happened after thread submission. Call order: {call_order}"
    )


# ---------------------------------------------------------------------------
# 3. End-to-end run_model_configs with pre-created resources
# ---------------------------------------------------------------------------


def test_parallel_haystack_passes_distinct_orchestrators():
    """
    Each parallel haystack thread must receive its own LLMOrchestrator
    instance (not the same object). This ensures mutable session state
    (_compressed_trace_buffer, last_compressed_view) is not shared.

    We verify by collecting the orchestrator object IDs passed to
    run_single_configuration and asserting they are all distinct.
    """
    cfg = _make_exp_config(
        compact_thresholds=[5000],
        haystack_thresholds=[20000, 40000],
        strategy_type="truncation",
    )

    orchestrator_ids = []

    def capture_run(**kwargs):
        orchestrator_ids.append(id(kwargs["orchestrator"]))

    orchestrator = MagicMock()
    orchestrator.cfg = cfg

    with (
        patch.object(cfb_run_eval, "run_single_configuration", side_effect=capture_run),
        patch.object(cfb_run_eval, "load_haystack_dataset", return_value=[]),
        patch(
            "memorch.llm_orchestrator.LLMOrchestrator",
            side_effect=lambda **kw: MagicMock(),
        ),
    ):
        cfb_run_eval.run_model_configs(
            model="gpt-test",
            memory_methods=["truncation"],
            orchestrator=orchestrator,
            run_timestamp="20260305_0000",
            resp_eval_runner=MagicMock(),
            selected_ids=None,
        )

    # 3 haystack values: None, 20000, 40000 -> 3 distinct orchestrators
    assert len(orchestrator_ids) == 3
    assert len(set(orchestrator_ids)) == 3, (
        "All parallel threads must receive distinct orchestrator instances"
    )


def test_parallel_haystack_correct_threshold_values():
    """
    Each parallel thread must receive the correct haystack_threshold value
    in its run_single_configuration call. With haystack_thresholds=[20000, 40000],
    the submitted values must be {None, 20000, 40000}.
    """
    cfg = _make_exp_config(
        compact_thresholds=[5000],
        haystack_thresholds=[20000, 40000],
        strategy_type="truncation",
    )

    submitted_thresholds = []

    def capture_run(**kwargs):
        submitted_thresholds.append(kwargs.get("haystack_threshold"))

    orchestrator = MagicMock()
    orchestrator.cfg = cfg

    with (
        patch.object(cfb_run_eval, "run_single_configuration", side_effect=capture_run),
        patch.object(cfb_run_eval, "load_haystack_dataset", return_value=[]),
        patch("memorch.llm_orchestrator.LLMOrchestrator", return_value=MagicMock()),
    ):
        cfb_run_eval.run_model_configs(
            model="gpt-test",
            memory_methods=["truncation"],
            orchestrator=orchestrator,
            run_timestamp="20260305_0000",
            resp_eval_runner=MagicMock(),
            selected_ids=None,
        )

    assert set(submitted_thresholds) == {None, 20000, 40000}


# ---------------------------------------------------------------------------
# 4. CompareFC sharing via optional parameter
# ---------------------------------------------------------------------------


def test_run_single_configuration_accepts_shared_compare_class():
    """
    run_single_configuration must accept an optional shared_compare_class
    parameter. When provided, it must skip internal CompareFC creation
    (avoiding redundant FlagModel loading) and use the provided instance.

    This verifies the function signature includes the new parameter.
    """
    import inspect

    sig = inspect.signature(cfb_run_eval.run_single_configuration)
    params = list(sig.parameters.keys())
    assert "shared_compare_class" in params, (
        "run_single_configuration must accept shared_compare_class parameter"
    )


def test_run_single_configuration_skips_compare_creation_when_provided():
    """
    When shared_compare_class is passed to run_single_configuration,
    the function must NOT create a new CompareFC internally. This saves
    ~1.3GB GPU memory per thread by reusing the FlagModel.

    We verify by patching CompareFC and asserting it's never called
    when a pre-built instance is provided.
    """
    cfg = _make_exp_config(compact_thresholds=[5000], strategy_type="truncation")

    mock_orchestrator = MagicMock()
    mock_orchestrator.cfg = cfg
    mock_orchestrator.cfg.results_dir = "results"
    mock_orchestrator.get_exp_config.return_value = {}

    mock_compare = MagicMock()  # Pre-built CompareFC

    with (
        patch.object(
            cfb_run_eval,
            "evaluate_single_case",
            return_value=(
                {
                    "id": "t-1",
                    "message": "Success.",
                    "count_dict": {
                        "success_turn_num": 1,
                        "total_turn_num": 1,
                        "correct_call_num": 1,
                        "total_call_num": 1,
                    },
                    "resp_eval": None,
                    "status": "Success",
                },
                [],
            ),
        ),
        patch.object(cfb_run_eval, "CompareFC") as mock_compare_cls,
        patch.object(cfb_run_eval, "calculate_metrics", return_value={}),
        patch.object(cfb_run_eval, "setup_directories", return_value="/tmp/test"),
        patch.object(cfb_run_eval, "save_results"),
    ):
        cfb_run_eval.run_single_configuration(
            orchestrator=mock_orchestrator,
            dataset=[{"id": "t-1", "conversations": [], "functions": []}],
            model="gpt-test",
            memory="truncation",
            run_timestamp="20260305_0000",
            resp_eval_runner=MagicMock(),
            compact_threshold=5000,
            haystack_threshold=None,
            shared_compare_class=mock_compare,
        )

        # CompareFC constructor must NOT be called when pre-built instance is passed
        mock_compare_cls.assert_not_called()
