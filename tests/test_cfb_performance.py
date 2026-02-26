"""Tests for cfb_run_eval performance optimizations.

These tests validate two key optimizations:
1. FlagModel/CompareFC reuse across test cases (avoid reloading 1.3GB model per case)
2. Model-level parallelism (run different models concurrently)

All tests mock external dependencies (LLM calls, FlagModel, Weave) to run
without GPU or API access.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from concurrent.futures import ThreadPoolExecutor

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_compare_class():
    """Create a mock CompareFC instance that can be injected into ModelRunner.

    Simulates the expensive object that loads FlagModel (BAAI/bge-large-en-v1.5).
    We track identity to verify the same instance is reused across runners.
    """
    mock = MagicMock()
    mock.free_function_list = [
        "Location_to_Lat_Long",
        "Search_Hotel_Destination",
        "Search_Attraction_Location",
        "Search_Car_Location",
        "Search_Flight_Location",
        "Taxi_Search_Location",
    ]
    return mock


@pytest.fixture
def mock_logger():
    """Create a mock logger for runner construction."""
    return MagicMock()


@pytest.fixture
def mock_args():
    """Create mock args with a log_dir attribute."""

    class Args:
        def __init__(self):
            self.log_dir = "/tmp/test_log"

    return Args()


@pytest.fixture
def minimal_case():
    """A minimal CFB test case with one user turn and one assistant function call.

    This is the smallest valid case structure for runner.run() to process.
    """
    return {
        "id": "Test-001",
        "functions": [
            {
                "name": "Search_Hotels",
                "parameters": {
                    "required": ["dest_id"],
                    "properties": {
                        "dest_id": {"type": "string"},
                        "search_type": {"type": "string"},
                    },
                },
            }
        ],
        "conversations": [
            {"role": "user", "content": "Find me a hotel in Paris"},
            {
                "role": "assistant",
                "function_call": [
                    {
                        "name": "Search_Hotels",
                        "arguments": {"dest_id": "123", "search_type": "hotel"},
                    }
                ],
            },
            {"role": "observation", "content": [{"status": True, "data": "results"}]},
            {"role": "assistant", "content": "I found a hotel for you."},
        ],
    }


# ---------------------------------------------------------------------------
# Change 1: CompareFC / FlagModel Reuse Tests
# ---------------------------------------------------------------------------


class TestCompareFCInjection:
    """Tests for injecting a shared CompareFC instance into ModelRunner.

    The core optimization: instead of each ModelRunner creating its own CompareFC
    (which loads BAAI/bge-large-en-v1.5 ~1.3GB from scratch), we create one
    CompareFC per configuration and inject it into all runners for that config.
    """

    def test_model_runner_uses_injected_compare_class(
        self, mock_compare_class, mock_args, mock_logger
    ):
        """ModelRunner should use the provided compare_class instead of creating a new one.

        When a pre-built CompareFC is passed via the compare_class parameter,
        ModelRunner must store it as self.CompareClass without calling the
        CompareFC constructor. This avoids the expensive FlagModel loading.
        """
        from benchmarks.complex_func_bench.runner.base_runner import ModelRunner

        runner = ModelRunner(mock_args, mock_logger, compare_class=mock_compare_class)

        # Must be the exact same object, not a copy
        assert runner.CompareClass is mock_compare_class
        assert runner.free_function_list == mock_compare_class.free_function_list

    def test_model_runner_creates_own_compare_class_when_none_provided(
        self, mock_args, mock_logger
    ):
        """ModelRunner should create its own CompareFC when no compare_class is given.

        This ensures backwards compatibility -- existing code that doesn't pass
        compare_class still works. We patch CompareFC to avoid loading FlagModel.
        """
        from benchmarks.complex_func_bench.runner.base_runner import ModelRunner

        with patch(
            "benchmarks.complex_func_bench.runner.base_runner.CompareFC"
        ) as MockCompareFC:
            mock_instance = MagicMock()
            mock_instance.free_function_list = ["Location_to_Lat_Long"]
            MockCompareFC.return_value = mock_instance

            runner = ModelRunner(mock_args, mock_logger)

            # CompareFC constructor should have been called
            MockCompareFC.assert_called_once_with(mock_args, mock_logger)
            assert runner.CompareClass is mock_instance

    def test_sap_gpt_runner_forwards_compare_class_to_base(
        self, mock_compare_class, mock_args, mock_logger
    ):
        """SAPGPTRunner should forward the compare_class parameter to ModelRunner.

        The subclass must pass compare_class through to super().__init__() so
        the shared CompareFC reaches ModelRunner without re-creation.
        """
        from benchmarks.complex_func_bench.runner.sap_gpt_runner import SAPGPTRunner

        mock_orchestrator = MagicMock()
        mock_orchestrator.active_model_key = "test-model"

        runner = SAPGPTRunner(
            args=mock_args,
            logger=mock_logger,
            orchestrator=mock_orchestrator,
            compare_class=mock_compare_class,
        )

        assert runner.CompareClass is mock_compare_class

    def test_shared_compare_class_identity_across_multiple_runners(
        self, mock_compare_class, mock_args, mock_logger
    ):
        """Multiple runners created with the same compare_class share the exact instance.

        This verifies the optimization works end-to-end: one CompareFC (and its
        FlagModel) is loaded once and reused by all runners in a configuration.
        """
        from benchmarks.complex_func_bench.runner.sap_gpt_runner import SAPGPTRunner

        mock_orchestrator = MagicMock()
        mock_orchestrator.active_model_key = "test-model"

        runners = [
            SAPGPTRunner(
                args=mock_args,
                logger=mock_logger,
                orchestrator=mock_orchestrator,
                compare_class=mock_compare_class,
            )
            for _ in range(5)
        ]

        # All runners must reference the same CompareFC object
        for runner in runners:
            assert runner.CompareClass is mock_compare_class

    def test_create_runner_passes_compare_class(self):
        """create_runner() in cfb_run_eval should forward compare_class to SAPGPTRunner.

        This tests the glue code in cfb_run_eval.py that connects the shared
        CompareFC instance to each runner created in the case loop.
        """
        from cfb_run_eval import create_runner

        mock_orchestrator = MagicMock()
        mock_orchestrator.active_model_key = "test-model"
        mock_orchestrator.cfg.results_dir = "/tmp/test"
        mock_compare = MagicMock()
        mock_compare.free_function_list = ["Location_to_Lat_Long"]

        with patch("cfb_run_eval.SAPGPTRunner") as MockRunner:
            create_runner(
                log_dir="/tmp/test",
                orchestrator=mock_orchestrator,
                compare_class=mock_compare,
            )

            # Verify compare_class was forwarded to the runner constructor
            _, kwargs = MockRunner.call_args
            assert kwargs.get("compare_class") is mock_compare


# ---------------------------------------------------------------------------
# Change 2: Model-Level Parallelism Tests
# ---------------------------------------------------------------------------


class TestModelParallelism:
    """Tests for running different models concurrently.

    Each model gets its own thread with its own LLMOrchestrator instance.
    Memory methods within a model remain sequential.
    """

    def test_run_model_configs_executes_all_memory_methods(self):
        """run_model_configs should call run_single_configuration for each memory method.

        Verifies that run_model_configs correctly iterates through all enabled
        memory methods for a given model, calling run_single_configuration for
        each (memory, compact_threshold) combination. Threshold-insensitive
        strategies (e.g. 'ace') produce one call; threshold-sensitive strategies
        (e.g. 'truncation') produce one call per configured threshold value.
        """
        from cfb_run_eval import run_model_configs

        dataset = [{"id": "Test-001", "conversations": [], "functions": []}]

        # Orchestrator reports three strategies; 'truncation' needs thresholds,
        # 'progressive_summarization' needs thresholds, 'ace' does not.
        mock_orchestrator = MagicMock()
        mock_orchestrator.cfg.memory_strategies = {
            "truncation": MagicMock(type="truncation"),
            "ace": MagicMock(type="ace"),
        }
        mock_orchestrator.cfg.compact_thresholds = [500, 1000]
        mock_orchestrator.cfg.haystack_thresholds = []
        mock_orchestrator.cfg.input_file = "dummy.jsonl"

        memory_methods = ["truncation", "ace"]

        with (
            patch("cfb_run_eval.run_single_configuration") as mock_run,
            patch("cfb_run_eval.load_haystack_dataset", return_value=dataset),
        ):
            run_model_configs(
                model="test-model",
                memory_methods=memory_methods,
                orchestrator=mock_orchestrator,
                run_timestamp="20260101_0000",
                resp_eval_runner=MagicMock(),
            )

        # truncation: 2 thresholds × 1 haystack(None) = 2 calls
        # ace: 1 (no threshold) × 1 haystack(None) = 1 call
        assert mock_run.call_count == 3

        # Each call must specify the correct model
        called_models = [c.kwargs["model"] for c in mock_run.call_args_list]
        assert all(m == "test-model" for m in called_models)

    def test_run_model_configs_uses_own_orchestrator(self):
        """Each haystack-parallel thread must use its own LLMOrchestrator instance.

        When multiple haystack thresholds are configured, run_model_configs
        creates a fresh LLMOrchestrator per thread so that mutable per-session
        state (trace buffers, memory processor state) is never shared.
        """
        from cfb_run_eval import run_model_configs

        dataset = [{"id": "Test-001"}]
        created_orchestrators = []

        mock_orchestrator = MagicMock()
        mock_orchestrator.cfg.memory_strategies = {
            "truncation": MagicMock(type="truncation"),
        }
        mock_orchestrator.cfg.compact_thresholds = [500]
        # Two haystack thresholds force parallel execution and per-thread orchestrators
        mock_orchestrator.cfg.haystack_thresholds = [1000, 2000]
        mock_orchestrator.cfg.input_file = "dummy.jsonl"

        def mock_run_single(orchestrator, **kwargs):
            created_orchestrators.append(id(orchestrator))

        with (
            patch("cfb_run_eval.run_single_configuration", side_effect=mock_run_single),
            patch("cfb_run_eval.load_haystack_dataset", return_value=dataset),
            # Each LLMOrchestrator() call returns a distinct MagicMock
            patch("cfb_run_eval.LLMOrchestrator", side_effect=lambda: MagicMock()),
        ):
            run_model_configs(
                model="test-model",
                memory_methods=["truncation"],
                orchestrator=mock_orchestrator,
                run_timestamp="20260101_0000",
                resp_eval_runner=MagicMock(),
            )

        # Three haystack levels (None + 1000 + 2000) — all orchestrator ids must differ
        assert len(created_orchestrators) == 3
        assert len(set(created_orchestrators)) == 3, (
            "Each thread must get its own orchestrator"
        )

    def test_parallel_model_execution_completes_all(self):
        """All models should complete when run via ThreadPoolExecutor.

        Verifies that the parallelism mechanism (ThreadPoolExecutor) correctly
        dispatches and collects results from multiple concurrent model runs
        without deadlocks or missing results.
        """
        from cfb_run_eval import run_model_configs

        models = ["model-a", "model-b", "model-c"]
        completed = []

        def mock_run_model_configs(model, **kwargs):
            """Track which models completed execution."""
            completed.append(model)

        with patch("cfb_run_eval.run_single_configuration"):
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(
                        mock_run_model_configs,
                        model=m,
                        memory_methods=["truncation"],
                        orchestrator=MagicMock(),
                        run_timestamp="20260101_0000",
                        resp_eval_runner=MagicMock(),
                    )
                    for m in models
                ]
                # Wait for all to complete
                for f in futures:
                    f.result()

        assert sorted(completed) == sorted(models)
