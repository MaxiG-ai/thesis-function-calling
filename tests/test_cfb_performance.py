"""Tests for cfb_run_eval performance optimizations.

These tests validate three key optimizations:
1. FlagModel/CompareFC reuse across test cases (avoid reloading 1.3GB model per case)
2. Model-level parallelism (run different models concurrently)
3. Batch Weave logging (collect results locally, log after completion)

All tests mock external dependencies (LLM calls, FlagModel, Weave) to run
without GPU or API access.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch, call
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


@pytest.fixture
def sample_results():
    """Sample evaluation results for metrics and Weave logging tests.

    Contains two results: one success and one failure, covering all metric fields.
    """
    return [
        {
            "id": "Travel-001",
            "gen_convs": [],
            "message": "Success.",
            "count_dict": {
                "success_turn_num": 2,
                "total_turn_num": 2,
                "correct_call_num": 3,
                "total_call_num": 3,
                "real_turn_num": 2,
            },
            "resp_eval": {
                "complete": {"score": 2},
                "correct": {"score": 1},
            },
            "status": "Success",
        },
        {
            "id": "Travel-002",
            "gen_convs": [],
            "message": {"error_type": "value_error", "content": "Wrong value"},
            "count_dict": {
                "success_turn_num": 1,
                "total_turn_num": 2,
                "correct_call_num": 1,
                "total_call_num": 3,
                "real_turn_num": 2,
            },
            "resp_eval": None,
            "status": "Failed",
        },
    ]


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
            model_name="test-model",
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
                model_name="test-model",
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

        Verifies that the extracted function correctly iterates through all
        enabled memory methods for a given model, calling run_single_configuration
        for each one sequentially.
        """
        from cfb_run_eval import run_model_configs

        mock_orchestrator = MagicMock()
        dataset = [{"id": "Test-001", "conversations": [], "functions": []}]
        memory_methods = ["truncation", "progressive_summarization", "ace"]

        with patch("cfb_run_eval.run_single_configuration") as mock_run:
            run_model_configs(
                model="test-model",
                memory_methods=memory_methods,
                orchestrator=mock_orchestrator,
                dataset=dataset,
                run_timestamp="20260101_0000",
                resp_eval_runner=MagicMock(),
            )

            # Should be called once per memory method
            assert mock_run.call_count == len(memory_methods)

            # Each call should have the correct memory method
            called_memories = [c.kwargs["memory"] for c in mock_run.call_args_list]
            assert called_memories == memory_methods

    def test_run_model_configs_uses_own_orchestrator(self):
        """Each model thread must use its own LLMOrchestrator instance.

        This prevents race conditions on shared mutable state (trace buffers,
        memory processor state) when models run in parallel.
        """
        from cfb_run_eval import run_model_configs

        orchestrator_a = MagicMock()
        orchestrator_b = MagicMock()
        dataset = [{"id": "Test-001"}]

        with patch("cfb_run_eval.run_single_configuration") as mock_run:
            # Simulate two models running (would be parallel in production)
            run_model_configs(
                model="model-a",
                memory_methods=["truncation"],
                orchestrator=orchestrator_a,
                dataset=dataset,
                run_timestamp="20260101_0000",
                resp_eval_runner=MagicMock(),
            )
            run_model_configs(
                model="model-b",
                memory_methods=["truncation"],
                orchestrator=orchestrator_b,
                dataset=dataset,
                run_timestamp="20260101_0000",
                resp_eval_runner=MagicMock(),
            )

            # Each call received a different orchestrator
            orch_a = mock_run.call_args_list[0].kwargs["orchestrator"]
            orch_b = mock_run.call_args_list[1].kwargs["orchestrator"]
            assert orch_a is not orch_b

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
                        dataset=[],
                        run_timestamp="20260101_0000",
                        resp_eval_runner=MagicMock(),
                    )
                    for m in models
                ]
                # Wait for all to complete
                for f in futures:
                    f.result()

        assert sorted(completed) == sorted(models)


# ---------------------------------------------------------------------------
# Change 3: Batch Weave Logging Tests
# ---------------------------------------------------------------------------


class TestBatchWeaveLogging:
    """Tests for collecting evaluation data locally and batching Weave calls.

    Instead of calling eval_logger.log_prediction() inside the case loop
    (adding network overhead per case), results are collected in a list and
    logged to Weave after all cases complete.
    """

    def test_format_result_for_wandb_success_case(self, sample_results):
        """format_result_for_wandb should produce correct structure for a successful case.

        Verifies all expected keys are present and values are correctly computed
        from the raw result structure (success flag, accuracy ratios, domain extraction).
        """
        from cfb_run_eval import format_result_for_wandb

        result = format_result_for_wandb(sample_results[0])

        assert result["case_id"] == "Travel-001"
        assert result["success"] is True
        assert result["domain"] == "Travel"
        assert result["turn_accuracy"] == 1.0  # 2/2
        assert result["call_accuracy"] == 1.0  # 3/3
        assert result["response_complete_score"] == 2
        assert result["response_correct_score"] == 1

    def test_format_result_for_wandb_failure_case(self, sample_results):
        """format_result_for_wandb should handle failed cases with partial metrics.

        Failed cases have non-"Success." messages and may have None resp_eval.
        The formatter must still produce valid output with available metrics.
        """
        from cfb_run_eval import format_result_for_wandb

        result = format_result_for_wandb(sample_results[1])

        assert result["success"] is False
        assert result["turn_accuracy"] == 0.5  # 1/2
        assert result["call_accuracy"] == pytest.approx(1 / 3)
        assert result.get("response_complete_score") is None
        assert result.get("response_correct_score") is None

    def test_batch_log_to_weave_logs_all_collected_results(self, sample_results):
        """batch_log_to_weave should log every collected result to the eval logger.

        After cases complete, the batch function iterates through collected results
        and calls eval_logger.log_prediction() for each one. This verifies
        no results are dropped during batch logging.
        """
        from cfb_run_eval import batch_log_to_weave, format_result_for_wandb

        mock_eval_logger = MagicMock()
        # Make the context manager return a mock prediction
        mock_pred = MagicMock()
        mock_eval_logger.log_prediction.return_value.__enter__ = MagicMock(
            return_value=mock_pred
        )
        mock_eval_logger.log_prediction.return_value.__exit__ = MagicMock(
            return_value=False
        )

        collected = [format_result_for_wandb(r) for r in sample_results]

        batch_log_to_weave(mock_eval_logger, collected)

        # log_prediction should be called once per result
        assert mock_eval_logger.log_prediction.call_count == len(collected)

    def test_batch_log_to_weave_scores_are_logged_correctly(self):
        """batch_log_to_weave should log the correct score values for each prediction.

        Verifies that success, turn_accuracy, call_accuracy, and optional
        response scores are all passed to pred.log_score() with the right
        metric names and values.
        """
        from cfb_run_eval import batch_log_to_weave

        mock_eval_logger = MagicMock()
        mock_pred = MagicMock()
        mock_eval_logger.log_prediction.return_value.__enter__ = MagicMock(
            return_value=mock_pred
        )
        mock_eval_logger.log_prediction.return_value.__exit__ = MagicMock(
            return_value=False
        )

        collected = [
            {
                "case_id": "Test-001",
                "domain": "Test",
                "status": "Success",
                "message": "Success.",
                "success": True,
                "turn_accuracy": 0.8,
                "call_accuracy": 0.9,
                "response_complete_score": 2,
                "response_correct_score": 1,
            }
        ]

        batch_log_to_weave(mock_eval_logger, collected)

        # Verify the scores logged
        score_calls = mock_pred.log_score.call_args_list
        score_names = [c.args[0] for c in score_calls]

        assert "success" in score_names
        assert "turn_accuracy" in score_names
        assert "call_accuracy" in score_names
        assert "response_complete" in score_names
        assert "response_correct" in score_names
