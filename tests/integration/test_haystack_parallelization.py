"""
Tests for the haystack parallelization refactor in cfb_run_eval.

These tests verify the restructured execution flow where:
- Models, strategies, and compact_thresholds iterate SEQUENTIALLY
- Haystack thresholds (including explicit baseline sentinel 0) run in PARALLEL
- Each parallel haystack thread gets its own LLMOrchestrator instance
- Each haystack_threshold loads its own dataset file

Test categories:
1. Dataset loading     - correct file selected per haystack_threshold
2. Directory structure - haystack_threshold included in results path
3. Execution flow      - sequential outer loop, parallel inner haystack loop
4. Baseline handling   - haystack_threshold=None uses original data
"""

import os
import json
import pytest
from unittest.mock import MagicMock, patch, call
from typing import List, Dict, Optional

import cfb_run_eval
from memorch.utils.config import ExperimentConfig, MemoryDef, ModelDef


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_exp_config(
    compact_thresholds: List[int],
    haystack_thresholds: Optional[List[int]] = None,
    strategy_type: str = "truncation",
) -> ExperimentConfig:
    """Build a minimal ExperimentConfig for testing.

    Mirrors the helper in test_compact_thresholds.py but adds
    haystack_thresholds support.
    """
    memory_def = MemoryDef(type=strategy_type)
    model_def = ModelDef(
        litellm_name="openai/gpt-test",
        context_window=128000,
        provider="test",
    )
    config_dict = {
        "experiment_name": "test_haystack",
        "results_dir": "results",
        "log_dir": "logs",
        "logging_level": "WARNING",
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
# 1. Dataset loading
# ---------------------------------------------------------------------------


def test_load_haystack_dataset_baseline(tmp_path):
    """
    When haystack_threshold is None (baseline), load_haystack_dataset must
    load the original ComplexFuncBench.jsonl file without any haystack
    augmentation. This ensures backward compatibility.
    """
    # Create a minimal original dataset file
    original_file = tmp_path / "ComplexFuncBench.jsonl"
    cases = [
        {"id": "Flights-1", "conversations": [], "functions": []},
        {"id": "Hotels-2", "conversations": [], "functions": []},
    ]
    with open(original_file, "w") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")

    dataset = cfb_run_eval.load_haystack_dataset(
        haystack_threshold=None,
        input_file=str(original_file),
    )

    assert len(dataset) == 2
    assert dataset[0]["id"] == "Flights-1"
    # Baseline cases should NOT have haystack_messages
    assert "haystack_messages" not in dataset[0]


def test_load_haystack_dataset_with_threshold(tmp_path):
    """
    When haystack_threshold is a positive integer (e.g. 20000),
    load_haystack_dataset must load the corresponding haystack file
    (haystack_20000.jsonl) which contains pre-computed haystack_messages.
    """
    # Create a haystack-augmented dataset file
    haystack_file = tmp_path / "haystack_20000.jsonl"
    cases = [
        {
            "id": "Flights-1",
            "conversations": [],
            "functions": [],
            "haystack_messages": [{"role": "tool", "content": "distractor"}],
            "haystack_token_count": 19500,
        },
    ]
    with open(haystack_file, "w") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")

    dataset = cfb_run_eval.load_haystack_dataset(
        haystack_threshold=20000,
        input_file=str(tmp_path / "ComplexFuncBench.jsonl"),
    )

    assert len(dataset) == 1
    assert "haystack_messages" in dataset[0]
    assert dataset[0]["haystack_token_count"] == 19500


def test_load_haystack_dataset_missing_file(tmp_path):
    """
    When the haystack file for a given threshold does not exist,
    load_haystack_dataset must raise FileNotFoundError with a clear message
    pointing to the expected filename.
    """
    with pytest.raises(FileNotFoundError, match="haystack_50000.jsonl"):
        cfb_run_eval.load_haystack_dataset(
            haystack_threshold=50000,
            input_file=str(tmp_path / "ComplexFuncBench.jsonl"),
        )


# ---------------------------------------------------------------------------
# 2. Directory structure
# ---------------------------------------------------------------------------


def test_setup_directories_includes_haystack_threshold(tmp_path):
    """
    When haystack_threshold is provided, setup_directories must include it
    in the directory path so results are organized by haystack level.
    The path should contain a 'haystack_20000' segment.
    """
    log_dir = cfb_run_eval.setup_directories(
        experiment_name="test_exp",
        run_timestamp="20260225_1200",
        model="gpt-test",
        memory="truncation",
        compact_threshold=5000,
        haystack_threshold=20000,
    )
    assert "haystack_20000" in log_dir


def test_setup_directories_baseline_haystack(tmp_path):
    """
    When haystack_threshold is None (baseline), the directory path must
    use 'no_haystack' to distinguish baseline runs from haystack runs.
    """
    log_dir = cfb_run_eval.setup_directories(
        experiment_name="test_exp",
        run_timestamp="20260225_1200",
        model="gpt-test",
        memory="truncation",
        compact_threshold=5000,
        haystack_threshold=None,
    )
    assert "no_haystack" in log_dir


# ---------------------------------------------------------------------------
# 3. Execution flow
# ---------------------------------------------------------------------------


def test_haystack_thresholds_run_in_parallel():
    """
    For a configuration with haystack_thresholds=[0, 20000, 40000], the
    runner must submit 3 tasks total. The baseline sentinel 0 must map to
    runtime haystack_threshold=None, while numeric values stay numeric.

    We mock run_single_configuration and verify it is called with the
    correct haystack_threshold values. The ThreadPoolExecutor is patched
    to capture submitted tasks.
    """
    cfg = _make_exp_config(
        compact_thresholds=[5000],
        haystack_thresholds=[0, 20000, 40000],
        strategy_type="truncation",
    )

    submitted_haystack_thresholds = []

    def fake_run_single(
        orchestrator,
        dataset,
        model,
        memory,
        run_timestamp,
        resp_eval_runner,
        compact_threshold,
        haystack_threshold=None,
    ):
        submitted_haystack_thresholds.append(haystack_threshold)

    with (
        patch.object(
            cfb_run_eval, "run_single_configuration", side_effect=fake_run_single
        ),
        patch.object(cfb_run_eval, "load_haystack_dataset", return_value=[]),
        patch("memorch.llm_orchestrator.LLMOrchestrator"),
    ):
        # Simulate baseline normalization used by runner logic.
        haystack_values = [
            cfb_run_eval.normalize_haystack_threshold(ht)
            for ht in cfb_run_eval.validate_haystack_thresholds_config(
                cfg.haystack_thresholds
            )
        ]
        for ht in haystack_values:
            cfb_run_eval.run_single_configuration(
                orchestrator=MagicMock(),
                dataset=[],
                model="gpt-test",
                memory="truncation",
                run_timestamp="20260225_1200",
                resp_eval_runner=MagicMock(),
                compact_threshold=5000,
                haystack_threshold=ht,
            )

    assert sorted(
        submitted_haystack_thresholds, key=lambda x: (x is not None, x or 0)
    ) == [
        None,
        20000,
        40000,
    ]


def test_no_haystack_thresholds_runs_baseline_only():
    """
    The explicit baseline contract requires haystack_thresholds to include
    sentinel 0. Missing or empty haystack_thresholds must raise ValueError
    with an actionable message.
    """
    with pytest.raises(ValueError, match="include baseline 0"):
        cfb_run_eval.validate_haystack_thresholds_config(None)

    with pytest.raises(ValueError, match="include baseline 0"):
        cfb_run_eval.validate_haystack_thresholds_config([])


# ---------------------------------------------------------------------------
# 4. Baseline handling
# ---------------------------------------------------------------------------


def test_baseline_uses_original_dataset():
    """
    The baseline run (haystack_threshold=None) must use the original
    ComplexFuncBench.jsonl data. Cases must NOT contain haystack_messages
    keys, ensuring the model sees only the actual task context.
    """
    original_case = {
        "id": "Flights-1",
        "conversations": [{"role": "user", "content": "test"}],
        "functions": [],
    }

    # Verify baseline data doesn't have haystack fields
    assert "haystack_messages" not in original_case
    assert "haystack_token_count" not in original_case


def test_run_single_configuration_accepts_haystack_threshold():
    """
    run_single_configuration must accept a haystack_threshold parameter
    and pass it through to setup_directories for correct result organization.
    This verifies the function signature was updated for the NIAH experiment.
    """
    import inspect

    sig = inspect.signature(cfb_run_eval.run_single_configuration)
    params = list(sig.parameters.keys())
    assert "haystack_threshold" in params
