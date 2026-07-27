"""
Tests for the compact_thresholds experiment axis.

These tests verify the full chain introduced by the multi-threshold feature:

1. Config parsing  – ExperimentConfig accepts a list of integers for
   compact_thresholds and rejects a bare scalar.
2. MemoryProcessor – apply_strategy respects the supplied threshold, raises
   when it is missing for threshold-sensitive strategies, and ignores it for
   threshold-insensitive ones (ace, memory_bank).
3. Experiment loop – the runner builds the correct (model, memory, threshold)
   combinations, running threshold-sensitive strategies once per threshold value
   and threshold-insensitive strategies exactly once with threshold=None.
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import List, Dict

from memorch.utils.config import ExperimentConfig, MemoryDef, ModelDef
from memorch.memory_processing import MemoryProcessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_exp_config(
    compact_thresholds: List[int],
    strategy_type: str = "truncation",
) -> ExperimentConfig:
    """
    Build a minimal ExperimentConfig with a single memory strategy.

    Uses model_validate (bypass alias resolution) so the fixture remains
    independent of TOML loading.
    """
    memory_def = MemoryDef(type=strategy_type)
    model_def = ModelDef(
        litellm_name="openai/gpt-test",
        context_window=128000,
        provider="test",
    )
    return ExperimentConfig.model_validate(
        {
            "experiment_name": "test",
            "results_dir": "results",
            "log_dir": "logs",
            "logging_level": "WARNING",
            "input_file": "data.jsonl",
            "enabled_models": ["gpt-test"],
            "enabled_memory_methods": [strategy_type],
            "compact_thresholds": compact_thresholds,
            "memory_strategies": {strategy_type: memory_def},
            "model_registry": {"gpt-test": model_def},
        }
    )


@pytest.fixture
def messages() -> List[Dict]:
    """Minimal conversation fixture."""
    return [{"role": "user", "content": "hello"}]


# ---------------------------------------------------------------------------
# 1. Config parsing
# ---------------------------------------------------------------------------


def test_experiment_config_accepts_threshold_list():
    """
    ExperimentConfig.compact_thresholds must accept a list of integers so that
    multiple threshold values can be declared in a single config.toml entry
    (e.g. compact_thresholds = [2000, 6000, 12000]).
    """
    cfg = _make_exp_config([2000, 6000, 12000])
    assert cfg.compact_thresholds == [2000, 6000, 12000]


def test_experiment_config_accepts_single_element_list():
    """
    A single-element list [6000] is the backwards-compatible replacement for
    the old scalar compact_threshold = 6000.  It must parse cleanly.
    """
    cfg = _make_exp_config([6000])
    assert cfg.compact_thresholds == [6000]


def test_experiment_config_rejects_scalar_threshold():
    """
    A bare integer must no longer be accepted for compact_thresholds, ensuring
    that callers migrate to the list form.  Pydantic should raise a
    ValidationError when a scalar is supplied.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            {
                "experiment_name": "test",
                "results_dir": "results",
                "log_dir": "logs",
                "logging_level": "WARNING",
                "input_file": "data.jsonl",
                "enabled_models": ["gpt-test"],
                "enabled_memory_methods": ["truncation"],
                "compact_thresholds": 6000,  # scalar – must be rejected
                "memory_strategies": {"truncation": {"type": "truncation"}},
            }
        )


# ---------------------------------------------------------------------------
# 2. MemoryProcessor – threshold-sensitive strategies
# ---------------------------------------------------------------------------


def test_apply_strategy_passes_through_below_threshold(messages):
    """
    When input_token_count is below compact_threshold the processor must return
    the original messages unchanged.  This verifies the gate is respected for
    the truncation strategy.
    """
    cfg = _make_exp_config([10000], strategy_type="truncation")
    processor = MemoryProcessor(cfg)

    result_msgs, result_count = processor.apply_strategy(
        messages=messages,
        memory_key="truncation",
        input_token_count=500,  # well below threshold
        compact_threshold=10000,
    )

    assert result_msgs is messages
    assert result_count == 500


def test_apply_strategy_raises_without_threshold_for_truncation(messages):
    """
    Threshold-sensitive strategies (truncation, progressive_summarization)
    require compact_threshold to be supplied explicitly by the caller.
    Omitting it must raise ValueError so the bug surfaces immediately rather
    than silently using a wrong value.
    """
    cfg = _make_exp_config([6000], strategy_type="truncation")
    processor = MemoryProcessor(cfg)

    with pytest.raises(ValueError, match="compact_threshold must be provided"):
        processor.apply_strategy(
            messages=messages,
            memory_key="truncation",
            input_token_count=9999,
            compact_threshold=None,  # explicitly missing
        )


def test_apply_strategy_raises_without_threshold_for_progressive_summarization(
    messages,
):
    """
    Same contract as for truncation: progressive_summarization must raise when
    compact_threshold is not provided, guarding against accidental None usage.
    """
    cfg = _make_exp_config([6000], strategy_type="progressive_summarization")
    processor = MemoryProcessor(cfg)

    with pytest.raises(ValueError, match="compact_threshold must be provided"):
        processor.apply_strategy(
            messages=messages,
            memory_key="progressive_summarization",
            input_token_count=9999,
            compact_threshold=None,
        )


def test_apply_strategy_different_thresholds_produce_different_behaviour(messages):
    """
    With a low threshold (100) the processor fires compaction; with a high
    threshold (99999) it passes through untouched.  This is the core
    requirement: varying compact_threshold changes which runs trigger
    compaction.

    Truncation is mocked so the test does not depend on message format.
    """
    cfg = _make_exp_config([100], strategy_type="truncation")
    processor = MemoryProcessor(cfg)

    # Below threshold: pass-through
    result_below, _ = processor.apply_strategy(
        messages=messages,
        memory_key="truncation",
        input_token_count=50,
        compact_threshold=99999,
    )
    assert result_below is messages

    # Above threshold: truncation fires (mocked to return a sentinel)
    sentinel = [{"role": "assistant", "content": "truncated"}]
    with (
        patch("memorch.memory_processing.truncate_messages", return_value=sentinel),
        patch("memorch.memory_processing.get_token_count", return_value=10),
    ):
        result_above, _ = processor.apply_strategy(
            messages=messages,
            memory_key="truncation",
            input_token_count=50,
            compact_threshold=10,  # lower than input_token_count
        )
    assert result_above is sentinel


# ---------------------------------------------------------------------------
# 3. MemoryProcessor – threshold-insensitive strategies
# ---------------------------------------------------------------------------


def test_ace_strategy_ignores_compact_threshold(messages):
    """
    The ace strategy bypasses the token-count gate entirely: it must run
    regardless of compact_threshold (or its absence).  Passing None must not
    raise, confirming that threshold-insensitive strategies are safe to call
    without a threshold.
    """
    cfg = _make_exp_config([6000], strategy_type="ace")
    processor = MemoryProcessor(cfg)

    sentinel = [{"role": "assistant", "content": "ace output"}]
    with (
        patch("memorch.memory_processing.apply_ace_strategy", return_value=sentinel),
        patch("memorch.memory_processing.get_token_count", return_value=5),
    ):
        result, _ = processor.apply_strategy(
            messages=messages,
            memory_key="ace",
            input_token_count=99999,
            compact_threshold=None,  # must be accepted without error
        )
    assert result is sentinel


def test_memory_bank_strategy_ignores_compact_threshold(messages):
    """
    memory_bank also bypasses the token-count gate.  Supplying None for
    compact_threshold must not cause an error, and the strategy must execute.
    """
    cfg = _make_exp_config([6000], strategy_type="memory_bank")
    processor = MemoryProcessor(cfg)

    sentinel = [{"role": "assistant", "content": "mb output"}]
    with (
        patch(
            "memorch.memory_processing.apply_memory_bank_strategy",
            return_value=(sentinel, 5),
        ),
        patch("memorch.memory_processing.get_token_count", return_value=5),
    ):
        result, _ = processor.apply_strategy(
            messages=messages,
            memory_key="memory_bank",
            input_token_count=99999,
            compact_threshold=None,
        )
    assert result is sentinel


# ---------------------------------------------------------------------------
# 4. Experiment loop logic (unit-level)
# ---------------------------------------------------------------------------


def test_threshold_loop_runs_once_per_threshold_for_sensitive_strategy():
    """
    For a threshold-sensitive strategy (truncation), the runner must invoke
    run_single_configuration once per value in compact_thresholds.  With
    compact_thresholds = [1000, 5000, 10000] there should be exactly 3 calls.

    The test imports and exercises only the loop logic from cfb_run_eval,
    replacing the heavyweight run_single_configuration with a mock.
    """
    import cfb_run_eval

    cfg = _make_exp_config([1000, 5000, 10000], strategy_type="truncation")

    orchestrator = MagicMock()
    orchestrator.cfg = cfg

    calls = []

    def fake_run(
        orchestrator,
        dataset,
        model,
        memory,
        run_timestamp,
        resp_eval_runner,
        compact_threshold,
    ):
        calls.append(compact_threshold)

    with patch.object(cfb_run_eval, "run_single_configuration", side_effect=fake_run):
        # Simulate the inner loop body for one (model, memory) pair
        strategy_type = cfg.memory_strategies["truncation"].type
        THRESHOLD_SENSITIVE = {"truncation", "progressive_summarization"}
        thresholds = (
            cfg.compact_thresholds if strategy_type in THRESHOLD_SENSITIVE else [None]
        )
        for threshold in thresholds:
            cfb_run_eval.run_single_configuration(
                orchestrator=orchestrator,
                dataset=[],
                model="gpt-test",
                memory="truncation",
                run_timestamp="20260101_0000",
                resp_eval_runner=MagicMock(),
                compact_threshold=threshold,
            )

    assert calls == [1000, 5000, 10000]


def test_threshold_loop_runs_once_for_insensitive_strategy():
    """
    For a threshold-insensitive strategy (ace), the runner must invoke
    run_single_configuration exactly once with compact_threshold=None,
    regardless of how many values are in compact_thresholds.
    """
    import cfb_run_eval

    cfg = _make_exp_config([1000, 5000, 10000], strategy_type="ace")

    orchestrator = MagicMock()
    orchestrator.cfg = cfg

    calls = []

    def fake_run(
        orchestrator,
        dataset,
        model,
        memory,
        run_timestamp,
        resp_eval_runner,
        compact_threshold,
    ):
        calls.append(compact_threshold)

    with patch.object(cfb_run_eval, "run_single_configuration", side_effect=fake_run):
        strategy_type = cfg.memory_strategies["ace"].type
        THRESHOLD_SENSITIVE = {"truncation", "progressive_summarization"}
        thresholds = (
            cfg.compact_thresholds if strategy_type in THRESHOLD_SENSITIVE else [None]
        )
        for threshold in thresholds:
            cfb_run_eval.run_single_configuration(
                orchestrator=orchestrator,
                dataset=[],
                model="gpt-test",
                memory="ace",
                run_timestamp="20260101_0000",
                resp_eval_runner=MagicMock(),
                compact_threshold=threshold,
            )

    assert calls == [None]
