"""Tests for experiment-specific config loading in cfb_run_eval.

These tests verify that the evaluation entrypoint can select a concrete
experiment config from the configs directory, preserve the exact TOML files
used for a run, and propagate the already-loaded config into helper classes
that would otherwise fall back to default file locations.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch


def test_resolve_config_paths_uses_configs_directory(tmp_path):
    """
    resolve_config_paths() must map an experiment selector to the matching
    configs/<selector>_config.toml file and the shared configs/model_config.toml.

    This keeps each run bound to an explicit experiment config instead of the
    mutable top-level config.toml file.
    """
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    experiment_path = config_dir / "demo_run_config.toml"
    model_path = config_dir / "model_config.toml"
    experiment_path.write_text('experiment_name = "demo_run"\n', encoding="utf-8")
    model_path.write_text("[models.demo]\nprovider = \"test\"\n", encoding="utf-8")

    import cfb_run_eval

    resolved_experiment, resolved_model = cfb_run_eval.resolve_config_paths(
        "demo_run",
        config_dir=config_dir,
    )

    assert resolved_experiment == experiment_path
    assert resolved_model == model_path


def test_build_log_path_includes_experiment_name():
    """
    build_log_path() must include the configured experiment name so concurrent
    runs with different experiments produce distinct top-level log files.
    """
    import cfb_run_eval

    log_path = cfb_run_eval.build_log_path(
        logs_dir=Path("logs"),
        experiment_name="full_haystack_run_claude",
        run_timestamp="20260312_1042",
    )

    assert log_path == Path(
        "logs/experiment_run_full_haystack_run_claude_20260312_1042.log"
    )


def test_save_config_snapshot_copies_exact_toml_inputs(tmp_path):
    """
    save_config_snapshot() must copy the exact experiment and model TOML files
    into the run directory so later analysis can recover the precise settings
    that produced a result set.
    """
    experiment_config = tmp_path / "source_experiment.toml"
    model_config = tmp_path / "source_model.toml"
    run_dir = tmp_path / "results" / "cfb" / "demo" / "20260312_1042"

    experiment_config.write_text('experiment_name = "demo"\n', encoding="utf-8")
    model_config.write_text("[models.demo]\nprovider = \"test\"\n", encoding="utf-8")

    import cfb_run_eval

    cfb_run_eval.save_config_snapshot(run_dir, experiment_config, model_config)

    snapshot_dir = run_dir / "used_configs"
    assert (snapshot_dir / experiment_config.name).read_text(encoding="utf-8") == (
        experiment_config.read_text(encoding="utf-8")
    )
    assert (snapshot_dir / model_config.name).read_text(encoding="utf-8") == (
        model_config.read_text(encoding="utf-8")
    )


def test_response_evaluator_uses_supplied_config():
    """
    RespEvalRunner must construct its internal orchestrator from the already
    loaded ExperimentConfig when one is supplied on args, ensuring response
    evaluation uses the same config files as the main benchmark run.
    """
    from benchmarks.complex_func_bench.runner.response_runner import RespEvalRunner

    config = object()
    args = SimpleNamespace(log_dir="unused", config=config)

    with (
        patch(
            "benchmarks.complex_func_bench.runner.response_runner.LLMOrchestrator"
        ) as mock_orchestrator,
        patch("benchmarks.complex_func_bench.runner.response_runner.SAPGPTModel"),
    ):
        RespEvalRunner(args=args, logger=MagicMock())

    mock_orchestrator.assert_called_once_with(config=config)


def test_compare_fc_uses_supplied_config():
    """
    CompareFC must also reuse the already-loaded ExperimentConfig when it builds
    its helper SAPGPTModel, otherwise comparison calls would fall back to the
    default config locations and ignore the experiment-specific TOMLs.
    """
    from benchmarks.complex_func_bench.utils.compare_method import CompareFC

    config = object()
    args = SimpleNamespace(log_dir="unused", config=config)
    tool_info_json = '{"booking-com15": {}}'

    with (
        patch("benchmarks.complex_func_bench.utils.compare_method.FlagModel"),
        patch("benchmarks.complex_func_bench.utils.compare_method.RapidAPICall"),
        patch("benchmarks.complex_func_bench.utils.compare_method.SAPGPTModel"),
        patch(
            "benchmarks.complex_func_bench.utils.compare_method.LLMOrchestrator"
        ) as mock_orchestrator,
        patch(
            "benchmarks.complex_func_bench.utils.compare_method.load_json",
            return_value={},
        ),
        patch("builtins.open", mock_open(read_data=tool_info_json)),
    ):
        CompareFC(args=args, logger=MagicMock())

    mock_orchestrator.assert_called_once_with(config=config)