"""Tests for experiment-specific config loading in cfb_run_eval.

These tests verify that the evaluation entrypoint can select a concrete
experiment config from the configs directory, preserve the exact TOML files
used for a run, and propagate the already-loaded config into helper classes
that would otherwise fall back to default file locations.

They also guard the evaluation model isolation fix: when benchmarking a
non-OpenAI model (e.g. qwen35, glm-4-7), all judge/evaluation LLM calls must
be routed through the configured evaluation_model (default: gpt-4-1-mini), not
through the active benchmarked model whose output format may be incompatible
with the structured-JSON evaluation prompts.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

from memorch.utils.config import ExperimentConfig, ModelDef


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
    model_path.write_text('[models.demo]\nprovider = "test"\n', encoding="utf-8")

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
    model_config.write_text('[models.demo]\nprovider = "test"\n', encoding="utf-8")

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

    mock_orchestrator.assert_called_once_with(config=config)  # type: ignore[union-attr]


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


# ---------------------------------------------------------------------------
# Evaluation model isolation tests
# ---------------------------------------------------------------------------


def _make_two_model_config(benchmarked: str = "qwen35") -> ExperimentConfig:
    """Build a minimal ExperimentConfig with a non-OpenAI benchmarked model and
    gpt-4-1-mini in the registry as the evaluation model."""
    return ExperimentConfig(
        experiment_name="test",
        results_dir="results",
        log_dir="logs",
        logging_level="INFO",
        input_file="data.jsonl",
        enabled_models=[benchmarked],
        enabled_memory_methods=["no_strategy"],
        compact_thresholds=[5000],
        memory_strategies={"no_strategy": {"type": "no_strategy"}},
        evaluation_model="gpt-4-1-mini",
        model_registry={
            benchmarked: ModelDef(
                litellm_name=f"openai/{benchmarked}",
                context_window=128_000,
                provider="ollama",
            ),
            "gpt-4-1-mini": ModelDef(
                litellm_name="sap/gpt-4.1-mini",
                context_window=128_000,
                provider="aicore",
            ),
        },
    )


def test_generate_plain_uses_evaluation_model_when_benchmarking_non_openai_model():
    """
    When the benchmarked model is a non-OpenAI model (e.g. qwen35, glm-4-7),
    LLMOrchestrator.generate_plain() must route evaluation calls through the
    configured evaluation_model ('gpt-4-1-mini'), not through the active
    benchmarked model.

    This is the core regression guard for the bug where evaluation of non-OpenAI
    models would fail because their output format is incompatible with the
    structured-JSON schemas expected by the completeness/correctness scoring and
    LLM-based function call comparison prompts.
    """
    from memorch.llm_orchestrator import LLMOrchestrator

    config = _make_two_model_config(benchmarked="qwen35")

    with (
        patch("memorch.llm_orchestrator.MemoryProcessor"),
        patch("memorch.llm_orchestrator.litellm.completion") as mock_completion,
    ):
        mock_completion.return_value = MagicMock()
        orch = LLMOrchestrator(config=config)
        # active_model_key is the benchmarked model after initialization
        assert orch.active_model_key == "qwen35"

        orch.generate_plain(input_messages=[{"role": "user", "content": "score this"}])

    called_model = mock_completion.call_args.kwargs.get("model")
    assert called_model == "sap/gpt-4.1-mini", (
        f"generate_plain() must use evaluation_model ('sap/gpt-4.1-mini'), "
        f"but called litellm with model='{called_model}'. "
        "Evaluation must not be routed through the benchmarked model."
    )


def test_experiment_configs_declare_evaluation_model():
    """
    All experiment config TOML files in the configs/ directory must declare an
    explicit evaluation_model key.  This makes the evaluation model visible and
    auditable for each run rather than relying on a hidden default.

    Configs that are example/template files (ending with .example.toml) are
    included in this check so templates stay up-to-date with the schema.
    """
    import tomllib
    from pathlib import Path

    configs_dir = Path("configs")
    toml_files = list(configs_dir.glob("*.toml"))
    # Exclude the model registry — it has no evaluation_model key by design
    experiment_configs = [f for f in toml_files if f.name != "model_config.toml"]

    assert experiment_configs, "Expected at least one experiment config in configs/"

    missing = []
    for toml_path in experiment_configs:
        with open(toml_path, "rb") as f:
            data = tomllib.load(f)
        if "evaluation_model" not in data:
            missing.append(toml_path.name)

    assert not missing, (
        f"The following experiment configs are missing 'evaluation_model': {missing}. "
        "Add 'evaluation_model = \"gpt-4-1-mini\"' to each file so evaluation is "
        "always routed through a reliable judge model."
    )
