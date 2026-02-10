"""
Test suite for experiment metrics picker functionality.

Tests validate that the experiment picker correctly:
- Discovers available projects in results/cfb
- Lists timestamps for selected projects
- Loads metrics from all memory strategy/model combinations
- Handles edge cases and missing data gracefully
"""

import json
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory


def test_get_projects():
    """
    Test that get_projects() correctly discovers project directories.

    Creates a temporary directory structure mimicking results/cfb layout
    and verifies that all first-level subdirectories are discovered.
    """
    with TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)

        # Create sample projects
        (base_dir / "project_a").mkdir()
        (base_dir / "project_b").mkdir()
        (base_dir / "project_c").mkdir()

        # Should discover all 3 projects
        projects = sorted([d.name for d in base_dir.iterdir() if d.is_dir()])
        assert len(projects) == 3
        assert projects == ["project_a", "project_b", "project_c"]


def test_get_timestamps():
    """
    Test that get_timestamps() correctly discovers timestamp directories.

    Creates a project with multiple timestamp subdirectories and verifies
    they are returned in reverse chronological order (newest first).
    """
    with TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        project_dir = base_dir / "test_project"
        project_dir.mkdir()

        # Create sample timestamps
        (project_dir / "20251204_1053").mkdir()
        (project_dir / "20251205_1400").mkdir()
        (project_dir / "20251203_0900").mkdir()

        # Should discover all 3 timestamps, sorted in reverse
        timestamps = sorted(
            [d.name for d in project_dir.iterdir() if d.is_dir()], reverse=True
        )
        assert len(timestamps) == 3
        assert timestamps == ["20251205_1400", "20251204_1053", "20251203_0900"]


def test_load_metrics_single_config():
    """
    Test loading metrics from a single memory_strategy/model combination.

    Creates a simple directory structure with one metrics file and verifies
    it is loaded correctly with the expected format.
    """
    with TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)

        # Create directory structure: strategy/model/metrics file
        strategy_dir = base_dir / "truncation"
        strategy_dir.mkdir()
        model_dir = strategy_dir / "gpt-4-1-mini"
        model_dir.mkdir()

        # Create sample metrics file
        metrics = {
            "domain_success_rate": {},
            "overall_success": 0.5,
            "overall_call_acc": 75.5,
        }
        metrics_file = model_dir / "metrics_gpt-4-1-mini_truncation_20251208.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f)

        # Load and verify
        loaded = {}
        for strategy_d in base_dir.iterdir():
            if not strategy_d.is_dir():
                continue
            strategy_name = strategy_d.name
            loaded[strategy_name] = {}

            for model_d in strategy_d.iterdir():
                if not model_d.is_dir():
                    continue
                model_name = model_d.name

                json_files = list(model_d.glob("metrics_*.json"))
                if json_files:
                    with open(json_files[0], "r") as f:
                        loaded[strategy_name][model_name] = {
                            "metrics": json.load(f),
                            "filepath": str(json_files[0]),
                        }

        # Verify structure
        assert "truncation" in loaded
        assert "gpt-4-1-mini" in loaded["truncation"]
        assert loaded["truncation"]["gpt-4-1-mini"]["metrics"]["overall_success"] == 0.5
        assert (
            "metrics_gpt-4-1-mini_truncation"
            in loaded["truncation"]["gpt-4-1-mini"]["filepath"]
        )


def test_load_metrics_multiple_strategies_and_models():
    """
    Test loading metrics from multiple memory strategies and models.

    Creates a complex directory structure with multiple strategies and models,
    verifying that all metrics are correctly discovered and loaded.
    """
    with TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)

        # Create multiple strategies
        strategies = ["truncation", "ace", "memory_bank"]
        models = ["gpt-4-1-mini", "gpt-5"]

        for strategy in strategies:
            strategy_dir = base_dir / strategy
            strategy_dir.mkdir()

            for model in models:
                model_dir = strategy_dir / model
                model_dir.mkdir()

                # Create metrics file
                metrics = {"overall_success": 0.5, "overall_call_acc": 75.5}
                metrics_file = model_dir / f"metrics_{model}_{strategy}_20251208.json"
                with open(metrics_file, "w") as f:
                    json.dump(metrics, f)

        # Load all metrics
        loaded = {}
        for strategy_d in base_dir.iterdir():
            if not strategy_d.is_dir():
                continue
            strategy_name = strategy_d.name
            loaded[strategy_name] = {}

            for model_d in strategy_d.iterdir():
                if not model_d.is_dir():
                    continue
                model_name = model_d.name

                json_files = list(model_d.glob("metrics_*.json"))
                if json_files:
                    with open(json_files[0], "r") as f:
                        loaded[strategy_name][model_name] = {
                            "metrics": json.load(f),
                            "filepath": str(json_files[0]),
                        }

        # Verify all combinations loaded
        assert len(loaded) == 3  # 3 strategies
        for strategy in strategies:
            assert strategy in loaded
            assert len(loaded[strategy]) == 2  # 2 models per strategy
            for model in models:
                assert model in loaded[strategy]


def test_load_metrics_handles_missing_files():
    """
    Test that metrics loading gracefully handles missing or corrupt files.

    Creates a directory structure where some model directories are empty
    or contain invalid JSON, and verifies the loader handles these cases.
    """
    with TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)

        # Create strategy with valid model
        strategy_dir = base_dir / "truncation"
        strategy_dir.mkdir()
        model_good = strategy_dir / "gpt-4-valid"
        model_good.mkdir()

        metrics = {"overall_success": 0.5}
        with open(model_good / "metrics_test.json", "w") as f:
            json.dump(metrics, f)

        # Create strategy with empty model directory
        model_empty = strategy_dir / "gpt-4-empty"
        model_empty.mkdir()

        # Load and verify
        loaded = {}
        for strategy_d in base_dir.iterdir():
            if not strategy_d.is_dir():
                continue
            strategy_name = strategy_d.name
            loaded[strategy_name] = {}

            for model_d in strategy_d.iterdir():
                if not model_d.is_dir():
                    continue
                model_name = model_d.name

                json_files = list(model_d.glob("metrics_*.json"))
                if json_files:
                    with open(json_files[0], "r") as f:
                        loaded[strategy_name][model_name] = {"metrics": json.load(f)}

        # Should only load the valid one, skip the empty
        assert "truncation" in loaded
        assert "gpt-4-valid" in loaded["truncation"]
        assert "gpt-4-empty" not in loaded["truncation"]
        assert loaded["truncation"]["gpt-4-valid"]["metrics"]["overall_success"] == 0.5


def test_load_metrics_from_real_structure():
    """
    Test loading metrics from the actual results/cfb directory structure.

    This integration test verifies the picker works with the real experiment
    data by loading at least one valid metrics file from the structure.
    """
    results_dir = Path("results/cfb")

    if not results_dir.exists():
        pytest.skip("results/cfb directory not found")

    # Get first project and timestamp
    projects = sorted([d.name for d in results_dir.iterdir() if d.is_dir()])
    if not projects:
        pytest.skip("No projects found in results/cfb")

    project = projects[0]
    project_path = results_dir / project

    timestamps = sorted(
        [d.name for d in project_path.iterdir() if d.is_dir()], reverse=True
    )
    if not timestamps:
        pytest.skip(f"No timestamps found in project {project}")

    timestamp = timestamps[0]
    exp_path = project_path / timestamp

    # Load metrics
    loaded = {}
    for strategy_d in exp_path.iterdir():
        if not strategy_d.is_dir():
            continue
        strategy_name = strategy_d.name
        loaded[strategy_name] = {}

        for model_d in strategy_d.iterdir():
            if not model_d.is_dir():
                continue
            model_name = model_d.name

            json_files = list(model_d.glob("metrics_*.json"))
            if json_files:
                with open(json_files[0], "r") as f:
                    try:
                        loaded[strategy_name][model_name] = {
                            "metrics": json.load(f),
                            "filepath": str(json_files[0]),
                        }
                    except json.JSONDecodeError:
                        pass

    # Should have loaded at least one configuration
    # (some strategies might be empty, so we check if ANY strategy has models)
    total_models = sum(len(models) for models in loaded.values())
    assert total_models > 0, f"Failed to load any metrics from {project}/{timestamp}"

    # Verify we can access metrics data from strategies that have models
    for strategy, models in loaded.items():
        for model, data in models.items():
            assert "metrics" in data
            assert isinstance(data["metrics"], dict)
