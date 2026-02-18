"""Tests for thesis_graphics utilities.

These tests validate that the graphics utilities:
- Load metrics with domain information from the expected directory structure
- Handle domain information correctly in the long-format DataFrame
- Create output folders and save CSV tables correctly
- Produce PDF plots with the requested naming pattern
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import pandas as pd
import pytest

import tools.thesis_graphics as tg


def test_load_metrics_long_extracts_domains():
    """Verify metrics are loaded with domain information.

    This test builds a temporary experiment folder with nested metrics
    (e.g., domain_turn_acc with Hotels and Cross keys). It confirms that
    the domain column is populated correctly.
    """
    with TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        metrics_path = base_dir / "project_a" / "20260101_0000" / "ace" / "gpt-4-1-mini"
        metrics_path.mkdir(parents=True)
        metrics_file = metrics_path / "metrics_test.json"
        metrics_file.write_text(
            """
            {
              "overall_success": 0.75,
              "domain_turn_acc": {"Hotels": 50.0, "Cross": 25.0}
            }
            """
        )

        df = tg.load_metrics_long(base_dir, "project_a", "20260101_0000")

        assert not df.empty
        assert "domain" in df.columns

        # Check overall metric has domain "Overall"
        overall_rows = df.loc[df["metric"] == "overall_success"]
        assert len(overall_rows) == 1
        assert overall_rows.iloc[0]["domain"] == "Overall"
        assert overall_rows.iloc[0]["value"] == 0.75

        # Check nested metrics have correct domains
        domain_rows = df.loc[df["metric"] == "domain_turn_acc"]
        assert len(domain_rows) == 2
        domains = set(domain_rows["domain"])
        assert domains == {"Hotels", "Cross"}


def test_save_metric_tables_creates_csv_files():
    """Ensure CSV tables are saved with correct structure.

    This test uses a small synthetic dataframe with domains and verifies
    that the tables are written correctly.
    """
    df = pd.DataFrame(
        [
            {
                "memory_strategy": "ace",
                "model": "gpt-4-1-mini",
                "metric": "overall_success",
                "domain": "Overall",
                "value": 0.5,
            },
            {
                "memory_strategy": "ace",
                "model": "gpt-5",
                "metric": "overall_success",
                "domain": "Overall",
                "value": 0.6,
            },
            {
                "memory_strategy": "truncation",
                "model": "gpt-4-1-mini",
                "metric": "overall_success",
                "domain": "Overall",
                "value": 0.4,
            },
            {
                "memory_strategy": "truncation",
                "model": "gpt-5",
                "metric": "overall_success",
                "domain": "Overall",
                "value": 0.7,
            },
            {
                "memory_strategy": "ace",
                "model": "gpt-4-1-mini",
                "metric": "domain_acc",
                "domain": "Hotels",
                "value": 10.0,
            },
            {
                "memory_strategy": "ace",
                "model": "gpt-4-1-mini",
                "metric": "domain_acc",
                "domain": "Cross",
                "value": 20.0,
            },
        ]
    )

    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # save_metric_tables now only saves all_metrics.csv (long format)
        tg.save_metric_tables(df, output_dir)
        tg.save_results_table(df, output_dir)

        assert (output_dir / "all_metrics.csv").exists()
        assert (output_dir / "results_table.csv").exists()

        # Check long-format CSV structure
        long_df = pd.read_csv(output_dir / "all_metrics.csv")
        assert set(long_df.columns) == {
            "memory_strategy",
            "model",
            "metric",
            "domain",
            "value",
        }


def test_save_all_plots_writes_pdfs():
    """Confirm PDF plot files are created with correct naming.

    This test generates a minimal dataframe and ensures that
    model-comparison and strategy-comparison plots are written as PDFs.
    """
    df = pd.DataFrame(
        [
            {
                "memory_strategy": "ace",
                "model": "gpt-4-1-mini",
                "metric": "overall_success",
                "domain": "Overall",
                "value": 0.5,
            },
            {
                "memory_strategy": "ace",
                "model": "gpt-5",
                "metric": "overall_success",
                "domain": "Overall",
                "value": 0.6,
            },
            {
                "memory_strategy": "truncation",
                "model": "gpt-4-1-mini",
                "metric": "overall_success",
                "domain": "Overall",
                "value": 0.4,
            },
        ]
    )

    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        saved_paths = tg.save_all_plots(df, output_dir)

        # Now saves both PDF and PNG for each plot
        assert len(saved_paths) >= 4  # model + strategy comparison x (PDF + PNG)
        assert any("model_comparison" in path.name for path in saved_paths)
        assert any("strategy_comparison" in path.name for path in saved_paths)
        # Should have both PDF and PNG files
        pdf_paths = [p for p in saved_paths if p.suffix == ".pdf"]
        png_paths = [p for p in saved_paths if p.suffix == ".png"]
        assert len(pdf_paths) >= 2
        assert len(png_paths) >= 2
        # Verify no dots in stems (except the extension)
        assert all("." not in path.stem for path in saved_paths)


def test_plot_model_comparison_returns_figure():
    """Verify plot_model_comparison returns a matplotlib Figure."""
    df = pd.DataFrame(
        [
            {
                "memory_strategy": "ace",
                "model": "gpt-4",
                "metric": "acc",
                "domain": "Overall",
                "value": 0.5,
            },
            {
                "memory_strategy": "ace",
                "model": "gpt-5",
                "metric": "acc",
                "domain": "Overall",
                "value": 0.6,
            },
        ]
    )

    fig = tg.plot_model_comparison(df, "acc")
    assert fig is not None
    plt.close(fig)


def test_plot_strategy_comparison_returns_figure():
    """Verify plot_strategy_comparison returns a matplotlib Figure."""
    df = pd.DataFrame(
        [
            {
                "memory_strategy": "ace",
                "model": "gpt-4",
                "metric": "acc",
                "domain": "Overall",
                "value": 0.5,
            },
            {
                "memory_strategy": "truncation",
                "model": "gpt-4",
                "metric": "acc",
                "domain": "Overall",
                "value": 0.6,
            },
        ]
    )

    fig = tg.plot_strategy_comparison(df, "acc")
    assert fig is not None
    plt.close(fig)


def test_load_metrics_from_real_structure():
    """Integration test with actual results directory using single timestamp loader.

    This test verifies the loader works with the real experiment data.
    """
    results_dir = Path("results/cfb")

    if not results_dir.exists():
        pytest.skip("results/cfb directory not found")

    projects = sorted([d.name for d in results_dir.iterdir() if d.is_dir()])
    if not projects:
        pytest.skip("No projects found in results/cfb")

    project = projects[0]
    project_path = results_dir / project

    timestamps = sorted(
        [d.name for d in project_path.iterdir() if d.is_dir()],
        reverse=True,
    )
    if not timestamps:
        pytest.skip(f"No timestamps found in project {project}")

    timestamp = timestamps[0]

    df = tg.load_metrics_long(results_dir, project, timestamp)

    assert not df.empty, f"Failed to load any metrics from {project}/{timestamp}"
    assert "memory_strategy" in df.columns
    assert "model" in df.columns
    assert "metric" in df.columns
    assert "domain" in df.columns
    assert "value" in df.columns


def test_load_metrics_project_aggregates_timestamps():
    """Verify load_metrics_project aggregates data across all timestamps.

    This test creates a project with two timestamps containing the same
    strategy/model combo to verify that duplicates are prefixed with timestamp.
    """
    with TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)

        # Create two timestamps with same strategy/model (should be disambiguated)
        for ts in ["20260101_1000", "20260102_1000"]:
            metrics_path = base_dir / "test_project" / ts / "ace" / "gpt-4"
            metrics_path.mkdir(parents=True)
            metrics_file = metrics_path / "metrics_test.json"
            metrics_file.write_text('{"accuracy": 0.8}')

        df = tg.load_metrics_project(base_dir, "test_project")

        assert not df.empty
        assert len(df) == 2  # Two rows (one per timestamp)

        # Strategies should be prefixed with timestamp since there are duplicates
        strategies = df["memory_strategy"].unique()
        assert len(strategies) == 2
        assert all("ace" in s for s in strategies)
        # Should have timestamp prefix
        assert any("20260101" in s for s in strategies)
        assert any("20260102" in s for s in strategies)


def test_load_metrics_project_no_prefix_for_unique():
    """Verify no timestamp prefix when strategy/model combos are unique.

    When each timestamp has different strategy/model combinations,
    no disambiguation prefix should be added.
    """
    with TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)

        # Timestamp 1: ace/gpt-4
        path1 = base_dir / "test_project" / "20260101_1000" / "ace" / "gpt-4"
        path1.mkdir(parents=True)
        (path1 / "metrics_test.json").write_text('{"accuracy": 0.8}')

        # Timestamp 2: truncation/gpt-5 (different combo)
        path2 = base_dir / "test_project" / "20260102_1000" / "truncation" / "gpt-5"
        path2.mkdir(parents=True)
        (path2 / "metrics_test.json").write_text('{"accuracy": 0.9}')

        df = tg.load_metrics_project(base_dir, "test_project")

        assert not df.empty
        strategies = set(df["memory_strategy"].unique())
        # No prefix since combos are unique
        assert strategies == {"ace", "truncation"}


def test_load_metrics_project_from_real_data():
    """Integration test: load project-level aggregated data from real results.

    This test uses load_metrics_project to aggregate all timestamps for a project.
    """
    results_dir = Path("results/cfb")

    if not results_dir.exists():
        pytest.skip("results/cfb directory not found")

    projects = sorted([d.name for d in results_dir.iterdir() if d.is_dir()])
    if not projects:
        pytest.skip("No projects found in results/cfb")

    project = projects[0]

    df = tg.load_metrics_project(results_dir, project)

    assert not df.empty, f"Failed to load metrics for project {project}"
    assert "memory_strategy" in df.columns
    assert "model" in df.columns
    assert "metric" in df.columns
    assert "domain" in df.columns
    assert "value" in df.columns


def test_load_compressed_project_extracts_token_metrics():
    """Verify compressed trace files are parsed correctly with all token metrics.

    This test creates a minimal compressed trace file structure and validates
    that all token metrics are extracted correctly: max_token_count, avg_token_count,
    final_token_count, min_token_count, num_steps, total_input_tokens,
    avg_compression_ratio, max_compression_ratio, token_growth_rate, and step_aggregates.
    Domain is extracted from task_id (e.g., "Hotels-104" -> "Hotels").
    """
    import json

    with TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)

        # Create compressed trace file with two tasks in two domains
        compressed_path = (
            base_dir / "test_proj" / "20260212_1011" / "ace" / "gpt-4-1-mini"
        )
        compressed_path.mkdir(parents=True)
        compressed_file = (
            compressed_path / "compressed_gpt-4-1-mini_ace_20260212_1011.json"
        )

        # Task 1: Hotels-104 with 3 steps
        # Task 2: Flights-25 with 2 steps
        compressed_data = [
            {
                "id": "Hotels-104",
                "memory_method": "ace",
                "compressed_trace": [
                    {
                        "step": 1,
                        "input_token_count": 100,
                        "compressed_token_count": 200,
                        "compression_ratio": 2.0,
                        "memory_method": "ace",
                    },
                    {
                        "step": 2,
                        "input_token_count": 150,
                        "compressed_token_count": 300,
                        "compression_ratio": 2.0,
                        "memory_method": "ace",
                    },
                    {
                        "step": 3,
                        "input_token_count": 200,
                        "compressed_token_count": 250,
                        "compression_ratio": 1.25,
                        "memory_method": "ace",
                    },
                ],
            },
            {
                "id": "Flights-25",
                "memory_method": "ace",
                "compressed_trace": [
                    {
                        "step": 1,
                        "input_token_count": 50,
                        "compressed_token_count": 100,
                        "compression_ratio": 2.0,
                        "memory_method": "ace",
                    },
                    {
                        "step": 2,
                        "input_token_count": 75,
                        "compressed_token_count": 200,
                        "compression_ratio": 2.67,
                        "memory_method": "ace",
                    },
                ],
            },
        ]

        compressed_file.write_text(json.dumps(compressed_data, indent=2))

        # Load compressed project
        df = tg.load_compressed_project(base_dir, "test_proj")

        assert not df.empty, "DataFrame should not be empty"

        # Check columns
        expected_cols = {
            "timestamp",
            "memory_strategy",
            "model",
            "task_id",
            "metric",
            "value",
            "domain",
        }
        assert set(df.columns) == expected_cols

        # Check Hotels-104 task
        hotels_df = df[df["task_id"] == "Hotels-104"]
        assert len(hotels_df) == 10, "Should have 10 metrics per task"
        assert hotels_df["domain"].iloc[0] == "Hotels", "Domain should be Hotels"
        assert hotels_df["memory_strategy"].iloc[0] == "ace"
        assert hotels_df["model"].iloc[0] == "gpt-4-1-mini"
        assert hotels_df["timestamp"].iloc[0] == "20260212_1011"

        # Verify metrics for Hotels-104 (steps: 200, 300, 250)
        hotels_metrics = {
            row["metric"]: row["value"] for _, row in hotels_df.iterrows()
        }
        assert hotels_metrics["max_token_count"] == 300.0
        assert hotels_metrics["avg_token_count"] == 250.0  # (200+300+250)/3
        assert hotels_metrics["final_token_count"] == 250.0
        assert hotels_metrics["min_token_count"] == 200.0
        assert hotels_metrics["num_steps"] == 3.0
        assert hotels_metrics["total_input_tokens"] == 450.0  # 100+150+200
        assert (
            abs(hotels_metrics["avg_compression_ratio"] - 1.75) < 0.01
        )  # (2.0+2.0+1.25)/3
        assert hotels_metrics["max_compression_ratio"] == 2.0
        # token_growth_rate: (250 - 200) / 200 = 0.25
        assert abs(hotels_metrics["token_growth_rate"] - 0.25) < 0.01

        # Check step_aggregates is JSON string
        step_agg = hotels_metrics["step_aggregates"]
        assert isinstance(step_agg, str)
        step_data = json.loads(step_agg)
        assert step_data == {"1": 200, "2": 300, "3": 250}

        # Check Flights-25 task
        flights_df = df[df["task_id"] == "Flights-25"]
        assert len(flights_df) == 10
        assert flights_df["domain"].iloc[0] == "Flights", "Domain should be Flights"

        # Verify metrics for Flights-25 (steps: 100, 200)
        flights_metrics = {
            row["metric"]: row["value"] for _, row in flights_df.iterrows()
        }
        assert flights_metrics["max_token_count"] == 200.0
        assert flights_metrics["avg_token_count"] == 150.0  # (100+200)/2
        assert flights_metrics["final_token_count"] == 200.0
        assert flights_metrics["min_token_count"] == 100.0
        assert flights_metrics["num_steps"] == 2.0
        # token_growth_rate: (200 - 100) / 100 = 1.0
        assert abs(flights_metrics["token_growth_rate"] - 1.0) < 0.01


def test_load_compressed_project_multiple_strategies():
    """Verify compressed data from multiple strategies and models is aggregated correctly.

    This test creates compressed trace files for different strategies (ace, truncation)
    and models (gpt-4, gpt-5) to ensure all combinations are loaded properly.
    """
    import json

    with TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        timestamp = "20260212_1011"

        # Create two strategies x two models = 4 combinations
        for strategy in ["ace", "truncation"]:
            for model in ["gpt-4", "gpt-5"]:
                compressed_path = base_dir / "multi_proj" / timestamp / strategy / model
                compressed_path.mkdir(parents=True)
                compressed_file = (
                    compressed_path / f"compressed_{model}_{strategy}_{timestamp}.json"
                )

                compressed_data = [
                    {
                        "id": f"Hotels-{strategy[:3]}-{model[-1]}",
                        "memory_method": strategy,
                        "compressed_trace": [
                            {
                                "step": 1,
                                "input_token_count": 100,
                                "compressed_token_count": 150,
                                "compression_ratio": 1.5,
                                "memory_method": strategy,
                            }
                        ],
                    }
                ]

                compressed_file.write_text(json.dumps(compressed_data))

        # Load all compressed data
        df = tg.load_compressed_project(base_dir, "multi_proj")

        assert not df.empty
        # 4 combinations x 1 task each x 10 metrics = 40 rows
        assert len(df) == 40

        # Check all strategies are present
        strategies = df["memory_strategy"].unique()
        assert set(strategies) == {"ace", "truncation"}

        # Check all models are present
        models = df["model"].unique()
        assert set(models) == {"gpt-4", "gpt-5"}

        # Check all tasks have Hotels domain
        domains = df["domain"].unique()
        assert "Hotels" in domains


def test_load_compressed_project_handles_empty_project():
    """Verify function returns empty DataFrame for non-existent project."""
    with TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        df = tg.load_compressed_project(base_dir, "nonexistent_project")
        assert df.empty


def test_load_compressed_project_handles_missing_compressed_files():
    """Verify function handles directories without compressed files gracefully."""
    with TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)

        # Create directory structure but no compressed files
        path = base_dir / "empty_proj" / "20260212_1011" / "ace" / "gpt-4"
        path.mkdir(parents=True)
        (path / "other_file.json").write_text("{}")

        df = tg.load_compressed_project(base_dir, "empty_proj")
        assert df.empty


def test_load_compressed_project_from_real_data():
    """Integration test: load compressed token data from real experiment results.

    This test uses the actual compressed trace files from the demo_trace_diff project
    to verify the function works with real data. It checks that all expected
    token metrics are present and have reasonable values.
    """
    results_dir = Path("results/cfb")

    if not results_dir.exists():
        pytest.skip("results/cfb directory not found")

    # Use demo_trace_diff project
    project = "demo_trace_diff"
    project_path = results_dir / project

    if not project_path.exists():
        pytest.skip(f"Project {project} not found")

    df = tg.load_compressed_project(results_dir, project)

    if df.empty:
        pytest.skip(f"No compressed files found in project {project}")

    # Verify DataFrame structure
    expected_cols = {
        "timestamp",
        "memory_strategy",
        "model",
        "task_id",
        "metric",
        "value",
        "domain",
    }
    assert set(df.columns) == expected_cols

    # Check all expected metrics are present
    expected_metrics = {
        "max_token_count",
        "avg_token_count",
        "final_token_count",
        "min_token_count",
        "num_steps",
        "total_input_tokens",
        "avg_compression_ratio",
        "max_compression_ratio",
        "token_growth_rate",
        "step_aggregates",
    }
    actual_metrics = set(df["metric"].unique())
    assert actual_metrics == expected_metrics

    # Verify domains are extracted from task_id
    domains = df["domain"].unique()
    assert len(domains) > 0
    assert all(domain != "Unknown" for domain in domains)

    # Check that task_ids follow the pattern Domain-Number
    task_ids = df["task_id"].unique()
    for task_id in task_ids:
        assert "-" in task_id, f"Task ID {task_id} should contain hyphen"
        domain = task_id.split("-")[0]
        assert domain in domains

    # Verify numeric metrics have reasonable values
    numeric_df = df[df["metric"] != "step_aggregates"]
    assert (numeric_df["value"] >= 0).all(), "Token counts should be non-negative"

    # Verify step_aggregates is valid JSON
    step_agg_df = df[df["metric"] == "step_aggregates"]
    for _, row in step_agg_df.iterrows():
        import json

        try:
            step_data = json.loads(row["value"])
            assert isinstance(step_data, dict)
        except json.JSONDecodeError:
            pytest.fail(f"step_aggregates should be valid JSON: {row['value']}")
