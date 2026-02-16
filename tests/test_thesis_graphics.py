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
