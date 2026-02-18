# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.19.9",
# ]
# ///

import marimo

__generated_with = "0.19.10"
app = marimo.App(width="full")


# =============================================================================
# 1. SETUP & CONFIGURATION
# =============================================================================


@app.cell(hide_code=True)
def _():
    """Import dependencies and configure paths."""
    import marimo as mo
    import pandas as pd
    import altair as alt
    import seaborn as sns
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Import helper modules
    from helpers import data_loading as dl
    from helpers import plotting as plot

    # Configuration
    BASE_DIR = Path("results/cfb")
    OUTPUT_ROOT = Path("thesis_assets")

    return BASE_DIR, OUTPUT_ROOT, Path, alt, dl, mo, pd, plt, plot, sns


# =============================================================================
# 2. PROJECT SELECTION
# =============================================================================


@app.cell(hide_code=True)
def _(BASE_DIR):
    """Get list of available projects."""

    def get_projects() -> list[str]:
        if not BASE_DIR.exists():
            return []
        return sorted([d.name for d in BASE_DIR.iterdir() if d.is_dir()])

    projects = get_projects()
    return (projects,)


@app.cell(hide_code=True)
def _(mo):
    """Display title and instructions."""
    mo.md("""
    # Experiment Analysis Dashboard

    Select a project from the dropdown below to load and analyze experiment results.
    """)
    return


@app.cell(hide_code=True)
def _(mo, projects):
    """Project selection dropdown."""
    project_dropdown = mo.ui.dropdown(
        options=projects,
        label="Project",
        allow_select_none=True,
        value=None,
    )
    project_dropdown
    return (project_dropdown,)


# =============================================================================
# 3. DATA LOADING
# =============================================================================


@app.cell(hide_code=True)
def _(BASE_DIR, dl, mo, project_dropdown):
    """Load all data for selected project."""
    # Stop if no selection made
    mo.stop(
        project_dropdown.value is None,
        mo.md("*Select a project to load data.*"),
    )

    # Load metrics in long format
    metrics_long_df = dl.load_metrics_long(BASE_DIR, project_dropdown.value)

    mo.stop(
        metrics_long_df.empty,
        mo.callout(mo.md("No metrics found for this project."), kind="warn"),
    )
    return (metrics_long_df,)


@app.cell(hide_code=True)
def _(BASE_DIR, dl, mo, project_dropdown):
    """Load task-level results."""
    mo.stop(project_dropdown.value is None, None)

    # Load task results and add domain column
    task_results_df = dl.load_task_results(BASE_DIR / project_dropdown.value)
    task_results_df = dl.add_domain_column(task_results_df)

    # Add turn category column
    task_results_df = dl.add_turn_category(
        task_results_df,
        column="total_call_num",
        bins=[0, 4, 8, 100],
        labels=["few_turns", "med_turns", "many_turns"],
    )
    return (task_results_df,)


# =============================================================================
# 4. OVERVIEW STATISTICS
# =============================================================================


@app.cell(hide_code=True)
def _(metrics_long_df, mo, project_dropdown):
    """Display project overview statistics."""
    _strategies = sorted(metrics_long_df["memory_strategy"].unique())
    _models = sorted(metrics_long_df["model"].unique())
    _n_metrics = metrics_long_df["metric"].nunique()

    mo.callout(
        mo.md(f"""
    **{project_dropdown.value}** - Experiment Overview

    - **Strategies:** {len(_strategies)}
    - **Models:** {len(_models)}
    - **Metrics:** {_n_metrics}
    - **Data aggregated across all timestamps**
        """),
        kind="info",
    )
    return


# =============================================================================
# 5. INTERACTIVE METRIC CHARTS (ALTAIR)
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    """Section header for interactive charts."""
    mo.md("## Interactive Metric Visualizations")
    return


@app.cell(hide_code=True)
def _(alt, metrics_long_df, mo):
    """Create interactive Altair charts for each metric."""
    altair_charts = {}
    _metrics = sorted(metrics_long_df["metric"].unique())

    for _metric in _metrics:
        _subset = metrics_long_df[metrics_long_df["metric"] == _metric]
        if _subset.empty:
            continue

        # Create grouped bar chart with domain as color
        _chart = (
            alt.Chart(_subset)
            .mark_bar()
            .encode(
                x=alt.X("model:N", title="Model"),
                y=alt.Y("value:Q", title=_metric.replace("_", " ").title()),
                color=alt.Color("domain:N", title="Domain"),
                xOffset="domain:N",
                tooltip=["model", "memory_strategy", "domain", "value"],
            )
            .facet(
                column=alt.Column("memory_strategy:N", title="Memory Strategy"),
            )
            .properties(title=_metric.replace("_", " ").title())
            .resolve_scale(y="independent")
        )
        altair_charts[_metric] = mo.ui.altair_chart(_chart)
    return (altair_charts,)


@app.cell(hide_code=True)
def _(altair_charts, mo):
    """Display interactive charts with save button."""
    save_btn = mo.ui.run_button(label="Save plots and tables to disk")
    mo.vstack(
        [
            mo.ui.tabs(altair_charts)
            if altair_charts
            else mo.md("_No charts available_"),
            mo.hstack(
                [
                    mo.md(
                        "Click to save Nature-quality plots (PDF/PNG) and CSV tables:"
                    ),
                    save_btn,
                ],
                justify="start",
            ),
        ]
    )
    return (save_btn,)


@app.cell(hide_code=True)
def _(OUTPUT_ROOT, dl, metrics_long_df, mo, plot, project_dropdown, save_btn):
    """Save plots and tables when button is clicked."""
    mo.stop(not save_btn.value, None)

    _output_dir = OUTPUT_ROOT / project_dropdown.value
    _output_dir.mkdir(parents=True, exist_ok=True)

    # Save tables
    dl.save_metric_tables(metrics_long_df, _output_dir)
    dl.save_results_table(metrics_long_df, _output_dir)

    # Save plots (Nature-quality PDFs)
    _saved = plot.save_all_plots(metrics_long_df, _output_dir)

    mo.callout(
        mo.md(f"""
    **Saved to** `{_output_dir}`

    - CSV tables: `all_metrics.csv`, `results_table.csv`
    - Plots: {len(_saved)} files (PDF + PNG at 300 DPI)
        """),
        kind="success",
    )
    return


# =============================================================================
# 6. LLM-AS-A-JUDGE ANALYSIS
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    """Section header for LLM judge analysis."""
    mo.md("## LLM-as-a-Judge Evaluation")
    return


@app.cell(hide_code=True)
def _(mo):
    """Subsection: Correctness evaluation."""
    mo.md("### Correctness Scores")
    return


@app.cell
def _(plot, plt, sns, task_results_df):
    """Plot LLM judge correctness evaluation."""
    plot.apply_nature_style()
    sns.set_theme(style="whitegrid")

    correctness_plot = sns.catplot(
        data=task_results_df,
        x="response_llm_judge_correct_score",
        hue="memory_strategy",
        col="model",
        kind="count",
        palette="viridis",
        height=7,
        aspect=0.8,
    )

    correctness_plot.set_axis_labels("LLM-as-a-Judge Correctness Score", "Count")
    correctness_plot.set_titles(col_template="{col_name}")

    plt.tight_layout()
    plt.show()
    return (correctness_plot,)


@app.cell(hide_code=True)
def _(mo):
    """Subsection: Completeness evaluation."""
    mo.md("### Completeness Scores")
    return


@app.cell
def _(plot, plt, sns, task_results_df):
    """Plot LLM judge completeness evaluation."""
    plot.apply_nature_style()
    sns.set_theme(style="whitegrid")

    completeness_plot = sns.catplot(
        data=task_results_df,
        x="response_llm_judge_complete_score",
        hue="memory_strategy",
        col="model",
        kind="count",
        palette="viridis",
        height=7,
        aspect=0.8,
    )

    completeness_plot.set_axis_labels("LLM-as-a-Judge Completeness Score", "Count")
    completeness_plot.set_titles(col_template="{col_name}")

    plt.tight_layout()
    plt.show()
    return (completeness_plot,)


# =============================================================================
# 7. TURN COUNT ANALYSIS
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    """Section header for turn analysis."""
    mo.md("## Turn Count Analysis")
    return


@app.cell(hide_code=True)
def _(mo):
    """Subsection: Turn distribution."""
    mo.md("### Turn Distribution Histogram")
    return


@app.cell
def _(sns, task_results_df):
    """Histogram of total call numbers."""
    sns.histplot(data=task_results_df["total_call_num"], bins=20)
    return


@app.cell(hide_code=True)
def _(mo):
    """Subsection: Success by turn category."""
    mo.md("### Success Rate by Turn Category")
    return


@app.cell
def _(plot, plt, sns, task_results_df):
    """Plot success count by turn category."""
    # Filter for successful trials
    success_df = task_results_df[task_results_df["status"] == "Success"].copy()

    plot.apply_nature_style()
    sns.set_theme(style="whitegrid")

    success_by_turns = sns.catplot(
        data=success_df,
        x="turns_cat",
        hue="memory_strategy",
        col="model",
        kind="count",
        palette="viridis",
        height=7,
        aspect=0.8,
    )

    success_by_turns.set_axis_labels("Turn Category", "Count of Successes")
    success_by_turns.set_titles(col_template="{col_name}")

    plt.tight_layout()
    plt.show()
    return (success_by_turns, success_df)


# =============================================================================
# 8. DATA TABLES
# =============================================================================


@app.cell(hide_code=True)
def _(mo):
    """Section header for data tables."""
    mo.md("## Raw Data Tables")
    return


@app.cell
def _(mo, project_dropdown, task_results_df):
    """Display raw data in tabbed view."""
    mo.stop(project_dropdown.value is None, None)

    tables = {
        "Task Results": task_results_df,
    }
    mo.ui.tabs(tables) if tables else mo.md("_No tables available_")
    return (tables,)


if __name__ == "__main__":
    app.run()
