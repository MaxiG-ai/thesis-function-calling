# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.19.9",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="full")


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
    return BASE_DIR, OUTPUT_ROOT, alt, dl, mo, plot, plt, sns


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


@app.cell(hide_code=True)
<<<<<<< HEAD
def _(BASE_DIR, mo, parse_experiment_results, project_dropdown):
    mo.stop(project_dropdown.value is None, None)
    turn_count_df = parse_experiment_results(BASE_DIR.as_posix() + "/" + project_dropdown.value)
    return (turn_count_df,)


@app.cell(hide_code=True)
def _(Path, json, parse_timestamp, pd):
    def parse_experiment_metrics(experiment_path: str) -> pd.DataFrame:
        """
        Parse all metrics files from subdirectories into a DataFrame.

        Iterates through all subdirectories of an experiment folder structured as:
        experiment_path/<timestamp>/<memory_strategy>/<model>/metrics_*.json

        Args:
            experiment_path: Path to experiment directory, e.g., "results/cfb/demo_trace_diff"

        Returns:
            DataFrame with columns:
            - memory_strategy: Memory strategy key, e.g., "ace"
            - model: Model name, e.g., "gpt-4-1-mini"
            - timestamp: ISO-formatted timestamp from subfolder
            - domain_success_rate: Dict of domain -> success rate
            - domain_turn_acc: Dict of domain -> turn accuracy
            - domain_call_acc: Dict of domain -> call accuracy
            - overall_success: Overall success rate (percentage)
            - overall_call_acc: Overall call accuracy (percentage)
            - complete_score_avg: Average completion score
            - correct_score_avg: Average correctness score
        """
        experiment_dir = Path(experiment_path)
        rows = []

        for timestamp_dir in experiment_dir.iterdir():
            if not timestamp_dir.is_dir():
                continue

            timestamp_str = timestamp_dir.name
            timestamp_iso = parse_timestamp(timestamp_str)
            if timestamp_iso is None:
                continue

            for strategy_dir in timestamp_dir.iterdir():
                if not strategy_dir.is_dir():
                    continue

                memory_strategy = strategy_dir.name

                for model_dir in strategy_dir.iterdir():
                    if not model_dir.is_dir():
                        continue

                    model = model_dir.name

                    # Find metrics JSON file
                    for json_file in model_dir.glob(f"metrics_{model}_{memory_strategy}_{timestamp_str}.json"):
                        with open(json_file, "r", encoding="utf-8") as f:
                            metrics = json.load(f)

                        row = {
                            "memory_strategy": memory_strategy,
                            "model": model,
                            "timestamp": timestamp_iso,
                            "domain_success_rate": metrics.get("domain_success_rate", {}),
                            "domain_turn_acc": metrics.get("domain_turn_acc", {}),
                            "domain_call_acc": metrics.get("domain_call_acc", {}),
                            "overall_success": metrics.get("overall_success"),
                            "overall_call_acc": metrics.get("overall_call_acc"),
                            "complete_score_avg": metrics.get("complete_score_avg"),
                            "correct_score_avg": metrics.get("correct_score_avg"),
                        }
                        rows.append(row)

        return pd.DataFrame(rows)

    return (parse_experiment_metrics,)


@app.cell(hide_code=True)
def _(BASE_DIR, mo, parse_experiment_metrics, project_dropdown):
    mo.stop(project_dropdown.value is None, None)
    agg_metrics_df = parse_experiment_metrics(BASE_DIR.as_posix() + "/" + project_dropdown.value)
    return (agg_metrics_df,)


@app.cell(hide_code=True)
def _(pd):
    def join_results_with_metrics(
        details_df: pd.DataFrame,
        metrics_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Join the detailed task-level results with high-level experiment metrics.

        The join is performed on (memory_strategy, model, timestamp) - the unique
        identifier for each experiment run.

        Args:
            details_df: DataFrame from parse_experiment_results() with task-level data
            metrics_df: DataFrame from parse_experiment_metrics() with aggregate metrics

        Returns:
            DataFrame with all columns from details_df plus the metrics columns:
            - domain_success_rate, domain_turn_acc, domain_call_acc
            - overall_success, overall_call_acc
            - complete_score_avg, correct_score_avg
        """
        join_keys = ["memory_strategy", "model", "timestamp"]

        return details_df.merge(
            metrics_df,
            on=join_keys,
            how="left",
            suffixes=("", "_agg"),
        )

    return (join_results_with_metrics,)


@app.cell(hide_code=True)
def _(
    agg_metrics_df,
    join_results_with_metrics,
    mo,
    project_dropdown,
    turn_count_df,
):
    mo.stop(project_dropdown.value is None, None)
    tables = {
        "Task-Result Table": turn_count_df,
        "Experiment Metrics Table": agg_metrics_df,
        "Task and Metrics Table": join_results_with_metrics(
            details_df=turn_count_df,
            metrics_df=agg_metrics_df
        ),
    }
    mo.vstack(
        [
            mo.md("## View Results as Plain Data"),
            mo.ui.tabs(tables)
            if tables
            else mo.md("No tables available"),
        ])
    return


@app.cell
def _(turn_count_df):
    turn_count_df["domain"] = [val.split("-")[0] for val in turn_count_df.task_id]
    turn_count_df.domain.value_counts()
    return


@app.cell
def _(turn_count_df):
    turn_count_df.dtypes
    return


@app.cell
def _(turn_count_df):
    turn_count_domain_grouped = turn_count_df.groupby(
        ["model", "timestamp", "memory_strategy", "domain"], 
        )["response_llm_judge_correct_score"].value_counts(dropna=False).unstack(fill_value=0)
    turn_count_domain_grouped
=======
def _(mo):
    """Section header for LLM judge analysis."""
    mo.md("## LLM-as-a-Judge Evaluation")
>>>>>>> f9e6443 (Refactor thesis graphics utilities into modular helper functions)
    return


@app.cell(hide_code=True)
def _(mo):
<<<<<<< HEAD
    mo.md(r"""
    ## LLM-as-a-Judge Correctness Evaluation
    """)
=======
    """Subsection: Correctness evaluation."""
    mo.md("### Correctness Scores")
>>>>>>> f9e6443 (Refactor thesis graphics utilities into modular helper functions)
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
    return


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
    return


@app.cell(hide_code=True)
def _(mo):
    """Section header for turn analysis."""
    mo.md("## Turn Count Analysis")
    return


@app.cell(hide_code=True)
def _(mo):
<<<<<<< HEAD
    mo.md(r"""
    ## Metrics per Turn Count
    """)
=======
    """Subsection: Turn distribution."""
    mo.md("### Turn Distribution Histogram")
>>>>>>> f9e6443 (Refactor thesis graphics utilities into modular helper functions)
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

<<<<<<< HEAD
    turn_count_df['turns_cat'] = pd.cut(turn_count_df['total_call_num'], bins=bins, labels=labels, ordered=True)
    return


@app.cell
def _(turn_count_df):
    turn_count_df
    return


@app.cell
def _(turn_count_df):
    #df = turn_count_df[["memory_strategy", "model", "timestamp", "turns_cat", "status"]].groupby(
    success_count = turn_count_df.groupby(
        ["model", "timestamp", "memory_strategy"], 
        )["status"].value_counts(dropna=False).unstack(fill_value=0)
    success_count
    return


@app.cell
def _(turn_count_df):
    turn_count_grouped = turn_count_df.groupby(
        ["model", "timestamp", "memory_strategy"], 
        )["turns_cat"].value_counts(dropna=False).unstack(fill_value=0)
    turn_count_grouped
    return


@app.cell
def _(plt, sns, turn_count_df):
    # 1. Filter for successful trials
    # Ensure your 'turns_cat' is an ordered category for the best visual flow
    success_df = turn_count_df[turn_count_df['status'] == 'Success'].copy()

    # 2. Generate the faceted count plot
=======
    plot.apply_nature_style()
>>>>>>> f9e6443 (Refactor thesis graphics utilities into modular helper functions)
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Token Count Analysis
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    """Section header for data tables."""
    mo.md("## Raw Data Tables")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Metrics per Max Token in Context
    """)
    return


@app.cell
<<<<<<< HEAD
def _(turn_count_df):
    turn_count_df
    return


@app.cell
def _(BASE_DIR, project_dropdown, tg):
    tokens_df = tg.load_compressed_project(BASE_DIR, project_dropdown.value)
    return (tokens_df,)


@app.cell
def _(tokens_df):
    tokens_df.dtypes
    return


@app.cell
def _(tokens_df):
    tokens_df.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Token Usage per Case
    """)
    return


@app.cell
def _(pd, plt, sns, tokens_df):
    # 1. Convert value to numeric (crucial since it's an object/string)
    tokens_df['value'] = pd.to_numeric(tokens_df['value'], errors='coerce')

    # 2. Filter for the metric of interest
    plot_df = tokens_df[tokens_df['metric'] == 'max_token_count'].copy()

    # 3. Filter for one task (e.g., Hotels-104) to avoid overcrowding
    target_task = 'Hotels-104'
    plot_df_task = plot_df[plot_df['task_id'] == target_task]

    # 4. Create the visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df_task, x='memory_strategy', y='value', hue='model')

    plt.title(f'Max Token Count per Strategy and Model (Task: {target_task})')
    plt.ylabel('Token Count')
    plt.xlabel('Memory Strategy')
    plt.legend(title='Model')

    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Further ideas for Evaluation

    - Distribution of Completion/Correctness Reason per model/memory method
    - Clustering of pattern for entire dataset
    """)
    return
=======
def _(mo, project_dropdown, task_results_df):
    """Display raw data in tabbed view."""
    mo.stop(project_dropdown.value is None, None)

    tables = {
        "Task Results": task_results_df,
    }
    mo.ui.tabs(tables) if tables else mo.md("_No tables available_")
<<<<<<< HEAD
    return (tables,)
>>>>>>> f9e6443 (Refactor thesis graphics utilities into modular helper functions)
=======
    return
>>>>>>> fb24573 (Update version and streamline function returns in experiment analysis script)


if __name__ == "__main__":
    app.run()
