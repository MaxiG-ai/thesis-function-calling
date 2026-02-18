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
    import json
    import marimo as mo
    import pandas as pd
    import altair as alt
    import thesis_graphics as tg
    import seaborn as sns
    import matplotlib.pyplot as plt

    from pathlib import Path
    from datetime import datetime
    from typing import Optional

    # Configuration
    BASE_DIR = Path("results/cfb")
    OUTPUT_ROOT = Path("thesis_assets")
    return (
        BASE_DIR,
        OUTPUT_ROOT,
        Optional,
        Path,
        alt,
        datetime,
        json,
        mo,
        pd,
        plt,
        sns,
        tg,
    )


@app.cell(hide_code=True)
def _(BASE_DIR):
    def get_projects() -> list[str]:
        if not BASE_DIR.exists():
            return []
        return sorted([d.name for d in BASE_DIR.iterdir() if d.is_dir()])

    projects = get_projects()
    return (projects,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Experiment Analysis

    Choose the Project below
    """)
    return


@app.cell(hide_code=True)
def _(mo, projects):
    project_dropdown = mo.ui.dropdown(
        options=projects,
        label="Project",
        allow_select_none=True,
        value=None,
    )
    project_dropdown
    return (project_dropdown,)


@app.cell(hide_code=True)
def _(BASE_DIR, mo, project_dropdown, tg):
    # Stop if no selection made
    mo.stop(
        project_dropdown.value is None,
        mo.md("*Select a project to load data.*"),
    )

    # Load all data for project (aggregates across all timestamps)
    metrics_df = tg.load_metrics_project(BASE_DIR, project_dropdown.value)

    mo.stop(
        metrics_df.empty,
        mo.callout(mo.md("No metrics found for this project."), kind="warn"),
    )
    return (metrics_df,)


@app.cell(hide_code=True)
def _(metrics_df, mo, project_dropdown):
    _strategies = sorted(metrics_df["memory_strategy"].unique())
    _models = sorted(metrics_df["model"].unique())
    _n_metrics = metrics_df["metric"].nunique()

    mo.callout(
        mo.md(f"""
    **{project_dropdown.value}** (aggregated across all timestamps)

    Strategies: {len(_strategies)} | Models: {len(_models)} | Metrics: {_n_metrics}
        """),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(alt, metrics_df, mo):
    """Create interactive Altair charts for display."""
    altair_charts = {}
    _metrics = sorted(metrics_df["metric"].unique())

    for _metric in _metrics:
        _subset = metrics_df[metrics_df["metric"] == _metric]
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
    """Display interactive charts."""
    save_btn = mo.ui.run_button(label="Save plots and tables to disk")
    mo.vstack(
        [
            mo.md("## Interactive Charts"),
            mo.ui.tabs(altair_charts)
            if altair_charts
            else mo.md("No charts available"),
            mo.hstack([mo.md("Click the button to save plots and tables as PDFs and CSVs."), save_btn], justify="start"),
            mo.md("---"),
        ]
    )
    return (save_btn,)


@app.cell(hide_code=True)
def _(OUTPUT_ROOT, metrics_df, mo, project_dropdown, save_btn, tg):
    mo.stop(not save_btn.value, None)

    _output_dir = OUTPUT_ROOT / project_dropdown.value
    _output_dir.mkdir(parents=True, exist_ok=True)

    # Save tables
    tg.save_metric_tables(metrics_df, _output_dir)
    tg.save_results_table(metrics_df, _output_dir)

    # Save plots (Nature-quality PDFs)
    _saved = tg.save_all_plots(metrics_df, _output_dir)

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
def _(Optional, Path, datetime, json, pd):
    def parse_experiment_results(experiment_path: str) -> pd.DataFrame:
        """
        Parse all experiment results from subdirectories into a DataFrame.

        Iterates through all subdirectories of an experiment folder structured as:
        experiment_path/<timestamp>/<memory_strategy>/<model>/cfb_*.json

        Args:
            experiment_path: Path to experiment directory, e.g., "results/cfb/demo_trace_diff"

        Returns:
            DataFrame with columns:
            - task_id: Task identifier, e.g., "Hotels-104"
            - memory_strategy: Memory strategy key, e.g., "prog_sum"
            - model: Model name, e.g., "gpt-4-1-mini"
            - timestamp: ISO-formatted timestamp from subfolder
            - success_turn_num: Number of successful turns
            - total_turn_num: Total number of turns
            - correct_call_num: Number of correct function calls
            - total_call_num: Total function calls made
            - real_turn_num: Actual turns (excluding observations)
            - response_llm_judge_complete_score: Completeness score (0, 1, or 2)
            - response_llm_judge_complete_reason: Explanation for completeness score
            - response_llm_judge_correct_score: Correctness score (0, 1, or 2)
            - response_llm_judge_correct_reason: Explanation for correctness score
            - status: Task status (Success/Failed)
            - gen_convs: Generated conversation trace
        """
        experiment_dir = Path(experiment_path)
        rows = []

        # Iterate through timestamp directories
        for timestamp_dir in experiment_dir.iterdir():
            if not timestamp_dir.is_dir():
                continue

            timestamp_str = timestamp_dir.name
            # Parse timestamp from format YYYYMMDD_HHMM to ISO format
            timestamp_iso = parse_timestamp(timestamp_str)
            if timestamp_iso is None:
                continue

            # Iterate through memory strategy directories
            for strategy_dir in timestamp_dir.iterdir():
                if not strategy_dir.is_dir():
                    continue

                memory_strategy = strategy_dir.name

                # Iterate through model directories
                for model_dir in strategy_dir.iterdir():
                    if not model_dir.is_dir():
                        continue

                    model = model_dir.name

                    # Find the main results JSON file (cfb_*.json pattern)
                    for json_file in model_dir.glob(f"cfb_{model}_{memory_strategy}_{timestamp_str}.json"):
                        rows.extend(
                            parse_json_file(
                                json_file, memory_strategy, model, timestamp_iso
                            )
                        )

        return pd.DataFrame(rows)


    def parse_timestamp(timestamp_str: str) -> Optional[str]:
        """
        Parse timestamp string from YYYYMMDD_HHMM format to ISO format.

        Args:
            timestamp_str: Timestamp in format YYYYMMDD_HHMM, e.g., "20260211_1046"

        Returns:
            ISO-formatted timestamp string, or None if parsing fails
        """
        try:
            dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M")
            return dt.isoformat()
        except ValueError:
            return None


    def parse_json_file(
        json_path: Path,
        memory_strategy: str,
        model: str,
        timestamp_iso: str,
    ) -> list[dict]:
        """
        Parse a single JSON results file into row dictionaries.

        Args:
            json_path: Path to the JSON file
            memory_strategy: Memory strategy name
            model: Model name
            timestamp_iso: ISO-formatted timestamp

        Returns:
            List of dictionaries, one per task in the JSON file
        """
        rows = []

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for task in data:
            row = {
                "task_id": task.get("id"),
                "memory_strategy": memory_strategy,
                "model": model,
                "timestamp": timestamp_iso,
                # Extract count_dict fields
                "success_turn_num": task.get("count_dict", {}).get("success_turn_num"),
                "total_turn_num": task.get("count_dict", {}).get("total_turn_num"),
                "correct_call_num": task.get("count_dict", {}).get("correct_call_num"),
                "total_call_num": task.get("count_dict", {}).get("total_call_num"),
                "real_turn_num": task.get("count_dict", {}).get("real_turn_num"),
                # Extract resp_eval fields (may be null)
                "response_llm_judge_complete_score": safe_get_nested(
                    task, "resp_eval", "complete", "score"
                ),
                "response_llm_judge_complete_reason": safe_get_nested(
                    task, "resp_eval", "complete", "reason"
                ),
                "response_llm_judge_correct_score": safe_get_nested(
                    task, "resp_eval", "correct", "score"
                ),
                "response_llm_judge_correct_reason": safe_get_nested(
                    task, "resp_eval", "correct", "reason"
                ),
                "status": task.get("status"),
                "gen_convs": task.get("gen_convs"),
            }
            rows.append(row)

        return rows


    def safe_get_nested(data: dict, *keys) -> Optional[any]:
        """
        Safely navigate nested dictionaries, returning None if any key is missing.

        Args:
            data: Dictionary to navigate
            *keys: Sequence of keys to traverse

        Returns:
            Value at nested path, or None if path doesn't exist
        """
        current = data
        for key in keys:
            if current is None or not isinstance(current, dict):
                return None
            current = current.get(key)
        return current

    return parse_experiment_results, parse_timestamp


@app.cell(hide_code=True)
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
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LLM-as-a-Judge Correctness Evaluation
    """)
    return


@app.cell
def _(plt, sns, turn_count_df):
    sns.set_theme(style="whitegrid")

    correctness_plot = sns.catplot(
        data=turn_count_df, 
        x='response_llm_judge_correct_score', 
        hue='memory_strategy', 
        col='model',
        kind='count',
        palette='viridis',
        height=7, 
        aspect=0.8,
    )

    # 3. Refine labels using LaTeX formatting for scientific clarity
    correctness_plot.set_axis_labels("LLM-as-a-Judge Correctness Evaluation", "Count of Correctness")
    correctness_plot.set_titles(col_template="{col_name}")

    # Prevent label overlapping
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LLM-as-a-Judge Completeness Evaluation
    """)
    return


@app.cell
def _(plt, sns, turn_count_df):
    sns.set_theme(style="whitegrid")

    completeness_plot = sns.catplot(
        data=turn_count_df, 
        x='response_llm_judge_complete_score', 
        hue='memory_strategy', 
        col='model',
        kind='count',
        palette='viridis',
        height=7, 
        aspect=0.8,
    )

    # 3. Refine labels using LaTeX formatting for scientific clarity
    completeness_plot.set_axis_labels("LLM-as-a-Judge Completeness Evaluation", "Count of Completeness")
    completeness_plot.set_titles(col_template="{col_name}")

    # Prevent label overlapping
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Metrics per Turn Count
    """)
    return


@app.cell
def _(sns, turn_count_df):
    sns.histplot(data=turn_count_df.total_call_num, bins=3)
    return


@app.cell
def _(pd, turn_count_df):
    bins = [0, 4, 8, 100]
    labels = ['few_turns', 'med_turns', 'many_turns']

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
    sns.set_theme(style="whitegrid")

    g = sns.catplot(
        data=success_df, 
        x='turns_cat', 
        hue='memory_strategy', 
        col='model',
        kind='count',
        palette='viridis',
        height=7, 
        aspect=0.8,
    )

    # 3. Refine labels using LaTeX formatting for scientific clarity
    g.set_axis_labels("Turns Category", "Count of Successes")
    g.set_titles(col_template="{col_name}")

    # Prevent label overlapping
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Metrics per Max Token in Context
    """)
    return


@app.cell
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


if __name__ == "__main__":
    app.run()
