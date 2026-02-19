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
    return BASE_DIR, OUTPUT_ROOT, Path, alt, dl, mo, pd, plot, plt, sns


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

    # Load task results and add domain columnK
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
                    for json_file in model_dir.glob(
                        f"metrics_{model}_{memory_strategy}_{timestamp_str}.json"
                    ):
                        with open(json_file, "r", encoding="utf-8") as f:
                            metrics = json.load(f)

                        row = {
                            "memory_strategy": memory_strategy,
                            "model": model,
                            "timestamp": timestamp_iso,
                            "domain_success_rate": metrics.get(
                                "domain_success_rate", {}
                            ),
                            "domain_turn_acc": metrics.get("domain_turn_acc", {}),
                            "domain_call_acc": metrics.get("domain_call_acc", {}),
                            "overall_success": metrics.get("overall_success"),
                            "overall_call_acc": metrics.get("overall_call_acc"),
                            "complete_score_avg": metrics.get("complete_score_avg"),
                            "correct_score_avg": metrics.get("correct_score_avg"),
                        }
                        rows.append(row)

        return pd.DataFrame(rows)

    return


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

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LLM-as-a-Judge Correctness Evaluation
    """)
    return


@app.cell
def _(alt, task_results_df):
    """Plot LLM judge correctness evaluation."""
    # Defining the chart
    _chart = alt.Chart(task_results_df).mark_bar().encode(
        x=alt.X(
            "response_llm_judge_correct_score:O", 
            title="LLM-as-a-Judge Correctness Score"
        ),
        y=alt.Y(
            "count():Q", 
            title="Count"
        ),
        color=alt.Color(
            "memory_strategy:N", 
            scale=alt.Scale(scheme='viridis'),
            legend=alt.Legend(title="Memory Strategy")
        ),
        column=alt.Column(
            "model:N", 
            title="Model Comparison"
        ),
        tooltip=[
            alt.Tooltip("model:N", title="Model"),
            alt.Tooltip("memory_strategy:N", title="Strategy"),
            alt.Tooltip("response_llm_judge_correct_score:O", title="Score"),
            alt.Tooltip("count():Q", title="Total Count")
        ]
    ).properties(
        title="LLM Judge Correctness Evaluation by Memory Strategy",
        width=200,
        height=400
    )
    _chart
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
    """Turn metrics summary subsection."""
    mo.md("### Turn Metrics Summary")
    return


@app.cell
def _(mo, task_results_df):
    """Summary statistics of turn metrics by strategy and model."""
    if task_results_df.empty:
        _ = mo.callout(mo.md("_No data available_"), kind="warn")
    else:
        _summary = (
            task_results_df.groupby(["memory_strategy", "model"])
            .agg(
                {
                    "total_call_num": ["mean", "std", "min", "max"],
                    "real_turn_num": ["mean", "std"],
                    "success_turn_num": ["mean", "std"],
                }
            )
            .round(2)
        )
        _summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Turn Count Distributions
    """)
    return


@app.cell
def _(alt, mo, task_results_df):
    """Turn count distribution by model (Altair interactive)."""
    if task_results_df.empty:
        _ = mo.callout(mo.md("_No data available_"), kind="warn")
    else:
        _dist_chart = (
            alt.Chart(task_results_df)
            .mark_bar(opacity=0.7)
            .encode(
                x=alt.X(
                    "total_call_num:Q",
                    bin=alt.Bin(maxbins=15),
                    title="Total Call Count",
                ),
                y="count()",
                color="memory_strategy:N",
            )
            .facet(column="model:N")
            .properties(width=200, height=300)
        )
        mo.ui.altair_chart(_dist_chart)
    return


@app.cell
def _(plot, plt, sns, task_results_df):
    """Turn count distribution by model (Seaborn histplot)."""
    if not task_results_df.empty:
        plot.apply_nature_style()
        _fig = plt.figure(figsize=(10, 6))
        sns.histplot(
            data=task_results_df,
            x="total_call_num",
            hue="memory_strategy",
            kde=True,
            stat="count",
            palette="deep",
        )
        plt.xlabel("Total Call Count")
        plt.ylabel("Frequency")
        plt.title("Distribution of Turn Counts")
        plt.tight_layout()
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    """Performance vs turn metrics subsection."""
    mo.md("### Turn Count vs Performance Metrics")
    return


@app.cell
def _(alt, mo, task_results_df):
    """Scatter plot: turn count vs call accuracy (Altair)."""
    if task_results_df.empty:
        _ = mo.callout(mo.md("_No data available_"), kind="warn")
    else:
        _combined = task_results_df.copy()
        _combined["call_accuracy"] = (
            _combined["correct_call_num"] / _combined["total_call_num"]
        ).fillna(0)

        _scatter = (
            alt.Chart(_combined)
            .mark_circle(size=100, opacity=0.6)
            .encode(
                x=alt.X("total_call_num:Q", title="Total Call Count"),
                y=alt.Y("call_accuracy:Q", title="Call Accuracy"),
                color="memory_strategy:N",
                tooltip=[
                    "task_id",
                    "memory_strategy",
                    "model",
                    "total_call_num",
                    "call_accuracy",
                ],
            )
            .facet(column="model:N")
            .properties(width=250, height=250)
        )
        mo.ui.altair_chart(_scatter)
    return


@app.cell
def _(plot, plt, sns, task_results_df):
    """Turn count vs success rate by strategy (Seaborn)."""
    if not task_results_df.empty:
        plot.apply_nature_style()

        _combined = task_results_df.copy()
        _combined["success"] = (_combined["status"] == "Success").astype(int)

        _fig = plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=_combined,
            x="total_call_num",
            y="success",
            hue="memory_strategy",
            style="model",
            palette="deep",
            s=100,
            alpha=0.6,
        )
        plt.ylabel("Success (1=Success, 0=Failed)")
        plt.xlabel("Total Call Count")
        plt.title("Task Success vs Turn Count")
        plt.tight_layout()
        plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    """Turn metrics by domain subsection."""
    mo.md("### Turn Count by Domain")
    return


@app.cell
def _(alt, mo, task_results_df):
    """Average turn count by domain and strategy (Altair)."""
    if task_results_df.empty:
        _ = mo.callout(mo.md("_No data available_"), kind="warn")
    else:
        _domain_agg = (
            task_results_df.groupby(["domain", "memory_strategy", "model"])
            .agg({"total_call_num": "mean", "real_turn_num": "mean"})
            .reset_index()
        )

        _chart = (
        alt.Chart(_domain_agg)
        .mark_bar()
        .encode(
            x=alt.X("domain:N", title="Domain"),
            y=alt.Y("total_call_num:Q", title="Avg Turn Count"),
            color="memory_strategy:N",
            xOffset="memory_strategy:N",
        )
        .facet(column="model:N")
        .properties(width=250, height=300)
        )
    mo.ui.altair_chart(_chart)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Metrics per Turn Count
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    """Subsection: Success by turn category."""
    mo.md("### Success Rate by Turn Category")
    return


@app.cell
def _(task_results_df):
    # df = turn_count_df[["memory_strategy", "model", "timestamp", "turns_cat", "status"]].groupby(
    turn_count_df = task_results_df
    success_count = (
        turn_count_df.groupby(
            ["model", "timestamp", "memory_strategy"],
        )["status"]
        .value_counts(dropna=False)
        .unstack(fill_value=0)
    )
    success_count
    return (turn_count_df,)


@app.cell
def _(turn_count_df):
    turn_count_grouped = (
        turn_count_df.groupby(
            ["model", "timestamp", "memory_strategy"],
        )["turns_cat"]
        .value_counts(dropna=False)
        .unstack(fill_value=0)
    )
    turn_count_grouped
    return


@app.cell
def _(plt, sns, turn_count_df):
    # 1. Filter for successful trials
    # Ensure your 'turns_cat' is an ordered category for the best visual flow
    success_df = turn_count_df[turn_count_df["status"] == "Success"].copy()

    # 2. Generate the faceted count plot
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


@app.cell(hide_code=True)
def _(BASE_DIR, dl, mo, project_dropdown):
    mo.stop(project_dropdown.value is None, None)
    token_metrics_df = dl.load_compressed_traces(BASE_DIR / project_dropdown.value)
    return (token_metrics_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Token Metrics Summary
    """)
    return


@app.cell
def _(token_metrics_df):
    summary = (
        token_metrics_df.groupby(["memory_strategy", "model"])
        .agg(
            {
                "max_token_count": ["mean", "std", "min", "max"],
                "avg_token_count": ["mean", "std"],
                "final_token_count": ["mean", "std"],
                "token_growth_rate": ["mean", "std"],
            }
        )
        .round(2)
    )
    summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Token Count Distributions
    """)
    return


@app.cell
def _(alt, mo, token_metrics_df):
    dist_chart = (
        alt.Chart(token_metrics_df)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X(
                "max_token_count:Q",
                bin=alt.Bin(maxbins=20),
                title="Max Token Count",
            ),
            y="count()",
            color="memory_strategy:N",
        )
        .facet(column="model:N")
        .properties(width=200, height=300)
    )
    mo.ui.altair_chart(dist_chart)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Token count distribution by model (Seaborn histplot).
    """)
    return


@app.cell
def _(plt, sns, token_metrics_df):
    _fig = plt.figure(figsize=(10, 6))
    sns.histplot(
        data=token_metrics_df,
        x="max_token_count",
        hue="memory_strategy",
        kde=True,
        stat="count",
        palette="deep",
    )
    plt.xlabel("Max Token Count")
    plt.ylabel("Frequency")
    plt.title("Distribution of Maximum Token Counts")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Token Count vs Performance Metrics
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Scatter plot: token count vs call accuracy (Altair).
    """)
    return


@app.cell
def _(pd, task_results_df, token_metrics_df):
    """Create dataframe, that holds both token and eval stats."""
    all_metrics_df = pd.merge(
        token_metrics_df,
        task_results_df[
            [
                "task_id",
                "memory_strategy",
                "model",
                "timestamp",
                "correct_call_num",
                "total_call_num",
                "total_turn_num",
                "response_llm_judge_correct_score",
                "response_llm_judge_complete_score",
            ]
        ],
        on=["task_id", "memory_strategy", "model", "timestamp"],
        how="left",
    )

    all_metrics_df
    return (all_metrics_df,)


@app.cell
def _(all_metrics_df, alt, mo):
    # 1. Prepare the correlation matrix
    # Note: Keeping your logic for selecting numbers and dropping specific columns
    corr_matrix = all_metrics_df.select_dtypes(include=['number']).drop(
        columns=["token_growth_rate", "min_token_count"], 
        errors='ignore'
    ).corr()

    # 2. Melt the data into long-form (Altair requirement)
    # This turns the matrix into: [Variable 1, Variable 2, Correlation]
    corr_df = corr_matrix.reset_index().melt(id_vars='index')
    corr_df.columns = ['var1', 'var2', 'correlation']

    # 3. Build the Heatmap
    base = alt.Chart(corr_df).encode(
        x=alt.X('var1:N', title=None),
        y=alt.Y('var2:N', title=None)
    )

    # The Rectangles (Heatmap cells)
    heatmap = base.mark_rect().encode(
        color=alt.Color('correlation:Q',
            scale=alt.Scale(scheme='redblue', domain=[-1, 1]),
            legend=alt.Legend(title="Pearson Corr")
        ),
        tooltip=[
            alt.Tooltip('var1:N', title='Variable A'),
            alt.Tooltip('var2:N', title='Variable B'),
            alt.Tooltip('correlation:Q', format='.2f', title='Correlation')
        ]
    )

    # The Text (Annotations)
    text = base.mark_text().encode(
        text=alt.Text('correlation:Q', format='.2f'),
        # Dynamically switch text color for readability against dark backgrounds
        color=alt.condition(
            abs(alt.datum.correlation) > 0.5,
            alt.value('white'),
            alt.value('black')
        )
    )

    # 4. Combine and Property Tune
    # In marimo, simply returning this variable will render it
    _token_corr_chart = (heatmap + text).properties(
        title='Token Metrics Correlation Heatmap',
        width=400,
        height=400
    ).configure_view(
        stroke='transparent'
    ).configure_axis(
        labelAngle=45
    )

    mo.ui.altair_chart(_token_corr_chart)
    return


@app.cell
def _(all_metrics_df, alt, mo):
    _scatter = (
        alt.Chart(all_metrics_df)
        .mark_circle(size=100, opacity=0.6)
        .encode(
            x=alt.X("max_token_count:Q", title="Max Token Count"),
            y=alt.Y("correct_call_num:Q", title="Call Accuracy"),
            color="memory_strategy:N",
            tooltip=[
                "task_id",
                "memory_strategy",
                "model",
                "max_token_count",
                "correct_call_num",
            ],
        )
        .facet(column="model:N")
        .properties(width=250, height=250)
    )
    mo.ui.altair_chart(_scatter)
    return


@app.cell
def _(all_metrics_df, alt, mo):
    _chart = alt.Chart(all_metrics_df).mark_circle(
        size=50, 
        opacity=0.6
    ).encode(
        x=alt.X("max_token_count", title="Max Token Count"),
        y=alt.Y("correct_call_num", title="Num. of correct calls"),
        color=alt.Color("memory_strategy", scale=alt.Scale(scheme="tableau10")),
        shape="model",
        # Adding tooltips makes the interactive chart much more useful
        tooltip=[
            "max_token_count",
            "avg_token_count",
            "memory_strategy", 
            "model"
        ]
    ).properties(
        title="Correct vs Maximum Token Count",
        width=750, # Approximate pixel equivalent to 10 inches
        height=250 # Approximate pixel equivalent to 6 inches
    ).interactive()

    mo.ui.altair_chart(_chart)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Token Metrics by Domain
    """)
    return


@app.cell
def _(alt, mo, token_metrics_df):
    _domain_agg = (
        token_metrics_df.groupby(["domain", "memory_strategy", "model"])
        .agg({"max_token_count": "mean", "avg_token_count": "mean"})
        .reset_index()
    )

    _chart = (
        alt.Chart(_domain_agg)
        .mark_bar()
        .encode(
            x=alt.X("domain:N", title="Domain"),
            y=alt.Y("max_token_count:Q", title="Avg Max Token Count"),
            color="memory_strategy:N",
            xOffset="memory_strategy:N",
        )
        .facet(column="model:N")
        .properties(width=250, height=300)
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Completeness / Correctness Distribution per model
    """)
    return


@app.cell
def _(alt, mo, task_results_df):
    _chart = (
        alt.Chart(task_results_df.loc[task_results_df["model"] != "claude-sonnet-4-5"])
        .mark_bar()
        .encode(
            x=alt.X("memory_strategy:N", title=""),
            y=alt.Y("real_turn_num:Q", title="Count of Results"),
            color="response_llm_judge_complete_score:O",
            tooltip=[
                alt.Tooltip("model:N"),
                alt.Tooltip("memory_strategy:N"),
                alt.Tooltip("count(response_llm_judge_complete_score):Q", title="Complete Count")
            ]
        )
        .facet(column="model:N")
        .properties(width=250, height=300,
                   title="Completeness per Model and Strategy")
        .configure_axis(
            labelFontSize=14,
            titleFontSize=16
        )
        .configure_legend(
            labelFontSize=14,
            titleFontSize=16
        )
        .configure_header( # Adjusts the 'model' facet titles
            labelFontSize=14
        )
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(alt, mo, task_results_df):
    _chart = (
        alt.Chart(task_results_df.loc[task_results_df["model"] != "claude-sonnet-4-5"])
        .mark_bar()
        .encode(
            x=alt.X("memory_strategy:N", title=""),
            y=alt.Y("real_turn_num:Q", title="Count of Results"),
            color="response_llm_judge_correct_score:O",
            tooltip=[
                alt.Tooltip("model:N"),
                alt.Tooltip("memory_strategy:N"),
                alt.Tooltip("count(real_turn_num):Q", title="Total Count")
            ]
        )
        .facet(column="model:N")
        .properties(width=250, height=300,
                   title="Completeness per Model and Strategy")
        .configure_axis(
            labelFontSize=14,
            titleFontSize=16
        )
        .configure_legend(
            labelFontSize=14,
            titleFontSize=16
        )
        .configure_header( # Adjusts the 'model' facet titles
            labelFontSize=14
        )
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Task Error Analysis
    """)
    return


@app.cell
def _(task_results_df):
    task_results_df.loc[task_results_df["model"] == "gpt-4-1"].message_error_reasoning.value_counts()
    return


@app.cell
def _(task_results_df):
    task_results_df.loc[task_results_df["model"] == "gemini-2-5-pro"].message_error_type.value_counts()
    return


@app.cell
def _():
    return


@app.cell
def _(alt, mo, task_results_df):
    _agg_df = (
        task_results_df
        .groupby(["model", "memory_strategy", "domain", "message_error_type"])
        .size()
        .reset_index(name="error_count")
    )

    # 2. Plot the aggregated data
    _error_type_chart = {}
    _models = sorted(_agg_df["model"].unique())

    for _model in _models:
        _subset = _agg_df[_agg_df["model"] == _model]
        if _subset.empty:
            continue
        
        _chart = (
            alt.Chart(_subset)
            .mark_bar()
            .encode(
                x=alt.X("message_error_type:N", title="Error Type", sort="-y"),
                # Use our explicitly calculated count column!
                y=alt.Y("error_count:Q", title="Count of Error Type"),
                # color=alt.Color("domain:N", title="Domain"),
                # xOffset="domain:N",
                tooltip=["model", "memory_strategy", "domain", "message_error_type", "error_count"]
            )
            .facet(
                column=alt.Column("memory_strategy:N", title="Memory Strategy"),
            )
            .properties(title=_model.replace("_", " ").title())
            .resolve_scale(y="independent")
        )
        _error_type_chart[_model] = mo.ui.altair_chart(_chart)

    mo.vstack(
        [
            mo.ui.tabs(_error_type_chart)
            if _error_type_chart
            else mo.md("_No charts available_")
        ]
    )
    return


@app.cell
def _(alt, mo, task_results_df):
    _agg_df = (
        task_results_df
        .groupby(["model", "memory_strategy", "domain", "message_error_type"])
        .size()
        .reset_index(name="error_count")
    )

    # 2. Plot the aggregated data
    error_type_chart_memory_strategy = {}
    _memory_strategies = sorted(_agg_df["memory_strategy"].unique())

    for _memory_strategy in _memory_strategies:
        _subset = _agg_df[_agg_df["memory_strategy"] == _memory_strategy]
        if _subset.empty:
            continue
        
        _chart = (
            alt.Chart(_subset)
            .mark_bar()
            .encode(
                x=alt.X("message_error_type:N", title="Error Type", sort="-y"),
                # Use our explicitly calculated count column!
                y=alt.Y("error_count:Q", title="Count of Error Type"),
                # color=alt.Color("domain:N", title="Domain"),
                # xOffset="domain:N",
                tooltip=["model", "memory_strategy", "domain", "message_error_type", "error_count"]
            )
            .facet(
                column=alt.Column("model:N", title="LLM Model"),
            )
            .properties(title=_memory_strategy.replace("_", " ").title())
            .resolve_scale(y="independent")
        )
        error_type_chart_memory_strategy[_memory_strategy] = mo.ui.altair_chart(_chart)

    mo.vstack(
        [
            mo.ui.tabs(error_type_chart_memory_strategy)
            if error_type_chart_memory_strategy
            else mo.md("_No charts available_")
        ]
    )
    return


@app.cell
def _():
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
