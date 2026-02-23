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
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Import helper modules
    from helpers import data_loading as dl
    from helpers import plotting as plot

    # Configuration
    BASE_DIR = Path("results/cfb")
    OUTPUT_ROOT = Path("thesis_assets")
    return BASE_DIR, alt, dl, mo, np, pd, plot, plt, sns


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
def _(mo, projects):
    """Project selection dropdown."""
    project_dropdown = mo.ui.dropdown(
        options=projects,
        label="Project",
        allow_select_none=True,
        value=None,
    )

    mo.vstack(
        [
         mo.md("""
    Select a project from the dropdown below to load and analyze experiment results.
    """), 
        project_dropdown
        ]
    )

    return (project_dropdown,)


@app.cell(hide_code=True)
def _(BASE_DIR, dl, mo, project_dropdown):
    """Load all data for selected project."""
    # Stop if no selection made
    mo.stop(
        project_dropdown.value is None,
        mo.md("No project selected!"),
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

    task_results_df.drop(columns="gen_convs", inplace=True)
    return (task_results_df,)


@app.cell(hide_code=True)
def _(metrics_long_df, mo, project_dropdown):
    """Display project overview statistics."""
    _strategies = sorted(metrics_long_df["memory_strategy"].unique())
    _models = sorted(metrics_long_df["model"].unique())
    _n_metrics = metrics_long_df["metric"].nunique()

    mo.md(f"""
    **{project_dropdown.value}** - Experiment Overview

    - **Strategies:** {len(_strategies)}
    - **Models:** {len(_models)}
    - **Metrics:** {_n_metrics}
    - **Data aggregated across all timestamps**
        """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Interactive Metric Visualizations
    """)
    return


@app.cell(hide_code=True)
def _(alt, metrics_long_df, mo):
    """Create and display Altair charts for each metric."""
    _altair_metrics_charts = {}
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
        _altair_metrics_charts[_metric] = mo.ui.altair_chart(_chart)
    

    mo.ui.tabs(_altair_metrics_charts) if _altair_metrics_charts else mo.md("_No charts available_")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LLM-as-a-Judge Correctness Evaluation
    """)
    return


@app.cell
def _(alt, metrics_long_df, mo):
    """Plot LLM judge correctness evaluation."""

    _chart_completeness = alt.Chart(metrics_long_df.loc[metrics_long_df.metric == "correct_score_avg"]).mark_bar().encode(
        x=alt.X(
            "memory_strategy:N",
            title=""
        ),
        y=alt.Y(
            "value",
            title="Average LLM-as-a-Judge Correct Score"
        ),
        tooltip=[
            alt.Tooltip("model:N", title="Model"),
            alt.Tooltip("memory_strategy:N", title="Strategy"),
            alt.Tooltip("value:Q", title="Score"),
        ]
    ).facet(
        column=alt.Column("model:N", title="LLM Model"),
    ).properties(
        title="LLM Judge Correct Evaluation by Memory Strategy",
        width=200,
        height=400
    ).configure_axis(
        labelFontSize=14,
        titleFontSize=16
    ).configure_legend(
        labelFontSize=14,
        titleFontSize=16
    ).configure_header( # Adjusts the 'model' facet titles
        labelFontSize=14
    )

    mo.ui.altair_chart(_chart_completeness)
    return


@app.cell(hide_code=True)
def _(mo):
    """Subsection: Completeness evaluation."""
    mo.md("### Completeness Scores")
    return


@app.cell
def _(metrics_long_df):
    metrics_long_df.columns
    return


@app.cell
def _(alt, metrics_long_df, mo):
    """Plot LLM judge completeness evaluation."""

    _chart_completeness = alt.Chart(metrics_long_df.loc[metrics_long_df.metric == "complete_score_avg"]).mark_bar().encode(
        x=alt.X(
            "memory_strategy:N",
            title=""
        ),
        y=alt.Y(
            "value",
            title="Average LLM-as-a-Judge Completion Score"
        ),
        tooltip=[
            alt.Tooltip("model:N", title="Model"),
            alt.Tooltip("memory_strategy:N", title="Strategy"),
            alt.Tooltip("value:Q", title="Score"),
        ]
    ).facet(
        column=alt.Column("model:N", title="LLM Model"),
    ).properties(
        title="LLM Judge Completeness Evaluation by Memory Strategy",
        width=200,
        height=400
    ).configure_axis(
        labelFontSize=14,
        titleFontSize=16
    ).configure_legend(
        labelFontSize=14,
        titleFontSize=16
    ).configure_header( # Adjusts the 'model' facet titles
        labelFontSize=14
    )

    mo.ui.altair_chart(_chart_completeness)
    return


@app.cell
def _(plot, plt, sns, task_results_df):
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
def _(alt, mo, strategy, task_results_df):
    """Turn count distribution by model & strategy (Altair interactive)."""

    _chart_tabs = {}

    # Iterate through unique memory strategies to build our tabs
    for _strategy in task_results_df['memory_strategy'].dropna().unique():
        _strat_df = task_results_df[task_results_df['memory_strategy'] == strategy]
        _dist_chart = (
                alt.Chart(task_results_df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "total_call_num:Q",
                        bin=alt.Bin(maxbins=15),
                        title="Total Call Count",
                    ),
                    y="count()",
                )
                .facet(column="model:N")
                .properties(width=200, height=300)
            )
        # Wrap with Marimo UI 
        _chart_tabs[str(_strategy)] = mo.ui.altair_chart(_dist_chart)

    # Display as Marimo tabs
    _tabs = mo.ui.tabs(_chart_tabs)
    _tabs
    return


@app.cell(hide_code=True)
def _(mo):
    """Performance vs turn metrics subsection."""
    mo.md("### Turn Count vs Performance Metrics")
    return


@app.cell
def _(alt, mo, np, task_results_df):
    """Turn count vs success rate by strategy (Seaborn)."""
    # Safely calculate call accuracy to prevent division by zero!
    task_results_df['call_accuracy'] = np.where(
        task_results_df['total_call_num'] > 0, 
        task_results_df['correct_call_num'] / task_results_df['total_call_num'], 
        0
    )

    chart_tabs = {}

    # Iterate through unique memory strategies to build our tabs
    for strategy in task_results_df['memory_strategy'].dropna().unique():
        _strat_df = task_results_df[task_results_df['memory_strategy'] == strategy]
    
        # Building the Altair chart
        scatter = alt.Chart(_strat_df).mark_circle(size=100, opacity=0.7).encode(
            x=alt.X('total_turn_num:Q', 
                title='Total Turn Count', 
                axis=alt.Axis(tickMinStep=1, format='d') # Forces integer ticks
            ),
            y=alt.Y('call_accuracy:Q', 
                title='Call Accuracy', 
                scale=alt.Scale(domain=[-0.05, 1.05])
           ),
            color=alt.Color('status:N', 
                    title='Status'),
            tooltip=[
                alt.Tooltip('task_id:N', title='Task ID'),
                alt.Tooltip('model:N', title='Model'),
                alt.Tooltip('domain:N', title='Domain'),
                alt.Tooltip('total_turn_num:Q', title='Total Turns'),
                alt.Tooltip('correct_call_num:Q', title='Correct Calls'),
                alt.Tooltip('total_call_num:Q', title='Total Calls'),
                alt.Tooltip('call_accuracy:Q', title='Accuracy', format='.2%'),
                alt.Tooltip('message_error_type:N', title='Error Type')
            ]
        ).properties(
            title=f'Agent Turn Count vs Call Accuracy ({strategy})',
            width=700,
            height=400
        ).configure_axis(
            titleFontSize=16,
            labelFontSize=14
        ).configure_title(
            fontSize=16
        ).configure_legend(
            titleFontSize=16,
            labelFontSize=14
        ).interactive() # Zooming and panning enabled
    
        # Wrap with Marimo UI 
        chart_tabs[str(strategy)] = mo.ui.altair_chart(scatter)

    # Display as Marimo tabs
    _tabs = mo.ui.tabs(chart_tabs)
    _tabs
    return (strategy,)


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


@app.cell(hide_code=True)
def _(alt, mo, task_results_df):
    _agg_df = (
        task_results_df
        .loc[task_results_df.message_error_type != "no error detected"]
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
            mo.md("Filtered out `no error detected`"),
            mo.ui.tabs(_error_type_chart) if _error_type_chart else mo.md("_No charts available_")
        ]
    )
    return


@app.cell(hide_code=True)
def _(alt, mo, task_results_df):
    _agg_df = (
        task_results_df
        .loc[task_results_df.message_error_type != "no error detected"]
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
        
            mo.md("Filtered out `no error detected`"),
            mo.ui.tabs(error_type_chart_memory_strategy) if error_type_chart_memory_strategy else mo.md("_No charts available_")
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Effect of max Tokens metric on error count
    """)
    return


@app.cell
def _(task_results_df, token_metrics_df):
    print(token_metrics_df.columns)
    print(task_results_df.columns)
    return


@app.cell
def _(task_results_df, token_metrics_df):
    """Combining the max tokens to the task_results_df"""
    task_token_df = task_results_df.join(
        other=token_metrics_df.set_index(['task_id', 'memory_strategy', 'model', 'timestamp']),
        on=['task_id', 'memory_strategy', 'model', 'timestamp'],
        lsuffix="_l"
    )
    return (task_token_df,)


@app.cell
def _(task_token_df):
    task_token_df.columns
    return


@app.cell
def _(alt, mo, task_token_df):
    _filtered_df = task_token_df[task_token_df.message_error_type != "no error detected"].loc[task_token_df.model != "claude-sonnet-4-5"]

    _error_type_chart = {}
    _metrics = [
        'max_token_count', 'avg_token_count', 'final_token_count', 'num_steps', 'total_input_tokens', 
        'avg_compression_ratio', 'max_compression_ratio', 'token_growth_rate'
    ]

    _base_columns = [
        'task_id', 'memory_strategy', 'model', 'timestamp', 
        'status', 'message_error_type', 'message_error_reasoning', 'domain'
    ]

    for _metric in _metrics:
        # Ensure the metric exists in columns to avoid the previous ValueError
        _cols_to_extract = _base_columns + [_metric]
        _subset = _filtered_df[_cols_to_extract]
    
        if _subset.empty:
            continue

        # 2. Optimized Chart Construction
        _base = (
            alt.Chart(_subset)
            .mark_bar()
            .encode(
                # Sorting the Y-axis in descending order of the X-axis metric
                y=alt.Y("message_error_type:N", 
                        title="Error Type", 
                        sort="-x"), 
                x=alt.X(f"mean({_metric}):Q", 
                        title=f"Avg {_metric.replace('_', ' ').title()}"),
                tooltip=["model", "memory_strategy", "message_error_type"]
            )
            .properties(width=250, height=250) # Smaller base size for grid layout
        )

        # 3. Proper Faceting: Use facet() for both Row and Column
        _final_chart = _base.facet(
            column=alt.Column("model:N", title="Models"),
            row=alt.Row("memory_strategy:N", title="Memory Strategy")
        ).properties(
            title=f"Error Analysis by {_metric.replace('_', ' ').title()}"
        ).resolve_scale(
            y="independent", 
            x="independent"
        )

        _error_type_chart[_metric] = mo.ui.altair_chart(_final_chart)

    # Render
    mo.vstack([
        mo.md(f"### 📊 Error Analysis (Filtered: {len(_filtered_df)} samples)"),
        mo.ui.tabs(_error_type_chart) if _error_type_chart else mo.md("_No data found for the selected metrics._")
    ])
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
