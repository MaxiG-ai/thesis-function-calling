import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo
    from pathlib import Path
    import pandas as pd
    import altair as alt
    import thesis_graphics as tg

    # Configuration
    BASE_DIR = Path("results/cfb")
    OUTPUT_ROOT = Path("thesis_assets")

    return mo, Path, pd, alt, tg, BASE_DIR, OUTPUT_ROOT


@app.cell
def __(mo, BASE_DIR):
    def get_projects() -> list[str]:
        if not BASE_DIR.exists():
            return []
        return sorted([d.name for d in BASE_DIR.iterdir() if d.is_dir()])

    projects = get_projects()
    return (projects,)


@app.cell
def __(mo, projects):
    mo.md("# Experiment Analysis")


@app.cell
def __(mo, projects):
    project_dropdown = mo.ui.dropdown(
        options=projects,
        label="Project",
        allow_select_none=True,
        value=None,
    )
    project_dropdown


@app.cell
def __(mo, project_dropdown, BASE_DIR, tg):
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


@app.cell
def __(mo, metrics_df, project_dropdown):
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


@app.cell
def __(mo, metrics_df, alt):
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


@app.cell
def __(mo, metrics_df):
    """Create results table for display."""
    # Pivot to get strategy x model table (average across domains)
    table_df = metrics_df.pivot_table(
        index="memory_strategy",
        columns=["model", "metric"],
        values="value",
        aggfunc="mean",
    ).round(2)

    # Flatten column names for display
    table_df.columns = [f"{m}_{metric}" for m, metric in table_df.columns]
    table_df = table_df.reset_index()

    results_table = mo.ui.table(
        table_df,
        selection=None,
        pagination=False,
    )

    return results_table, table_df


@app.cell
def __(mo, results_table):
    """Display results table."""
    mo.vstack(
        [
            mo.md("### Results Table"),
            mo.md("*Select cells and copy to clipboard*"),
            results_table,
        ]
    )


@app.cell
def __(mo, altair_charts):
    """Display interactive charts."""
    mo.vstack(
        [
            mo.md("### Interactive Charts"),
            mo.ui.tabs(altair_charts)
            if altair_charts
            else mo.md("No charts available"),
        ]
    )


@app.cell
def __(mo):
    save_btn = mo.ui.run_button(label="Save plots and tables to disk")
    return (save_btn,)


@app.cell
def __(mo, save_btn):
    mo.hstack([mo.md("### Export"), save_btn], justify="start")


@app.cell
def __(
    mo,
    save_btn,
    metrics_df,
    project_dropdown,
    OUTPUT_ROOT,
    tg,
):
    mo.stop(
        not save_btn.value,
        mo.md("Click the button to save all plots and tables as PDFs and CSVs."),
    )

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
