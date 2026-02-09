import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    from pathlib import Path
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    from great_tables import GT
    return GT, Path, pl, plt, sns


@app.cell
def _():
    EXPERIMENT_RUN="results/cfb/gpt41mini_memory_test/20251215_1633"
    EXPERIMENT_NAME = EXPERIMENT_RUN.lstrip("results/cfb/")
    return EXPERIMENT_NAME, EXPERIMENT_RUN


@app.cell
def _(EXPERIMENT_NAME):
    print(EXPERIMENT_NAME)
    return


@app.cell
def _(Path, pl):
    def read_exp_to_polars(base_dir: str) -> pl.DataFrame:
        p = Path(base_dir)
        json_files = [x for x in p.rglob('metrics*.json')]

        if not json_files:
            return pl.DataFrame()

        dfs = []
        # Identify keys that should stay as index columns
        index_cols = [
            "experiment_name", "run_timestamp", "memory_method", "model_name", 
            "overall_success", "overall_call_acc", "complete_score_avg", "correct_score_avg"
        ]

        for path in json_files:
            parts = path.parts
            metadata = {
                "experiment_name": parts[-5],
                "run_timestamp": parts[-4],
                "memory_method": parts[-3],
                "model_name": parts[-2]
            }

            try:
                df = pl.read_json(path)
                df = df.with_columns([pl.lit(v).alias(k) for k, v in metadata.items()])

                # 1. Unpivot domain metrics into a temporary struct column
                # Replacement for .melt() -> .unpivot()
                df_long = df.unpivot(
                    index=index_cols,
                    on=["domain_success_rate", "domain_turn_acc", "domain_call_acc"],
                    variable_name="metric_type",
                    value_name="domain_values"
                )

                # 2. Flatten the domain keys and unpivot them
                # This handles the "Hotels", "Attraction", etc.
                df_flat = (
                    df_long.unnest("domain_values")
                    .unpivot(
                        index=index_cols + ["metric_type"],
                        variable_name="domain",
                        value_name="score"
                    )
                    .drop_nulls("score")
                )

                # 3. Pivot the metric_type back to columns to minimize metadata duplication
                # This gives you: [metadata] | domain | domain_success_rate | domain_turn_acc | ...
                df_final = df_flat.pivot(
                    on="metric_type",
                    index=index_cols + ["domain"],
                    values="score"
                )

                dfs.append(df_final)
            except Exception as e:
                print(f"Error processing {path}: {e}")

        return pl.concat(dfs, how="diagonal") if dfs else pl.DataFrame()
    return (read_exp_to_polars,)


@app.cell
def _(EXPERIMENT_RUN, read_exp_to_polars):
    df = read_exp_to_polars(EXPERIMENT_RUN)
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(cols_keep, df):
    df[cols_keep].unique()
    return


@app.cell
def _(EXPERIMENT_NAME, GT, Path, cols_keep, df, pl):
    def df_to_latex(df):
        latex_output = (
        GT(df)
        .fmt_number(columns=pl.col(pl.Float64), decimals=2)
        .as_latex()
        )
        print(latex_output)
        latex_path = Path(f"thesis_assets/{EXPERIMENT_NAME}/exp_result_table.txt")
        with open(latex_path, "w") as f:
            f.write(latex_output)
            print(f"Written to: {latex_path}")

    df_to_latex(df[cols_keep].unique())
    return


@app.cell
def _(EXPERIMENT_NAME, Path, df, plt, sns):
    # 1. Deduplicate to get one row per experiment configuration
    # We only want the unique 'overall' scores for this specific comparison
    cols_keep = [
      "memory_method",
      "model_name",
      "overall_success",
      "overall_call_acc",
      "complete_score_avg",
      "correct_score_avg",
    ]

    metrics = [
      "overall_success",
      "overall_call_acc",
      "complete_score_avg",
      "correct_score_avg",
    ]

    df_unique = df[cols_keep]

    # 2. Setup paper aesthetics
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams['font.family'] = 'serif'

    # 3. Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        sns.barplot(
            data=df_unique,
            x='memory_method',
            y=metric,
            hue='model_name',
            ax=axes[i],
            palette='viridis',
            edgecolor='0.2'
        )

        # Clean up labels for a paper
        title = metric.replace('_', ' ').title()
        axes[i].set_title(title, fontweight='bold', fontsize=16)
        axes[i].set_xlabel('Memory Strategy', fontsize=12)
        axes[i].set_ylabel('Score', fontsize=12)
        axes[i].legend(title='Model', loc='best', frameon=True)

    plt.tight_layout()
    output_path = Path(f"thesis_assets/{EXPERIMENT_NAME}/experiment_comparison_paper.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    return (cols_keep,)


@app.cell
def _(EXPERIMENT_NAME):
    print(f"thesis_assets/{EXPERIMENT_NAME}/experiment_comparison_paper.png")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
