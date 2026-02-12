"""Thesis graphics utilities for Nature-quality experiment visualizations.

Provides functions to load metrics, create publication-ready plots using seaborn,
and save results as CSV tables and PDF figures.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

# Nature journal style settings
NATURE_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.titlesize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.0,
}

# Nature single column width ~89mm, double column ~183mm
FIG_WIDTH_SINGLE = 3.5  # inches (~89mm)
FIG_WIDTH_DOUBLE = 7.2  # inches (~183mm)
FIG_HEIGHT_SINGLE = 2.5
FIG_HEIGHT_DOUBLE = 4.5


def _apply_nature_style() -> None:
    """Apply Nature journal styling to matplotlib."""
    plt.rcParams.update(NATURE_RC)
    sns.set_theme(style="whitegrid", rc=NATURE_RC)


def _sanitize_filename(value: str) -> str:
    """Sanitize string for use in filenames (no dots or special chars)."""
    value = re.sub(r"\s+", "_", value.strip())
    value = value.replace(".", "_")
    value = re.sub(r"[^a-zA-Z0-9_-]+", "", value)
    return value or "item"


def _format_metric_label(metric: str) -> str:
    """Format metric name for display (replace underscores, capitalize)."""
    # Handle nested metrics like domain_turn_acc.Hotels
    if "." in metric:
        parts = metric.split(".")
        base = parts[0].replace("_", " ").title()
        domain = parts[1]
        return f"{base} ({domain})"
    return metric.replace("_", " ").title()


def load_metrics_long(base_dir: Path, project: str, timestamp: str) -> pd.DataFrame:
    """Load metrics from a single timestamp into a long-format DataFrame.

    Args:
        base_dir: Root directory containing experiment results.
        project: Project name (subdirectory).
        timestamp: Timestamp folder name.

    Returns:
        DataFrame with columns: memory_strategy, model, metric, value, domain.
        Domain is extracted from nested metrics (e.g., domain_turn_acc.Hotels).
    """
    exp_path = base_dir / project / timestamp
    rows: list[dict[str, object]] = []
    if not exp_path.exists():
        return pd.DataFrame()

    for strategy_dir in exp_path.iterdir():
        if not strategy_dir.is_dir():
            continue
        for model_dir in strategy_dir.iterdir():
            if not model_dir.is_dir():
                continue
            json_files = list(model_dir.glob("metrics_*.json"))
            if not json_files:
                continue
            metrics_file = json_files[0]
            try:
                with metrics_file.open("r") as handle:
                    metrics = json.load(handle)
            except json.JSONDecodeError:
                continue

            # Process metrics, extracting domain info
            for key, value in metrics.items():
                if isinstance(value, dict):
                    # Nested metric (e.g., domain_turn_acc with domain keys)
                    for domain, domain_value in value.items():
                        if isinstance(domain_value, (int, float)):
                            rows.append(
                                {
                                    "memory_strategy": strategy_dir.name,
                                    "model": model_dir.name,
                                    "metric": key,
                                    "domain": domain,
                                    "value": float(domain_value),
                                }
                            )
                elif isinstance(value, (int, float)):
                    rows.append(
                        {
                            "memory_strategy": strategy_dir.name,
                            "model": model_dir.name,
                            "metric": key,
                            "domain": "Overall",
                            "value": float(value),
                        }
                    )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_records(rows)


def load_metrics_project(base_dir: Path, project: str) -> pd.DataFrame:
    """Load all metrics for a project, aggregating across all timestamps.

    If multiple timestamps contain the same strategy/model combination,
    they are disambiguated by prefixing with a short timestamp identifier.

    Args:
        base_dir: Root directory containing experiment results.
        project: Project name (subdirectory).

    Returns:
        DataFrame with columns: memory_strategy, model, metric, value, domain.
        Strategy/model names may include timestamp prefix if duplicates exist.
    """
    project_path = base_dir / project
    if not project_path.exists():
        return pd.DataFrame()

    # Collect all timestamp directories
    timestamps = sorted([d.name for d in project_path.iterdir() if d.is_dir()])
    if not timestamps:
        return pd.DataFrame()

    # First pass: collect all data with timestamp info to detect duplicates
    all_rows: list[dict[str, object]] = []
    combo_counts: dict[tuple[str, str], int] = {}  # (strategy, model) -> count

    for ts in timestamps:
        ts_path = project_path / ts
        for strategy_dir in ts_path.iterdir():
            if not strategy_dir.is_dir():
                continue
            for model_dir in strategy_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                json_files = list(model_dir.glob("metrics_*.json"))
                if not json_files:
                    continue
                metrics_file = json_files[0]
                try:
                    with metrics_file.open("r") as handle:
                        metrics = json.load(handle)
                except json.JSONDecodeError:
                    continue

                combo = (strategy_dir.name, model_dir.name)
                combo_counts[combo] = combo_counts.get(combo, 0) + 1

                # Process metrics, extracting domain info
                for key, value in metrics.items():
                    if isinstance(value, dict):
                        for domain, domain_value in value.items():
                            if isinstance(domain_value, (int, float)):
                                all_rows.append(
                                    {
                                        "timestamp": ts,
                                        "memory_strategy": strategy_dir.name,
                                        "model": model_dir.name,
                                        "metric": key,
                                        "domain": domain,
                                        "value": float(domain_value),
                                    }
                                )
                    elif isinstance(value, (int, float)):
                        all_rows.append(
                            {
                                "timestamp": ts,
                                "memory_strategy": strategy_dir.name,
                                "model": model_dir.name,
                                "metric": key,
                                "domain": "Overall",
                                "value": float(value),
                            }
                        )

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(all_rows)

    # Find combos that appear more than once (need disambiguation)
    duplicated_combos = {combo for combo, count in combo_counts.items() if count > 1}

    if duplicated_combos:
        # Add timestamp prefix only for duplicated combos
        def maybe_prefix(row: pd.Series) -> str:
            combo = (row["memory_strategy"], row["model"])
            if combo in duplicated_combos:
                # Use short timestamp (first 8 chars: YYYYMMDD)
                ts_short = row["timestamp"][:8]
                return f"{ts_short}_{row['memory_strategy']}"
            return row["memory_strategy"]

        df["memory_strategy"] = df.apply(maybe_prefix, axis=1)

    # Drop timestamp column (no longer needed)
    df = df.drop(columns=["timestamp"])
    return df


def save_metric_tables(
    metrics_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Save metrics as CSV tables (long format only)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # Save long format only
    long_path = output_dir / "all_metrics.csv"
    metrics_df.to_csv(long_path, index=False)
    saved.append(long_path)

    return saved


def save_results_table(metrics_df: pd.DataFrame, output_dir: Path) -> Path | None:
    """Save comprehensive results table with all metrics."""
    if metrics_df.empty:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate by strategy/model/metric (mean across domains)
    agg_df = metrics_df.groupby(["memory_strategy", "model", "metric"], as_index=False)[
        "value"
    ].mean()

    table = agg_df.pivot_table(
        index="memory_strategy",
        columns=["model", "metric"],
        values="value",
    )
    table_path = output_dir / "results_table.csv"
    table.to_csv(table_path)
    return table_path


def plot_model_comparison(
    metrics_df: pd.DataFrame,
    metric: str,
) -> Figure | None:
    """Create grouped bar chart comparing models, with domain as hue.

    This is the overview grid plot for display in the notebook.
    """
    _apply_nature_style()
    subset = metrics_df.loc[metrics_df["metric"] == metric].copy()
    if subset.empty:
        return None

    n_strategies = subset["memory_strategy"].nunique()
    n_domains = subset["domain"].nunique()

    # Determine figure size based on data
    fig_width = min(FIG_WIDTH_DOUBLE, FIG_WIDTH_SINGLE * max(2, n_strategies / 2))
    fig_height = FIG_HEIGHT_DOUBLE if n_domains > 1 else FIG_HEIGHT_SINGLE

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Use domain as hue for grouped bars
    if n_domains > 1:
        sns.barplot(
            data=subset,
            x="model",
            y="value",
            hue="domain",
            ax=ax,
            palette="deep",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.legend(
            title="Domain",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
        )
    else:
        sns.barplot(
            data=subset,
            x="model",
            y="value",
            hue="memory_strategy",
            ax=ax,
            palette="deep",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.legend(
            title="Memory Strategy",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
        )

    ax.set_title(_format_metric_label(metric), fontweight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel(_format_metric_label(metric))
    ax.tick_params(axis="x", rotation=30)

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=6, padding=2)

    fig.tight_layout()
    return fig


def plot_strategy_comparison(
    metrics_df: pd.DataFrame,
    metric: str,
) -> Figure | None:
    """Create grouped bar chart comparing strategies, with domain as hue.

    This is the overview grid plot for display in the notebook.
    """
    _apply_nature_style()
    subset = metrics_df.loc[metrics_df["metric"] == metric].copy()
    if subset.empty:
        return None

    n_models = subset["model"].nunique()
    n_domains = subset["domain"].nunique()

    fig_width = min(FIG_WIDTH_DOUBLE, FIG_WIDTH_SINGLE * max(2, n_models / 2))
    fig_height = FIG_HEIGHT_DOUBLE if n_domains > 1 else FIG_HEIGHT_SINGLE

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if n_domains > 1:
        sns.barplot(
            data=subset,
            x="memory_strategy",
            y="value",
            hue="domain",
            ax=ax,
            palette="deep",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.legend(
            title="Domain",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
        )
    else:
        sns.barplot(
            data=subset,
            x="memory_strategy",
            y="value",
            hue="model",
            ax=ax,
            palette="deep",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.legend(
            title="Model",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
        )

    ax.set_title(_format_metric_label(metric), fontweight="bold")
    ax.set_xlabel("Memory Strategy")
    ax.set_ylabel(_format_metric_label(metric))
    ax.tick_params(axis="x", rotation=30)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=6, padding=2)

    fig.tight_layout()
    return fig


def save_all_plots(
    metrics_df: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Save all individual plots as PDFs for the paper.

    Creates one plot per metric showing model comparison with domain hue,
    and one plot per metric showing strategy comparison with domain hue.
    """
    _apply_nature_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    metrics = sorted(metrics_df["metric"].unique())

    for metric in metrics:
        subset = metrics_df.loc[metrics_df["metric"] == metric].copy()
        if subset.empty:
            continue

        n_domains = subset["domain"].nunique()
        fig_width = FIG_WIDTH_DOUBLE
        fig_height = FIG_HEIGHT_DOUBLE if n_domains > 1 else FIG_HEIGHT_SINGLE

        # Model comparison plot
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        if n_domains > 1:
            sns.barplot(
                data=subset,
                x="model",
                y="value",
                hue="domain",
                ax=ax,
                palette="deep",
                edgecolor="black",
                linewidth=0.5,
            )
            ax.legend(
                title="Domain",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                frameon=False,
            )
        else:
            sns.barplot(
                data=subset,
                x="model",
                y="value",
                hue="memory_strategy",
                ax=ax,
                palette="deep",
                edgecolor="black",
                linewidth=0.5,
            )
            ax.legend(
                title="Strategy",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                frameon=False,
            )

        ax.set_title(_format_metric_label(metric), fontweight="bold")
        ax.set_xlabel("Model")
        ax.set_ylabel(_format_metric_label(metric))
        ax.tick_params(axis="x", rotation=30)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f", fontsize=6, padding=2)
        fig.tight_layout()

        filename_base = f"model_comparison_{_sanitize_filename(metric)}"
        pdf_path = output_dir / f"{filename_base}.pdf"
        png_path = output_dir / f"{filename_base}.png"
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.extend([pdf_path, png_path])

        # Strategy comparison plot
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        if n_domains > 1:
            sns.barplot(
                data=subset,
                x="memory_strategy",
                y="value",
                hue="domain",
                ax=ax,
                palette="deep",
                edgecolor="black",
                linewidth=0.5,
            )
            ax.legend(
                title="Domain",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                frameon=False,
            )
        else:
            sns.barplot(
                data=subset,
                x="memory_strategy",
                y="value",
                hue="model",
                ax=ax,
                palette="deep",
                edgecolor="black",
                linewidth=0.5,
            )
            ax.legend(
                title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False
            )

        ax.set_title(_format_metric_label(metric), fontweight="bold")
        ax.set_xlabel("Memory Strategy")
        ax.set_ylabel(_format_metric_label(metric))
        ax.tick_params(axis="x", rotation=30)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f", fontsize=6, padding=2)
        fig.tight_layout()

        filename_base = f"strategy_comparison_{_sanitize_filename(metric)}"
        pdf_path = output_dir / f"{filename_base}.pdf"
        png_path = output_dir / f"{filename_base}.png"
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.extend([pdf_path, png_path])

    return saved
