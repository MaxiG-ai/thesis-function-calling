"""Plotting utilities for Nature-quality experiment visualizations.

Provides reusable functions to create publication-ready plots using seaborn
and matplotlib, with Nature journal styling.
"""

from __future__ import annotations

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


def apply_nature_style() -> None:
    """Apply Nature journal styling to matplotlib and seaborn."""
    plt.rcParams.update(NATURE_RC)
    sns.set_theme(style="whitegrid", rc=NATURE_RC)


def sanitize_filename(value: str) -> str:
    """Sanitize string for use in filenames (no dots or special chars).

    Args:
        value: String to sanitize

    Returns:
        Sanitized string safe for use in filenames
    """
    value = re.sub(r"\s+", "_", value.strip())
    value = value.replace(".", "_")
    value = re.sub(r"[^a-zA-Z0-9_-]+", "", value)
    return value or "item"


def format_metric_label(metric: str) -> str:
    """Format metric name for display (replace underscores, capitalize).

    Args:
        metric: Metric name, e.g., "domain_turn_acc" or "domain_turn_acc.Hotels"

    Returns:
        Formatted label, e.g., "Domain Turn Acc" or "Domain Turn Acc (Hotels)"
    """
    # Handle nested metrics like domain_turn_acc.Hotels
    if "." in metric:
        parts = metric.split(".")
        base = parts[0].replace("_", " ").title()
        domain = parts[1]
        return f"{base} ({domain})"
    return metric.replace("_", " ").title()


def plot_metric_comparison(
    df: pd.DataFrame,
    metric: str,
    x_col: str,
    hue_col: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure | None:
    """Create grouped bar chart comparing a metric across groups.

    Args:
        df: DataFrame with columns: metric, value, and grouping columns
        metric: Metric name to filter on
        x_col: Column to use for x-axis (e.g., "model" or "memory_strategy")
        hue_col: Column to use for grouping/color (e.g., "domain", "memory_strategy", "model")
        figsize: Figure size in inches (width, height), or None for auto-sizing

    Returns:
        Matplotlib Figure object, or None if no data
    """
    apply_nature_style()
    subset = df.loc[df["metric"] == metric].copy()
    if subset.empty:
        return None

    # Auto-size figure if not provided
    if figsize is None:
        n_groups = subset[x_col].nunique()
        if hue_col:
            n_hues = subset[hue_col].nunique()
            fig_height = FIG_HEIGHT_DOUBLE if n_hues > 1 else FIG_HEIGHT_SINGLE
        else:
            fig_height = FIG_HEIGHT_SINGLE
        fig_width = min(FIG_WIDTH_DOUBLE, FIG_WIDTH_SINGLE * max(2, n_groups / 2))
        figsize = (fig_width, fig_height)

    fig, ax = plt.subplots(figsize=figsize)

    # Create bar plot
    sns.barplot(
        data=subset,
        x=x_col,
        y="value",
        hue=hue_col,
        ax=ax,
        palette="deep",
        edgecolor="black",
        linewidth=0.5,
    )

    # Legend
    if hue_col:
        ax.legend(
            title=format_metric_label(hue_col),
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False,
        )

    # Labels
    ax.set_title(format_metric_label(metric), fontweight="bold")
    ax.set_xlabel(format_metric_label(x_col))
    ax.set_ylabel(format_metric_label(metric))
    ax.tick_params(axis="x", rotation=30)

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", fontsize=6, padding=2)

    fig.tight_layout()
    return fig


def plot_faceted_count(
    df: pd.DataFrame,
    x: str,
    hue: str,
    col: str,
    title: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    palette: str = "viridis",
    height: float = 7,
    aspect: float = 0.8,
) -> sns.FacetGrid:
    """Create faceted count plot for categorical data.

    Args:
        df: DataFrame with categorical columns
        x: Column for x-axis
        hue: Column for color grouping
        col: Column for faceting (creates separate plots)
        title: Overall title (optional)
        x_label: X-axis label (default: formatted column name)
        y_label: Y-axis label (default: "Count")
        palette: Seaborn color palette name
        height: Height of each facet in inches
        aspect: Aspect ratio of each facet

    Returns:
        Seaborn FacetGrid object
    """
    sns.set_theme(style="whitegrid")

    g = sns.catplot(
        data=df,
        x=x,
        hue=hue,
        col=col,
        kind="count",
        palette=palette,
        height=height,
        aspect=aspect,
    )

    # Set labels
    x_label = x_label or format_metric_label(x)
    y_label = y_label or "Count"
    g.set_axis_labels(x_label, y_label)
    g.set_titles(col_template="{col_name}")

    if title:
        g.figure.suptitle(title, y=1.02)

    plt.tight_layout()
    return g


def save_all_plots(
    metrics_df: pd.DataFrame,
    output_dir: Path | str,
) -> list[Path]:
    """Save all individual plots as PDFs and PNGs for the paper.

    Creates one plot per metric showing model comparison with domain hue,
    and one plot per metric showing strategy comparison with domain hue.

    Args:
        metrics_df: DataFrame with metrics in long format (metric, value, domain columns)
        output_dir: Directory to save plots

    Returns:
        List of paths to saved files
    """
    apply_nature_style()
    output_dir = Path(output_dir)
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

        ax.set_title(format_metric_label(metric), fontweight="bold")
        ax.set_xlabel("Model")
        ax.set_ylabel(format_metric_label(metric))
        ax.tick_params(axis="x", rotation=30)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f", fontsize=6, padding=2)
        fig.tight_layout()

        filename_base = f"model_comparison_{sanitize_filename(metric)}"
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

        ax.set_title(format_metric_label(metric), fontweight="bold")
        ax.set_xlabel("Memory Strategy")
        ax.set_ylabel(format_metric_label(metric))
        ax.tick_params(axis="x", rotation=30)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f", fontsize=6, padding=2)
        fig.tight_layout()

        filename_base = f"strategy_comparison_{sanitize_filename(metric)}"
        pdf_path = output_dir / f"{filename_base}.pdf"
        png_path = output_dir / f"{filename_base}.png"
        fig.savefig(pdf_path, bbox_inches="tight")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.extend([pdf_path, png_path])

    return saved
