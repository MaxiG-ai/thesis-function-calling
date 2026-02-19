"""Data loading and transformation utilities for experiment analysis.

Provides functions to load task results and metrics from experiment directories,
transform dataframes, and save results to CSV files.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def parse_timestamp(timestamp_str: str) -> str | None:
    """Parse timestamp string from YYYYMMDD_HHMM format to ISO format.

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


def safe_get_nested(data: dict, *keys) -> Any | None:
    """Safely navigate nested dictionaries, returning None if any key is missing.

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


def load_task_results(experiment_path: Path | str) -> pd.DataFrame:
    """Load task-level results from cfb_*.json files into a DataFrame.

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
                for json_file in model_dir.glob(
                    f"cfb_{model}_{memory_strategy}_{timestamp_str}.json"
                ):
                    rows.extend(
                        _parse_task_json_file(
                            json_file, memory_strategy, model, timestamp_iso
                        )
                    )

    return pd.DataFrame(rows)


def _parse_task_json_file(
    json_path: Path,
    memory_strategy: str,
    model: str,
    timestamp_iso: str,
) -> list[dict]:
    """Parse a single JSON results file into row dictionaries.

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
        message_error_type, message_error_reasoning = extract_error_message(task)

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
            "message_error_type": message_error_type,
            "message_error_reasoning": message_error_reasoning,
        }
        rows.append(row)

    return rows


def load_metrics_long(base_dir: Path | str, project: str) -> pd.DataFrame:
    """Load all metrics for a project in long format, aggregating across timestamps.

    If multiple timestamps contain the same strategy/model combination,
    they are disambiguated by prefixing with a short timestamp identifier.

    Args:
        base_dir: Root directory containing experiment results.
        project: Project name (subdirectory).

    Returns:
        DataFrame with columns: memory_strategy, model, metric, value, domain.
        Strategy/model names may include timestamp prefix if duplicates exist.
        Domain is extracted from nested metrics or set to "Overall" for top-level metrics.
    """
    base_dir = Path(base_dir)
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
                        # Nested metric (e.g., domain_turn_acc with domain keys)
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


def add_domain_column(df: pd.DataFrame) -> pd.DataFrame:
    """Extract domain from task_id column and add as new domain column.

    Args:
        df: DataFrame with task_id column (e.g., "Hotels-104")

    Returns:
        DataFrame with new "domain" column (e.g., "Hotels")
    """
    df = df.copy()
    df["domain"] = df["task_id"].str.split("-").str[0]
    return df


def extract_error_message(task: Any) -> tuple[str, str]:
    """Extract error type and explanation from the task `message` payload.

    Supported payload formats:
    - `{"message": {"error_type": str, "content": str}}`
    - `{"message": [{"error_type": str, "content": str}, ...]}`

    For list payloads, error types are aggregated as count-strings in first-seen
    order (e.g., "2 value_error; 1 decode_error"), and contents are concatenated
    with "; " in original order.

    Args:
        task: Task-level object that may contain a `message` key.

    Returns:
        Tuple of `(error_type_summary, error_reasoning_summary)`.
        If payload does not match supported shapes, returns:
        ("no error detected", "no error detected").
    """
    no_error = "no error detected"

    if not isinstance(task, dict):
        return no_error, no_error

    message = task.get("message")

    if isinstance(message, dict):
        error_type = message.get("error_type")
        content = message.get("content")
        if isinstance(error_type, str) and isinstance(content, str):
            return error_type, content
        return no_error, no_error

    if isinstance(message, list):
        counts: dict[str, int] = {}
        order: list[str] = []
        contents: list[str] = []

        for item in message:
            if not isinstance(item, dict):
                continue

            error_type = item.get("error_type")
            content = item.get("content")

            if isinstance(error_type, str):
                if error_type not in counts:
                    counts[error_type] = 0
                    order.append(error_type)
                counts[error_type] += 1

            if isinstance(content, str):
                contents.append(content)

        if not counts and not contents:
            return no_error, no_error

        error_type_summary = (
            "; ".join(f"{counts[error_type]} {error_type}" for error_type in order)
            if counts
            else no_error
        )
        error_reasoning_summary = "; ".join(contents) if contents else no_error
        return error_type_summary, error_reasoning_summary

    return no_error, no_error


def add_turn_category(
    df: pd.DataFrame,
    column: str = "total_call_num",
    bins: list[int] | None = None,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Categorize tasks by turn count into bins.

    Args:
        df: DataFrame with turn count column
        column: Name of column to categorize (default: "total_call_num")
        bins: Bin edges for pd.cut (default: [0, 4, 8, 100])
        labels: Labels for categories (default: ["few_turns", "med_turns", "many_turns"])

    Returns:
        DataFrame with new "turns_cat" column
    """
    if bins is None:
        bins = [0, 4, 8, 100]
    if labels is None:
        labels = ["few_turns", "med_turns", "many_turns"]

    df = df.copy()
    df["turns_cat"] = pd.cut(df[column], bins=bins, labels=labels, ordered=True)
    return df


def join_results_with_metrics(
    task_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join task-level results with aggregated experiment metrics.

    The join is performed on (memory_strategy, model, timestamp) - the unique
    identifier for each experiment run.

    Args:
        task_df: DataFrame from load_task_results() with task-level data
        metrics_df: DataFrame with aggregate metrics (must have memory_strategy, model, timestamp)

    Returns:
        DataFrame with all columns from task_df plus the metrics columns
    """
    join_keys = ["memory_strategy", "model", "timestamp"]

    return task_df.merge(
        metrics_df,
        on=join_keys,
        how="left",
        suffixes=("", "_agg"),
    )


def save_metric_tables(
    metrics_df: pd.DataFrame,
    output_dir: Path | str,
) -> list[Path]:
    """Save metrics as CSV tables in long format.

    Args:
        metrics_df: DataFrame with metrics data
        output_dir: Directory to save CSV files

    Returns:
        List of paths to saved CSV files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # Save long format
    long_path = output_dir / "all_metrics.csv"
    metrics_df.to_csv(long_path, index=False)
    saved.append(long_path)

    return saved


def save_results_table(metrics_df: pd.DataFrame, output_dir: Path | str) -> Path | None:
    """Save comprehensive results table with all metrics in wide format.

    Args:
        metrics_df: DataFrame with metrics in long format
        output_dir: Directory to save CSV file

    Returns:
        Path to saved CSV file, or None if DataFrame is empty
    """
    if metrics_df.empty:
        return None

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Aggregate by strategy/model/metric (mean across domains)
    agg_df = metrics_df.groupby(["memory_strategy", "model", "metric"], as_index=False)[
        "value"
    ].mean()

    # Pivot to wide format
    table = agg_df.pivot_table(
        index="memory_strategy",
        columns=["model", "metric"],
        values="value",
    )
    table_path = output_dir / "results_table.csv"
    table.to_csv(table_path)
    return table_path


def load_compressed_traces(experiment_path: Path | str) -> pd.DataFrame:
    """Load compressed token traces and compute token metrics per task.

    Iterates through all subdirectories of an experiment folder structured as:
    experiment_path/<timestamp>/<memory_strategy>/<model>/compressed_*.json

    Computes the following metrics per task from the compressed_trace array:
    - max_token_count: Maximum compressed tokens across all steps
    - avg_token_count: Average compressed tokens across steps
    - final_token_count: Token count at conversation end
    - min_token_count: Minimum compressed tokens
    - num_steps: Total conversation steps
    - total_input_tokens: Sum of input tokens
    - avg_compression_ratio: Average compression ratio
    - max_compression_ratio: Maximum compression achieved
    - token_growth_rate: Relative growth from first to last step
    - step_aggregates: JSON string with per-step token counts
    - domain: Extracted from task_id (e.g., "Hotels-104" -> "Hotels")

    Args:
        experiment_path: Path to experiment directory

    Returns:
        DataFrame with columns: task_id, memory_strategy, model, timestamp,
        domain, and all computed token metrics. Empty if no data found.
    """
    experiment_dir = Path(experiment_path)
    rows = []

    if not experiment_dir.exists():
        return pd.DataFrame()

    # Iterate through timestamp directories
    for timestamp_dir in experiment_dir.iterdir():
        if not timestamp_dir.is_dir():
            continue

        timestamp_str = timestamp_dir.name
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

                # Find compressed_*.json files
                for json_file in model_dir.glob(
                    f"compressed_{model}_{memory_strategy}_{timestamp_str}.json"
                ):
                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            tasks = json.load(f)

                        for task in tasks:
                            task_id = task.get("id")
                            trace = task.get("compressed_trace", [])

                            if not trace:
                                continue

                            # Extract token counts from each step
                            token_counts = [
                                step.get("compressed_token_count", 0) for step in trace
                            ]
                            input_tokens = [
                                step.get("input_token_count", 0) for step in trace
                            ]
                            compression_ratios = [
                                step.get("compression_ratio", 1.0) for step in trace
                            ]

                            # Compute metrics
                            num_steps = len(token_counts)
                            max_token = max(token_counts) if token_counts else 0
                            min_token = min(token_counts) if token_counts else 0
                            avg_token = (
                                sum(token_counts) / num_steps if num_steps > 0 else 0
                            )
                            final_token = token_counts[-1] if token_counts else 0
                            first_token = token_counts[0] if token_counts else 0
                            total_input = sum(input_tokens)
                            avg_compression = (
                                sum(compression_ratios) / num_steps
                                if num_steps > 0
                                else 0
                            )
                            max_compression = (
                                max(compression_ratios) if compression_ratios else 1.0
                            )
                            token_growth = (
                                (final_token - first_token) / first_token
                                if first_token > 0
                                else 0
                            )

                            # Step aggregates as JSON string (step number -> token count)
                            step_agg = {
                                str(i + 1): int(tc) for i, tc in enumerate(token_counts)
                            }
                            step_agg_str = json.dumps(step_agg)

                            # Extract domain from task_id
                            domain = (
                                task_id.split("-")[0] if "-" in task_id else "Unknown"
                            )

                            row = {
                                "task_id": task_id,
                                "memory_strategy": memory_strategy,
                                "model": model,
                                "timestamp": timestamp_iso,
                                "domain": domain,
                                "max_token_count": max_token,
                                "avg_token_count": avg_token,
                                "final_token_count": final_token,
                                "min_token_count": min_token,
                                "num_steps": num_steps,
                                "total_input_tokens": total_input,
                                "avg_compression_ratio": avg_compression,
                                "max_compression_ratio": max_compression,
                                "token_growth_rate": token_growth,
                                "step_aggregates": step_agg_str,
                            }
                            rows.append(row)
                    except (json.JSONDecodeError, ValueError):
                        # Skip files that can't be parsed
                        continue

    return pd.DataFrame(rows)


def add_token_metrics_to_df(
    task_df: pd.DataFrame, experiment_path: Path | str
) -> pd.DataFrame:
    """Add token metrics columns to task results DataFrame.

    Loads compressed traces and joins computed token metrics to the task
    results DataFrame on (task_id, memory_strategy, model, timestamp).

    Args:
        task_df: DataFrame from load_task_results() with task-level data
        experiment_path: Path to experiment directory for loading compressed traces

    Returns:
        DataFrame with original task_df columns plus token metric columns.
        Rows without matching compressed trace data will have NaN for token metrics.
    """
    token_df = load_compressed_traces(experiment_path)

    if token_df.empty:
        # Return original DataFrame if no token data found
        return task_df

    # Join on task_id, memory_strategy, model, timestamp
    join_keys = ["task_id", "memory_strategy", "model", "timestamp"]

    return task_df.merge(
        token_df,
        on=join_keys,
        how="left",
        suffixes=("", "_token"),
    )
