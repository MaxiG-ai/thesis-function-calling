"""
NiceGUI Trace Viewer for local trace inspection.

A privacy-friendly, local-first web interface for inspecting experiment traces.
Replaces the previous Streamlit + Weave Cloud implementation.

Usage: uv run python tools/trace_viewer.py

Features:
- Browse experiments and timestamps from results/cfb directory
- View conversation traces as a chat interface
- Side-by-side model/strategy comparison
- Search cases by ID with autocomplete
- JSON inspector for debugging individual messages
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nicegui import app, ui
from nicegui.events import KeyEventArguments

# === CONSTANTS (configurable via environment variables) ===
RESULTS_ROOT = Path(os.environ.get("TRACE_VIEWER_RESULTS_ROOT", "results/cfb"))
PORT = int(os.environ.get("TRACE_VIEWER_PORT", "8080"))


# === DATA STRUCTURES ===
@dataclass
class LoadedTrace:
    """A single trace from one model/strategy combination."""

    strategy: str
    model: str
    case: dict[str, Any]  # The full case object from trace JSON
    metrics: dict[str, Any] | None = None  # Optional metrics data


@dataclass
class AppState:
    """Application state for the current session."""

    experiment: str = ""
    timestamp: str = ""
    case_index: dict[str, list[LoadedTrace]] = field(default_factory=dict)
    selected_case_id: str = ""
    compare_mode: bool = False
    model_a: str = ""
    model_b: str = ""
    overall_metrics: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )  # {strategy/model: metrics}
    search_filter: str = ""  # Filter for case list sidebar


# Per-user state via NiceGUI's user storage to avoid cross-session leakage


def get_state() -> AppState:
    """Get or create the application state for the current user/session."""
    state = app.storage.user.get("trace_viewer_state")
    if state is None:
        state = AppState()
        app.storage.user["trace_viewer_state"] = state
    return state


# === DATA LAYER: Directory Discovery ===
def list_experiments(root: Path = RESULTS_ROOT) -> list[str]:
    """
    List all experiment folders in the results directory.

    Args:
        root: Root directory to scan (defaults to results/cfb)

    Returns:
        List of experiment folder names, sorted alphabetically
    """
    if not root.exists():
        return []
    return sorted([d.name for d in root.iterdir() if d.is_dir()])


def list_timestamps(experiment: str, root: Path = RESULTS_ROOT) -> list[str]:
    """
    List all timestamp folders for an experiment, sorted newest-first.

    Args:
        experiment: Name of the experiment folder
        root: Root directory (defaults to results/cfb)

    Returns:
        List of timestamp folder names (YYYYMMDD_HHMM format), newest first
    """
    exp_path = root / experiment
    if not exp_path.exists():
        return []

    timestamps = []
    for d in exp_path.iterdir():
        # Filter: must be directory, match timestamp pattern, not 'temp'
        if d.is_dir() and d.name != "temp" and len(d.name) == 13 and d.name[8] == "_":
            try:
                # Validate it looks like a timestamp (YYYYMMDD_HHMM)
                int(d.name[:8])  # Date part
                int(d.name[9:])  # Time part
                timestamps.append(d.name)
            except ValueError:
                continue

    return sorted(timestamps, reverse=True)


def discover_configurations(
    experiment: str, timestamp: str, root: Path = RESULTS_ROOT
) -> dict[str, dict[str, dict[str, Path | None]]]:
    """
    Discover all strategy/model configurations within a timestamp folder.

    Scans the directory structure: {timestamp}/{strategy}/{model}/ and identifies
    trace (cfb_*.json) and metrics (metrics_*.json) files.

    Args:
        experiment: Experiment folder name
        timestamp: Timestamp folder name
        root: Root directory (defaults to results/cfb)

    Returns:
        Nested dict: {strategy: {model: {'trace_path': Path, 'metrics_path': Path|None}}}
    """
    ts_path = root / experiment / timestamp
    if not ts_path.exists():
        return {}

    configs: dict[str, dict[str, dict[str, Path | None]]] = {}

    for strategy_dir in ts_path.iterdir():
        # Skip non-directories and temp folder
        if not strategy_dir.is_dir() or strategy_dir.name == "temp":
            continue

        strategy = strategy_dir.name
        configs[strategy] = {}

        for model_dir in strategy_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model = model_dir.name
            trace_path = None
            metrics_path = None

            # Find trace and metrics files
            for f in model_dir.iterdir():
                if f.name.startswith("cfb_") and f.suffix == ".json":
                    trace_path = f
                elif f.name.startswith("metrics_") and f.suffix == ".json":
                    metrics_path = f

            if trace_path:  # Only include if we found a trace file
                configs[strategy][model] = {
                    "trace_path": trace_path,
                    "metrics_path": metrics_path,
                }

        # Remove empty strategy entries
        if not configs[strategy]:
            del configs[strategy]

    return configs


# === DATA LAYER: File Loading ===
def load_trace_file(path: Path | None) -> list[dict[str, Any]]:
    """
    Load and parse a trace JSON file.

    Args:
        path: Path to the trace JSON file

    Returns:
        List of case dictionaries, or empty list if file doesn't exist/is invalid
    """
    if not path or not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def load_metrics_file(path: Path | None) -> dict[str, Any] | None:
    """
    Load and parse a metrics JSON file.

    Args:
        path: Path to the metrics JSON file (can be None)

    Returns:
        Metrics dictionary, or None if file doesn't exist/is invalid
    """
    if not path or not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# === DATA LAYER: Indexing ===
def build_case_index(
    experiment: str, timestamp: str, root: Path = RESULTS_ROOT
) -> dict[str, list[LoadedTrace]]:
    """
    Build an index mapping case_id to list of LoadedTrace objects.

    This enables side-by-side comparison by grouping the same case from
    different model/strategy combinations.

    Args:
        experiment: Experiment folder name
        timestamp: Timestamp folder name
        root: Root directory (defaults to results/cfb)

    Returns:
        Dict mapping case_id (e.g., 'Hotels-104') to list of LoadedTrace objects
    """
    configs = discover_configurations(experiment, timestamp, root)
    index: dict[str, list[LoadedTrace]] = {}

    for strategy, models in configs.items():
        for model, paths in models.items():
            trace_path = paths["trace_path"]
            metrics_path = paths["metrics_path"]

            cases = load_trace_file(trace_path)
            metrics = load_metrics_file(metrics_path)

            for case in cases:
                case_id = case.get("id", "unknown")
                loaded = LoadedTrace(
                    strategy=strategy, model=model, case=case, metrics=metrics
                )
                if case_id not in index:
                    index[case_id] = []
                index[case_id].append(loaded)

    return index


def get_available_configs(
    case_index: dict[str, list[LoadedTrace]],
) -> list[tuple[str, str]]:
    """
    Get unique (strategy, model) combinations from the case index.

    Returns:
        List of (strategy, model) tuples
    """
    configs = set()
    for traces in case_index.values():
        for trace in traces:
            configs.add((trace.strategy, trace.model))
    return sorted(configs)


# === UI HELPERS ===
def format_config_label(strategy: str, model: str) -> str:
    """Format a strategy/model combination for display."""
    return f"{strategy} / {model}"


def get_case_status_color(status: str) -> str:
    """Get badge color based on case status."""
    status_lower = status.lower() if status else ""
    if status_lower in ("success", "completed"):
        return "green"
    elif status_lower == "failed":
        return "red"
    return "grey"


# === UI COMPONENTS ===
@ui.refreshable
def header_breadcrumbs() -> None:
    """Refreshable breadcrumbs section of header."""
    state = get_state()
    with ui.row().classes("items-center gap-2"):
        ui.icon("folder").classes("text-blue-600")
        if state.experiment:
            ui.label(state.experiment).classes("font-semibold")
            if state.timestamp:
                ui.label("/").classes("text-gray-400")
                ui.label(state.timestamp).classes("text-gray-600")
        else:
            ui.label("No data loaded").classes("text-gray-400 italic")


def create_header(state: AppState) -> None:
    """Create the sticky header with breadcrumbs, search, and metrics."""
    with ui.header().classes("bg-white text-black shadow-md items-center px-4"):
        # Left: Breadcrumbs (refreshable)
        header_breadcrumbs()

        ui.space()

        # Right: Change source button
        ui.button("Change Source", on_click=lambda: show_nav_modal(state)).props(
            "flat dense"
        )


def select_case(state: AppState, case_id: str) -> None:
    """Select a case and refresh the main content."""
    state.selected_case_id = case_id
    main_content.refresh()


def show_nav_modal(state: AppState) -> None:
    """Show the navigation modal for selecting experiment and timestamp."""
    with ui.dialog() as dialog, ui.card().classes("w-96"):
        ui.label("Select Data Source").classes("text-xl font-bold mb-4")

        experiments = list_experiments()
        exp_select = ui.select(
            experiments,
            label="Experiment",
            value=state.experiment or (experiments[0] if experiments else None),
        ).classes("w-full")

        ts_select = ui.select([], label="Timestamp").classes("w-full")

        def update_timestamps():
            exp = exp_select.value
            if exp:
                timestamps = list_timestamps(exp)
                ts_select.options = timestamps
                ts_select.value = timestamps[0] if timestamps else None

        exp_select.on("update:model-value", lambda _: update_timestamps())
        update_timestamps()  # Initial population

        with ui.row().classes("w-full justify-end mt-4"):
            ui.button("Cancel", on_click=dialog.close).props("flat")

            async def load_data():
                if exp_select.value and ts_select.value:
                    state.experiment = exp_select.value
                    state.timestamp = ts_select.value
                    state.case_index = build_case_index(
                        state.experiment, state.timestamp
                    )
                    state.selected_case_id = ""
                    dialog.close()
                    ui.notify(f"Loaded {len(state.case_index)} cases", type="positive")
                    header_breadcrumbs.refresh()
                    main_content.refresh()

            ui.button("Load", on_click=load_data).props("color=primary")

    dialog.open()


def render_user_message(content: str, idx: int, raw_msg: dict) -> None:
    """Render a user message bubble (right-aligned)."""
    with ui.row().classes("w-full justify-end"):
        with ui.card().classes("bg-blue-50 w-full"):
            with ui.row().classes("items-center gap-2"):
                ui.label(f"#{idx}").classes("text-xs text-gray-400")
                ui.label("User").classes("font-semibold text-blue-700")
                # Debug button
                ui.button(
                    icon="bug_report",
                    on_click=lambda _, m=raw_msg: show_json_inspector(m),
                ).props("flat dense size=sm")
            ui.markdown(content).classes("text-gray-800")


def render_assistant_message(msg: dict, idx: int) -> None:
    """Render an assistant message bubble (left-aligned) with function calls."""
    with ui.row().classes("w-full justify-start"):
        with ui.card().classes("bg-gray-50 w-full"):
            with ui.row().classes("items-center gap-2"):
                ui.label(f"#{idx}").classes("text-xs text-gray-400")
                ui.label("Assistant").classes("font-semibold text-gray-700")
                ui.button(
                    icon="bug_report",
                    on_click=lambda _, m=msg: show_json_inspector(m),
                ).props("flat dense size=sm")

            # Content if present
            content = msg.get("content", "")
            if content:
                ui.markdown(content).classes("text-gray-800")

            # Function calls
            func_calls = msg.get("function_call", [])
            if func_calls:
                with ui.column().classes("mt-2 gap-1 w-full"):
                    for fc in func_calls:
                        name = fc.get("name", "unknown")
                        args = fc.get("arguments", {})
                        with ui.expansion(f"Tool: {name}").classes(
                            "bg-purple-50 w-full"
                        ):
                            ui.code(
                                json.dumps(args, indent=2), language="json"
                            ).classes("text-sm")


def render_observation_message(msg: dict, idx: int) -> None:
    """Render an observation (tool result) as a collapsible card."""
    with ui.row().classes("w-full justify-start"):
        with ui.card().classes("bg-green-50 w-full border-l-4 border-green-400"):
            with ui.row().classes("items-center gap-2"):
                ui.label(f"#{idx}").classes("text-xs text-gray-400")
                ui.icon("build").classes("text-green-600")
                ui.label("Tool Result").classes("font-semibold text-green-700")
                ui.button(
                    icon="bug_report",
                    on_click=lambda _, m=msg: show_json_inspector(m),
                ).props("flat dense size=sm")

            content = msg.get("content", [])
            if isinstance(content, list):
                for i, result in enumerate(content):
                    status = result.get("status", False)
                    status_text = "Success" if status else "Failed"
                    status_color = "green" if status else "red"

                    with ui.expansion(f"Result {i + 1}: {status_text}").classes(
                        "w-full"
                    ):
                        ui.badge(status_text, color=status_color)
                        message = result.get("message", "")
                        if message:
                            ui.label(message).classes("text-sm text-gray-600")
                        data = result.get("data")
                        if data:
                            ui.code(
                                json.dumps(data, indent=2, default=str)[:500] + "...",
                                language="json",
                            ).classes("text-xs max-h-40 overflow-auto")
            else:
                ui.code(str(content)[:500], language="text")


def render_message(msg: dict, idx: int) -> None:
    """Render a single message based on its role."""
    role = msg.get("role", "unknown")
    if role == "user":
        render_user_message(msg.get("content", ""), idx, msg)
    elif role == "assistant":
        render_assistant_message(msg, idx)
    elif role == "observation":
        render_observation_message(msg, idx)
    else:
        # Unknown role - render generically
        with ui.card().classes("bg-yellow-50 max-w-[70%]"):
            ui.label(f"#{idx} - {role}").classes("font-semibold")
            ui.code(json.dumps(msg, indent=2, default=str), language="json")


def show_json_inspector(data: dict) -> None:
    """Show a drawer with raw JSON for a message."""
    with ui.dialog() as dialog, ui.card().classes("w-[600px] max-h-[80vh]"):
        ui.label("JSON Inspector").classes("text-xl font-bold")
        ui.separator()
        with ui.scroll_area().classes("h-[60vh]"):
            ui.code(json.dumps(data, indent=2, default=str), language="json").classes(
                "text-sm"
            )
        ui.button("Close", on_click=dialog.close).classes("mt-4")
    dialog.open()


def render_trace_view(trace: LoadedTrace) -> None:
    """Render a complete trace as a chat interface."""
    case = trace.case
    gen_convs = case.get("gen_convs", [])

    # Case header with metadata
    with ui.card().classes("w-full mb-4 bg-gray-100"):
        with ui.row().classes("items-center gap-4"):
            ui.label(case.get("id", "Unknown")).classes("text-xl font-bold")
            # Status badge
            status = case.get("status", "unknown")
            ui.badge(status, color=get_case_status_color(status))
            # Memory method badge
            memory_method = case.get("memory_method", "")
            if memory_method:
                ui.badge(f"Memory: {memory_method}", color="purple")

        # Count stats
        count_dict = case.get("count_dict", {})
        if count_dict:
            with ui.row().classes("gap-4 text-sm text-gray-600 mt-2"):
                ui.label(
                    f"Turns: {count_dict.get('success_turn_num', 0)}/{count_dict.get('total_turn_num', 0)}"
                )
                ui.label(
                    f"Calls: {count_dict.get('correct_call_num', 0)}/{count_dict.get('total_call_num', 0)}"
                )

        # Error messages for failed cases
        raw_message = case.get("message")
        if raw_message and status.lower() == "failed":
            # Normalize message to a list of error dicts
            if isinstance(raw_message, list):
                messages = raw_message
            elif isinstance(raw_message, dict):
                messages = [raw_message]
            elif isinstance(raw_message, str):
                messages = [{"error_type": "error", "content": raw_message}]
            else:
                messages = []

            if messages:
                with ui.column().classes("mt-2 gap-1 w-full"):
                    for err in messages:
                        error_type = err.get("error_type", "unknown")
                        content = err.get("content", "")
                        with ui.row().classes(
                            "items-center gap-2 bg-red-50 p-2 rounded"
                        ):
                            ui.badge(error_type, color="red").classes("text-xs")
                            ui.label(content).classes("text-sm text-red-700")
    # Render conversation
    with ui.column().classes("w-full gap-2"):
        for idx, msg in enumerate(gen_convs):
            render_message(msg, idx)


@ui.refreshable
def main_content() -> None:
    """Main content area - refreshable when state changes."""
    state = get_state()

    if not state.case_index:
        # Show prompt to load data
        with ui.column().classes("items-center justify-center h-[80vh] gap-4"):
            ui.icon("folder_open").classes("text-6xl text-gray-300")
            ui.label("No data loaded").classes("text-xl text-gray-500")
            ui.button("Select Data Source", on_click=lambda: show_nav_modal(state))
        return

    # Case list and trace view - use fixed width sidebar instead of splitter
    with ui.row().classes("w-full h-[calc(100vh-64px)]"):
        # Case list sidebar - fixed width
        with ui.column().classes("w-64 min-w-64 h-full border-r bg-gray-50"):
            ui.label("Cases").classes("text-lg font-bold p-3")

            # Search input for filtering cases
            def on_search_change(e):
                """Update filter and refresh case list."""
                # e.args contains the new value for update:model-value events
                value = e.args if isinstance(e.args, str) else ""
                state.search_filter = value.strip()
                case_list_items.refresh()

            ui.input(placeholder="Search cases...").classes("mx-2 mb-2").props(
                "dense outlined clearable"
            ).on("update:model-value", on_search_change)

            ui.separator()

            # Refreshable case list
            @ui.refreshable
            def case_list_items():
                """Render filtered case list items."""
                filter_text = state.search_filter.lower()
                filtered_cases = [
                    cid
                    for cid in sorted(state.case_index.keys())
                    if not filter_text or filter_text in cid.lower()
                ]

                if not filtered_cases:
                    ui.label("No matching cases").classes(
                        "text-gray-400 italic p-3 text-center"
                    )
                    return

                for case_id in filtered_cases:
                    traces = state.case_index[case_id]
                    first_trace = traces[0]
                    status = first_trace.case.get("status", "")

                    is_selected = case_id == state.selected_case_id
                    bg_class = "bg-blue-100" if is_selected else "hover:bg-gray-100"

                    with (
                        ui.card()
                        .classes(f"w-full cursor-pointer mb-1 {bg_class}")
                        .on("click", lambda _, cid=case_id: select_case(state, cid))
                    ):
                        with ui.row().classes("items-center justify-between w-full"):
                            ui.label(case_id).classes("font-medium text-sm")
                            ui.badge(
                                status, color=get_case_status_color(status)
                            ).classes("text-xs")
                        ui.label(f"{len(traces)} config(s)").classes(
                            "text-xs text-gray-500"
                        )

            with ui.scroll_area().classes("flex-grow"):
                case_list_items()

        # Main content area - takes remaining space
        with ui.column().classes("flex-1 h-full p-4 overflow-auto"):
            if state.selected_case_id:
                traces = state.case_index.get(state.selected_case_id, [])

                if not state.compare_mode:
                    # Single view - show first trace or let user pick
                    if traces:
                        # Config selector
                        configs = [(t.strategy, t.model, t) for t in traces]
                        if len(configs) > 1:
                            with ui.row().classes("items-center gap-2 mb-4"):
                                ui.label("Configuration:").classes("font-semibold")
                                config_options = {
                                    format_config_label(s, m): t for s, m, t in configs
                                }
                                selected_config = ui.select(
                                    list(config_options.keys()),
                                    value=list(config_options.keys())[0],
                                ).classes("min-w-64")

                                ui.button(
                                    "Compare Mode",
                                    on_click=lambda: toggle_compare(state),
                                ).props("outline")

                            # Render selected config (value is guaranteed non-None from initial value)
                            config_key = (
                                selected_config.value or list(config_options.keys())[0]
                            )
                            render_trace_view(config_options[config_key])
                        else:
                            render_trace_view(traces[0])
                else:
                    # Compare mode - side by side
                    render_comparison_view(state, traces)
            else:
                with ui.column().classes("items-center justify-center h-full gap-2"):
                    ui.icon("touch_app").classes("text-4xl text-gray-300")
                    ui.label("Select a case from the list").classes("text-gray-500")


def toggle_compare(state: AppState) -> None:
    """Toggle comparison mode."""
    state.compare_mode = not state.compare_mode
    main_content.refresh()


def render_comparison_view(state: AppState, traces: list[LoadedTrace]) -> None:
    """Render side-by-side comparison of two configurations."""
    if len(traces) < 2:
        ui.label("Need at least 2 configurations to compare").classes("text-gray-500")
        ui.button("Exit Compare Mode", on_click=lambda: toggle_compare(state))
        return

    configs = [(format_config_label(t.strategy, t.model), t) for t in traces]
    config_dict = dict(configs)
    config_labels = list(config_dict.keys())

    with ui.row().classes("w-full items-center gap-4 mb-4"):
        ui.label("Compare:").classes("font-semibold")
        select_a = ui.select(config_labels, value=config_labels[0], label="Config A")
        ui.label("vs").classes("text-gray-500")
        select_b = ui.select(
            config_labels,
            value=config_labels[1] if len(config_labels) > 1 else config_labels[0],
            label="Config B",
        )
        ui.button("Exit Compare", on_click=lambda: toggle_compare(state)).props(
            "outline"
        )

    # Side by side columns - ensure both have equal width
    with ui.row().classes("w-full gap-4"):
        with ui.column().classes("flex-1 min-w-0"):
            label_a = select_a.value or config_labels[0]
            ui.label(label_a).classes("font-bold text-lg mb-2")
            with ui.scroll_area().classes("h-[70vh]"):
                render_trace_view(config_dict[label_a])

        ui.separator().props("vertical")

        with ui.column().classes("flex-1 min-w-0"):
            label_b = select_b.value or config_labels[1]
            ui.label(label_b).classes("font-bold text-lg mb-2")
            with ui.scroll_area().classes("h-[70vh]"):
                render_trace_view(config_dict[label_b])


# === KEYBOARD BINDINGS ===
def setup_keyboard_bindings() -> None:
    """Set up global keyboard shortcuts."""
    ui.keyboard(
        on_key=handle_keyboard,
        ignore=["input", "textarea"],
    )


def handle_keyboard(e: KeyEventArguments) -> None:
    """Handle keyboard events."""
    # Esc to clear selection
    if e.key == "Escape":
        state = get_state()
        state.selected_case_id = ""
        main_content.refresh()


# === MAIN PAGE ===
@ui.page("/")
def index():
    """Main page of the trace viewer."""
    state = get_state()
    create_header(state)
    setup_keyboard_bindings()

    with ui.column().classes("w-full p-4"):
        main_content()


# === ENTRY POINT ===
if __name__ == "__main__":
    ui.run(
        port=PORT,
        title="Trace Viewer",
        reload=False,  # Disable for production
        show=False,  # Don't auto-open browser
        storage_secret="trace_viewer_secret",  # Required for user storage
    )
