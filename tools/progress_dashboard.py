"""
Lightweight live progress dashboard for experiment runs.

Replaces Weave/W&B remote tracing with a zero-network-overhead Rich console
display that auto-refreshes in the terminal during experiment execution.

Architecture
------------
- ``ConfigKey``      -- Hashable identifier for one (model, memory, thresholds) combo.
- ``ActiveConfig``   -- Mutable per-config counters updated while cases run.
- ``CompletedConfig``-- Frozen snapshot stored after a config finishes.
- ``ExperimentProgress`` -- Thread-safe registry that drives a Rich ``Live``
                          display showing active evaluations and completed results.

Usage (in cfb_run_eval.py)
--------------------------
    progress = ExperimentProgress()
    progress.start_experiment()                        # start Live display

    key = ConfigKey(model=..., memory=..., ...)
    progress.start_configuration(key, total_cases=N)  # register a running eval
    progress.complete_case(key, case_id=..., success=True/False)  # per-case hook
    progress.complete_configuration(key, metrics={...})           # eval done

    progress.finish_experiment()                       # stop display

The ``Live`` display is optional for tests: constructing ``ExperimentProgress``
and calling ``start_configuration`` / ``complete_case`` / ``complete_configuration``
works without ever calling ``start_experiment`` / ``finish_experiment``.

Thread safety
-------------
A single ``threading.Lock`` guards all mutations so parallel haystack threads
can safely call ``complete_case`` concurrently.
"""

import threading
from dataclasses import dataclass, field
from typing import Dict, Optional

from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn
from rich.table import Table
from rich.text import Text


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigKey:
    """Immutable, hashable identifier for one experiment configuration.

    Used as dict key in ``ExperimentProgress`` so we can look up the correct
    counter set for each concurrent haystack thread.
    """

    model: str
    memory: str
    compact_threshold: Optional[int]
    haystack_threshold: Optional[int]

    def __str__(self) -> str:
        parts = [self.model, self.memory]
        if self.compact_threshold is not None:
            parts.append(f"t{self.compact_threshold}")
        if self.haystack_threshold is not None:
            parts.append(f"h{self.haystack_threshold}")
        return "/".join(parts)


@dataclass
class ActiveConfig:
    """Live counters for a running configuration.

    Mutated by ``complete_case``; protected by ``ExperimentProgress._lock``.
    """

    key: ConfigKey
    total_cases: int
    cases_done: int = 0
    success_count: int = 0
    # Rich Progress task ID for the per-config progress bar
    task_id: Optional[TaskID] = None

    @property
    def success_rate(self) -> float:
        """Percentage of completed cases that succeeded (0–100)."""
        if self.cases_done == 0:
            return 0.0
        return self.success_count / self.cases_done * 100


@dataclass
class CompletedConfig:
    """Immutable snapshot captured when a configuration finishes."""

    key: ConfigKey
    total_cases: int
    cases_done: int
    success_count: int
    metrics: Dict  # aggregate metrics from calculate_metrics()

    @property
    def success_rate(self) -> float:
        if self.cases_done == 0:
            return 0.0
        return self.success_count / self.cases_done * 100


# ---------------------------------------------------------------------------
# Main dashboard class
# ---------------------------------------------------------------------------


class ExperimentProgress:
    """Thread-safe experiment progress tracker with Rich Live display.

    Designed to be constructed once in ``main()`` and shared across all
    threads.  All state mutations are protected by a single lock to handle
    the parallel haystack threshold execution.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Active and completed configuration registries
        self.active_configs: Dict[ConfigKey, ActiveConfig] = {}
        self.completed_configs: Dict[ConfigKey, CompletedConfig] = {}

        # Global counter: total individual cases completed across all configs
        self.total_cases_done: int = 0

        # Rich Live display internals (None until start_experiment is called)
        self._live: Optional[Live] = None
        self._console = Console()

        # Per-configuration Rich Progress bars shown inside the active panel
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=30),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("[green]{task.fields[success_rate]:.0f}%✓"),
            console=self._console,
            transient=False,
        )

    # ------------------------------------------------------------------
    # Experiment lifecycle
    # ------------------------------------------------------------------

    def start_experiment(self) -> None:
        """Start the Rich Live display.  Call once from main()."""
        self._live = Live(
            self._build_layout(),
            console=self._console,
            refresh_per_second=2,
            # redirect_stderr keeps non-Live log output visible in the console
            redirect_stderr=False,
        )
        self._live.start()

    def finish_experiment(self) -> None:
        """Stop the Live display and print a final summary."""
        if self._live is not None:
            self._live.stop()
        self._console.print(self._build_final_summary())

    # ------------------------------------------------------------------
    # Configuration lifecycle
    # ------------------------------------------------------------------

    def start_configuration(self, key: ConfigKey, total_cases: int) -> None:
        """Register a new running configuration.

        Args:
            key:         Unique identifier for this (model, memory, thresholds) tuple.
            total_cases: Number of dataset cases that will be evaluated.
        """
        with self._lock:
            # Add a Rich Progress task for the per-config bar
            task_id = self._progress.add_task(
                str(key), total=total_cases, success_rate=0.0
            )
            self.active_configs[key] = ActiveConfig(
                key=key, total_cases=total_cases, task_id=task_id
            )
        self._refresh()

    def complete_case(self, key: ConfigKey, case_id: str, success: bool) -> None:
        """Record that one case has finished for the given configuration.

        Safe to call from multiple threads simultaneously (different keys).

        Args:
            key:     Configuration this case belongs to.
            case_id: Case identifier (used for internal bookkeeping only).
            success: Whether the case returned ``message == "Success."``.
        """
        with self._lock:
            if key not in self.active_configs:
                return  # Guard against late arrivals after complete_configuration
            entry = self.active_configs[key]
            entry.cases_done += 1
            if success:
                entry.success_count += 1
            self.total_cases_done += 1

            # Update the Rich Progress bar for this config
            if entry.task_id is not None:
                self._progress.update(
                    entry.task_id,
                    completed=entry.cases_done,
                    success_rate=entry.success_rate,
                )
        self._refresh()

    def complete_configuration(self, key: ConfigKey, metrics: Dict) -> None:
        """Mark a configuration as done and store its aggregate metrics.

        Moves the entry from ``active_configs`` to ``completed_configs``.

        Args:
            key:     Configuration that finished.
            metrics: Output of ``calculate_metrics()`` (success rates, call acc, etc.)
        """
        with self._lock:
            if key not in self.active_configs:
                return
            entry = self.active_configs.pop(key)

            # Remove the progress bar task (it will appear in the summary table)
            if entry.task_id is not None:
                self._progress.remove_task(entry.task_id)

            self.completed_configs[key] = CompletedConfig(
                key=key,
                total_cases=entry.total_cases,
                cases_done=entry.cases_done,
                success_count=entry.success_count,
                metrics=metrics,
            )
        self._refresh()

    # ------------------------------------------------------------------
    # Rich renderables
    # ------------------------------------------------------------------

    def make_progress_table(self) -> Table:
        """Build a Rich Table summarising completed configurations.

        Returns a Table with one row per completed configuration showing
        model, memory strategy, thresholds, case counts, success rate, and
        key metrics from ``calculate_metrics()``.
        """
        table = Table(
            title="Completed Configurations",
            show_header=True,
            header_style="bold cyan",
            expand=True,
        )
        table.add_column("Model", style="bold")
        table.add_column("Memory")
        table.add_column("Compact", justify="right")
        table.add_column("Haystack", justify="right")
        table.add_column("Cases", justify="right")
        table.add_column("Success%", justify="right", style="green")
        table.add_column("CallAcc%", justify="right")
        table.add_column("Complete", justify="right")
        table.add_column("Correct", justify="right")

        for cfg in self.completed_configs.values():
            m = cfg.metrics
            table.add_row(
                cfg.key.model,
                cfg.key.memory,
                str(cfg.key.compact_threshold)
                if cfg.key.compact_threshold is not None
                else "—",
                str(cfg.key.haystack_threshold)
                if cfg.key.haystack_threshold is not None
                else "—",
                f"{cfg.cases_done}/{cfg.total_cases}",
                f"{cfg.success_rate:.1f}",
                f"{m.get('overall_call_acc', 0):.1f}" if m else "—",
                f"{m.get('complete_score_avg', 0):.2f}" if m else "—",
                f"{m.get('correct_score_avg', 0):.2f}" if m else "—",
            )
        return table

    def make_active_panel(self) -> Panel:
        """Build a Rich Panel showing currently running configurations.

        When no configurations are active it shows an idle message.
        Returns a Panel (always, even when empty) so the Live layout is stable.
        """
        if not self.active_configs:
            content = Text("No active evaluations", style="dim italic")
        else:
            content = self._progress

        return Panel(
            content,
            title=f"[bold yellow]Active Evaluations[/] ({len(self.active_configs)} running)",
            border_style="yellow",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_layout(self):
        """Compose the full Live renderable from the two panels."""
        return Columns(
            [self.make_active_panel(), self.make_progress_table()],
            equal=False,
            expand=True,
        )

    def _refresh(self) -> None:
        """Push an updated layout to the Live display (no-op if not started)."""
        if self._live is not None:
            self._live.update(self._build_layout())

    def _build_final_summary(self) -> Panel:
        """Render a summary panel after the experiment finishes."""
        lines: list[Text] = []
        lines.append(
            Text(f"Total cases completed: {self.total_cases_done}", style="bold")
        )
        lines.append(Text(""))

        for cfg in self.completed_configs.values():
            m = cfg.metrics
            success_pct = f"{cfg.success_rate:.1f}%"
            call_acc = f"{m.get('overall_call_acc', 0):.1f}%" if m else "—"
            lines.append(
                Text.assemble(
                    (str(cfg.key), "bold cyan"),
                    "  →  ",
                    ("Success: ", "dim"),
                    (success_pct, "green"),
                    ("  CallAcc: ", "dim"),
                    (call_acc, "blue"),
                )
            )

        from rich.console import Group  # local import to avoid top-level circular

        return Panel(
            Group(*lines),
            title="[bold green]Experiment Complete[/]",
            border_style="green",
        )
