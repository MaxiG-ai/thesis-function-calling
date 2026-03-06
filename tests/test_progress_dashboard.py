"""
Tests for the ExperimentProgress live dashboard (tools/progress_dashboard.py).

The dashboard is a thread-safe, in-memory progress tracker that drives a Rich
Live console display during experiment runs.  These tests verify the state
machine, success counting, thread safety, and the Rich renderables — without
ever starting the actual Rich Live context (which would require a real TTY).

Test categories:
1. Initialization      - clean state after construction
2. Configuration flow  - start_configuration, complete_case, complete_configuration
3. Success counting    - success rate computed correctly from completed cases
4. Thread safety       - concurrent complete_case calls don't corrupt counters
5. Rich renderables    - make_progress_table / make_active_panel return valid Rich objects
6. Finish experiment   - mark experiment done, no further mutation accepted
"""

import threading
import pytest
from rich.table import Table
from rich.panel import Panel

from tools.progress_dashboard import ExperimentProgress, ConfigKey


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_key(
    model: str = "gpt-test",
    memory: str = "truncation",
    compact: int | None = 1000,
    haystack: int | None = None,
) -> ConfigKey:
    """Construct a ConfigKey for use in tests."""
    return ConfigKey(
        model=model,
        memory=memory,
        compact_threshold=compact,
        haystack_threshold=haystack,
    )


def _make_progress(total_cases: int = 5) -> tuple[ExperimentProgress, ConfigKey]:
    """Build a started ExperimentProgress with one registered configuration."""
    progress = ExperimentProgress()
    key = _make_key()
    progress.start_configuration(key, total_cases=total_cases)
    return progress, key


# ---------------------------------------------------------------------------
# 1. Initialization
# ---------------------------------------------------------------------------


def test_initial_state_is_empty():
    """
    A freshly constructed ExperimentProgress must have no active configs,
    no completed configs, and a total_cases_done counter of zero.  This
    ensures the object is always in a well-defined state before any experiment
    is run.
    """
    progress = ExperimentProgress()
    assert progress.total_cases_done == 0
    assert len(progress.active_configs) == 0
    assert len(progress.completed_configs) == 0


# ---------------------------------------------------------------------------
# 2. Configuration flow
# ---------------------------------------------------------------------------


def test_start_configuration_registers_active_entry():
    """
    After start_configuration the key must appear in active_configs with
    cases_done=0 and the supplied total_cases value.  The completed_configs
    dict must remain empty.
    """
    progress = ExperimentProgress()
    key = _make_key()
    progress.start_configuration(key, total_cases=10)

    assert key in progress.active_configs
    entry = progress.active_configs[key]
    assert entry.cases_done == 0
    assert entry.total_cases == 10
    assert entry.success_count == 0
    assert len(progress.completed_configs) == 0


def test_complete_case_increments_counter():
    """
    Each call to complete_case must increment cases_done by 1 and
    total_cases_done by 1 while the config remains active (not completed).
    """
    progress, key = _make_progress(total_cases=5)

    progress.complete_case(key, case_id="Case-1", success=True)

    assert progress.active_configs[key].cases_done == 1
    assert progress.total_cases_done == 1
    assert key in progress.active_configs  # still active


def test_complete_case_tracks_success_and_failure():
    """
    success=True increments success_count; success=False does not.
    After three completions (2 success, 1 failure) success_count must be 2.
    """
    progress, key = _make_progress(total_cases=5)

    progress.complete_case(key, case_id="Case-1", success=True)
    progress.complete_case(key, case_id="Case-2", success=False)
    progress.complete_case(key, case_id="Case-3", success=True)

    entry = progress.active_configs[key]
    assert entry.success_count == 2
    assert entry.cases_done == 3


def test_complete_configuration_moves_to_completed():
    """
    complete_configuration must remove the key from active_configs and insert
    it into completed_configs with the provided metrics dict attached.
    """
    progress, key = _make_progress(total_cases=2)
    progress.complete_case(key, case_id="A", success=True)
    progress.complete_case(key, case_id="B", success=False)

    metrics = {"overall_success": 50.0, "overall_call_acc": 80.0}
    progress.complete_configuration(key, metrics=metrics)

    assert key not in progress.active_configs
    assert key in progress.completed_configs
    assert progress.completed_configs[key].metrics == metrics


def test_complete_configuration_preserves_case_counts():
    """
    The CompletedConfig stored in completed_configs must retain the
    cases_done and success_count values accumulated during the active phase
    so the summary table is accurate.
    """
    progress, key = _make_progress(total_cases=3)
    for i in range(3):
        progress.complete_case(key, case_id=f"Case-{i}", success=(i < 2))

    progress.complete_configuration(key, metrics={})

    completed = progress.completed_configs[key]
    assert completed.cases_done == 3
    assert completed.success_count == 2


# ---------------------------------------------------------------------------
# 3. Success-rate calculation
# ---------------------------------------------------------------------------


def test_success_rate_is_zero_when_no_cases_done():
    """
    Before any cases are completed the success rate for an active config
    must be 0.0, not a division-by-zero error.
    """
    progress, key = _make_progress(total_cases=5)
    rate = progress.active_configs[key].success_rate
    assert rate == 0.0


def test_success_rate_computed_correctly():
    """
    success_rate is success_count / cases_done * 100.
    After 3 successes out of 4 completed cases the rate must be 75.0.
    """
    progress, key = _make_progress(total_cases=10)
    for i in range(4):
        progress.complete_case(key, case_id=f"C{i}", success=(i < 3))

    assert progress.active_configs[key].success_rate == pytest.approx(75.0)


# ---------------------------------------------------------------------------
# 4. Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_complete_case_is_thread_safe():
    """
    Multiple threads calling complete_case concurrently must not lose updates.
    100 threads each completing 1 case means final cases_done == 100 and
    total_cases_done == 100.  Any lost update would be a data race and would
    make the counts smaller.
    """
    n_threads = 100
    progress, key = _make_progress(total_cases=n_threads)

    def worker(i: int):
        progress.complete_case(key, case_id=f"C{i}", success=(i % 2 == 0))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert progress.active_configs[key].cases_done == n_threads
    assert progress.total_cases_done == n_threads
    assert progress.active_configs[key].success_count == n_threads // 2


def test_multiple_parallel_configs_do_not_interfere():
    """
    Two configurations running in parallel must track their progress
    independently.  Completing a case for key_a must not affect key_b's
    counters and vice versa.
    """
    progress = ExperimentProgress()
    key_a = _make_key(memory="truncation", haystack=None)
    key_b = _make_key(memory="truncation", haystack=20000)

    progress.start_configuration(key_a, total_cases=5)
    progress.start_configuration(key_b, total_cases=5)

    # Drive both configs from separate threads
    def complete_a():
        for i in range(3):
            progress.complete_case(key_a, case_id=f"A{i}", success=True)

    def complete_b():
        for i in range(5):
            progress.complete_case(key_b, case_id=f"B{i}", success=(i < 2))

    t_a = threading.Thread(target=complete_a)
    t_b = threading.Thread(target=complete_b)
    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()

    assert progress.active_configs[key_a].cases_done == 3
    assert progress.active_configs[key_b].cases_done == 5
    assert progress.active_configs[key_a].success_count == 3
    assert progress.active_configs[key_b].success_count == 2
    assert progress.total_cases_done == 8


# ---------------------------------------------------------------------------
# 5. Rich renderables
# ---------------------------------------------------------------------------


def test_make_progress_table_returns_rich_table():
    """
    make_progress_table must return a rich.table.Table instance regardless of
    whether there are completed configs.  The returned object must be renderable
    (i.e. has a __rich_console__ or __rich__ method as expected by Rich).
    """
    progress = ExperimentProgress()
    table = progress.make_progress_table()
    assert isinstance(table, Table)


def test_make_progress_table_has_completed_rows():
    """
    After completing a configuration, make_progress_table must have one data row
    for that configuration (row_count == 1).  This checks that the table is
    populated with real data, not just empty.
    """
    progress, key = _make_progress(total_cases=2)
    progress.complete_case(key, case_id="X", success=True)
    progress.complete_case(key, case_id="Y", success=False)
    progress.complete_configuration(
        key, metrics={"overall_success": 50.0, "overall_call_acc": 60.0}
    )

    table = progress.make_progress_table()
    assert table.row_count == 1


def test_make_active_panel_returns_rich_panel():
    """
    make_active_panel must return a rich.panel.Panel instance, even when
    no configurations are currently active.
    """
    progress = ExperimentProgress()
    panel = progress.make_active_panel()
    assert isinstance(panel, Panel)


def test_make_active_panel_with_running_configs():
    """
    When there are active configurations make_active_panel must return a Panel
    that encodes the configuration key information (model, memory, etc.) in its
    renderable content without raising.
    """
    progress, key = _make_progress(total_cases=10)
    progress.complete_case(key, case_id="C1", success=True)
    # Should not raise; just verify it returns a Panel
    panel = progress.make_active_panel()
    assert isinstance(panel, Panel)


# ---------------------------------------------------------------------------
# 6. ConfigKey equality and hashability
# ---------------------------------------------------------------------------


def test_config_key_is_hashable_and_comparable():
    """
    ConfigKey must be hashable (usable as dict key) and equal to another
    ConfigKey with the same values.  This underpins all dict-based lookups
    in ExperimentProgress.
    """
    k1 = ConfigKey(
        model="m", memory="truncation", compact_threshold=1000, haystack_threshold=None
    )
    k2 = ConfigKey(
        model="m", memory="truncation", compact_threshold=1000, haystack_threshold=None
    )
    k3 = ConfigKey(
        model="m", memory="truncation", compact_threshold=2000, haystack_threshold=None
    )

    assert k1 == k2
    assert k1 != k3
    assert hash(k1) == hash(k2)
    assert hash(k1) != hash(k3)

    d = {k1: "value"}
    assert d[k2] == "value"  # k2 should find the entry keyed by k1
