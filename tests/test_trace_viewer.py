"""
Tests for the NiceGUI trace viewer data layer.

These tests verify the directory traversal, data loading, and indexing
functionality used by the trace viewer to discover and load local trace files.

Test Strategy:
- Use the actual results/cfb directory structure for integration tests
- Test functions in isolation with predictable inputs
- Verify edge cases like missing files, empty directories, malformed JSON
"""

import json
import sys
from pathlib import Path

import pytest

# Add project root to path for importing tools module
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.trace_viewer import (
    RESULTS_ROOT,
    LoadedTrace,
    build_case_index,
    discover_configurations,
    list_experiments,
    list_timestamps,
    load_metrics_file,
    load_trace_file,
)


class TestDirectoryDiscovery:
    """Tests for directory traversal and discovery functions."""

    def test_list_experiments_returns_folder_names(self):
        """
        Verify list_experiments() returns experiment folder names from results/cfb.

        The function should scan the RESULTS_ROOT directory and return a list of
        subdirectory names that represent different experiments (e.g., 'test_ace',
        'test_mcpbench'). It should exclude files and hidden directories.
        """
        experiments = list_experiments()
        assert isinstance(experiments, list)
        assert len(experiments) > 0
        # Known experiments from the actual directory
        assert "test_ace" in experiments

    def test_list_experiments_excludes_files(self):
        """
        Verify list_experiments() only returns directories, not files.

        The results/cfb directory may contain log files or other non-directory
        entries. These should be filtered out.
        """
        experiments = list_experiments()
        for exp in experiments:
            exp_path = RESULTS_ROOT / exp
            assert exp_path.is_dir(), f"{exp} should be a directory"

    def test_list_timestamps_returns_sorted_timestamps(self):
        """
        Verify list_timestamps() returns timestamp folders in descending order.

        Timestamps follow the format YYYYMMDD_HHMM. The function should return
        them sorted newest-first for easier navigation in the UI.
        """
        timestamps = list_timestamps("test_ace")
        assert isinstance(timestamps, list)
        assert len(timestamps) > 0
        # Verify format: YYYYMMDD_HHMM
        for ts in timestamps:
            assert len(ts) == 13, f"Timestamp {ts} should be 13 chars (YYYYMMDD_HHMM)"
            assert ts[8] == "_", f"Timestamp {ts} should have underscore at position 8"
        # Verify sorted descending (newest first)
        assert timestamps == sorted(timestamps, reverse=True)

    def test_list_timestamps_invalid_experiment_returns_empty(self):
        """
        Verify list_timestamps() returns empty list for non-existent experiment.

        Rather than raising an exception, the function should gracefully return
        an empty list when the experiment folder doesn't exist.
        """
        timestamps = list_timestamps("nonexistent_experiment_xyz")
        assert timestamps == []

    def test_list_timestamps_excludes_temp_folders(self):
        """
        Verify list_timestamps() excludes 'temp' and other non-timestamp folders.

        Some timestamp directories contain a 'temp' subfolder for intermediate
        files. This should not appear in the timestamps list.
        """
        timestamps = list_timestamps("test_ace")
        assert "temp" not in timestamps
        for ts in timestamps:
            # All entries should match timestamp pattern
            assert ts[0:8].isdigit(), f"{ts} should start with 8 digits"


class TestConfigurationDiscovery:
    """Tests for discovering model/strategy configurations within a timestamp."""

    def test_discover_configurations_structure(self):
        """
        Verify discover_configurations() returns nested dict of strategy->model->paths.

        The function should scan a timestamp directory and return a structure like:
        {
            'memory_bank': {
                'claude-sonnet-4-5': {
                    'trace_path': Path(...),
                    'metrics_path': Path(...)
                }
            },
            'truncation': {...}
        }
        """
        configs = discover_configurations("test_ace", "20260203_1039")
        assert isinstance(configs, dict)
        assert len(configs) > 0
        # Check structure
        for strategy, models in configs.items():
            assert isinstance(strategy, str)
            assert isinstance(models, dict)
            for model, paths in models.items():
                assert isinstance(model, str)
                assert "trace_path" in paths
                assert "metrics_path" in paths

    def test_discover_configurations_finds_known_strategies(self):
        """
        Verify discover_configurations() finds expected strategies.

        The 20260203_1039 timestamp should contain memory_bank, truncation,
        and progressive_summarization strategies based on the actual data.
        """
        configs = discover_configurations("test_ace", "20260203_1039")
        # At least one of these should exist
        known_strategies = {"memory_bank", "truncation", "progressive_summarization"}
        found_strategies = set(configs.keys())
        assert found_strategies & known_strategies, (
            "Should find at least one known strategy"
        )

    def test_discover_configurations_excludes_temp(self):
        """
        Verify discover_configurations() excludes 'temp' directory.

        The temp folder should not be treated as a strategy.
        """
        configs = discover_configurations("test_ace", "20260203_1039")
        assert "temp" not in configs

    def test_discover_configurations_invalid_path_returns_empty(self):
        """
        Verify discover_configurations() returns empty dict for invalid paths.
        """
        configs = discover_configurations("nonexistent", "invalid")
        assert configs == {}


class TestDataLoading:
    """Tests for loading and parsing trace/metrics JSON files."""

    def test_load_trace_file_parses_cases(self):
        """
        Verify load_trace_file() correctly parses a trace JSON file.

        The trace file contains an array of case objects. Each case should have
        at minimum: id, gen_convs (conversation), status, and memory_method.
        """
        # Find a known trace file
        configs = discover_configurations("test_ace", "20260203_1039")
        assert configs, "Need at least one configuration to test"

        # Get first available trace path
        strategy = next(iter(configs))
        model = next(iter(configs[strategy]))
        trace_path = configs[strategy][model]["trace_path"]

        cases = load_trace_file(trace_path)
        assert isinstance(cases, list)
        assert len(cases) > 0

        # Verify case structure
        case = cases[0]
        assert "id" in case
        assert "gen_convs" in case
        assert isinstance(case["gen_convs"], list)

    def test_load_trace_file_nonexistent_returns_empty(self):
        """
        Verify load_trace_file() returns empty list for missing files.

        Rather than raising FileNotFoundError, gracefully return empty list.
        """
        cases = load_trace_file(Path("/nonexistent/path.json"))
        assert cases == []

    def test_load_metrics_file_parses_metrics(self):
        """
        Verify load_metrics_file() correctly parses a metrics JSON file.

        The metrics file contains key-value pairs for various metrics like
        overall_success, domain_success_rate, etc.
        """
        configs = discover_configurations("test_ace", "20260203_1039")
        assert configs, "Need at least one configuration to test"

        strategy = next(iter(configs))
        model = next(iter(configs[strategy]))
        metrics_path = configs[strategy][model]["metrics_path"]

        if metrics_path and metrics_path.exists():
            metrics = load_metrics_file(metrics_path)
            assert isinstance(metrics, dict)
            # Should have at least some metrics
            assert len(metrics) > 0

    def test_load_metrics_file_nonexistent_returns_none(self):
        """
        Verify load_metrics_file() returns None for missing files.

        Metrics files are optional - not all runs generate them.
        """
        metrics = load_metrics_file(Path("/nonexistent/metrics.json"))
        assert metrics is None


class TestCaseIndexing:
    """Tests for building the case_id -> traces index."""

    def test_build_case_index_groups_by_case_id(self):
        """
        Verify build_case_index() creates mapping from case_id to LoadedTrace list.

        The index enables side-by-side comparison by grouping traces from
        different model/strategy combinations by their case_id (e.g., 'Hotels-104').
        """
        index = build_case_index("test_ace", "20260203_1039")
        assert isinstance(index, dict)
        assert len(index) > 0

        # Each key should be a case_id string
        for case_id, traces in index.items():
            assert isinstance(case_id, str)
            assert isinstance(traces, list)
            assert len(traces) > 0
            # Each trace should be a LoadedTrace
            for trace in traces:
                assert isinstance(trace, LoadedTrace)

    def test_build_case_index_loaded_trace_structure(self):
        """
        Verify LoadedTrace dataclass contains expected fields.

        LoadedTrace should have: strategy, model, case (the case dict),
        and optionally metrics.
        """
        index = build_case_index("test_ace", "20260203_1039")
        assert index, "Need at least one case"

        case_id = next(iter(index))
        trace = index[case_id][0]

        assert hasattr(trace, "strategy")
        assert hasattr(trace, "model")
        assert hasattr(trace, "case")
        assert hasattr(trace, "metrics")

        # Verify types
        assert isinstance(trace.strategy, str)
        assert isinstance(trace.model, str)
        assert isinstance(trace.case, dict)
        assert trace.metrics is None or isinstance(trace.metrics, dict)

    def test_build_case_index_multiple_models_same_case(self):
        """
        Verify build_case_index() groups same case_id from different models.

        If both memory_bank/claude and truncation/claude processed 'Hotels-104',
        both should appear in index['Hotels-104'].
        """
        index = build_case_index("test_ace", "20260203_1039")

        # Find a case that might have multiple traces
        multi_trace_cases = [cid for cid, traces in index.items() if len(traces) > 1]

        if multi_trace_cases:
            case_id = multi_trace_cases[0]
            traces = index[case_id]
            # Verify they have different strategy/model combinations
            configs = [(t.strategy, t.model) for t in traces]
            assert len(configs) == len(set(configs)), (
                "Each trace should be unique config"
            )

    def test_build_case_index_invalid_path_returns_empty(self):
        """
        Verify build_case_index() returns empty dict for invalid experiment/timestamp.
        """
        index = build_case_index("nonexistent", "invalid")
        assert index == {}


class TestConversationParsing:
    """Tests for parsing conversation structure within traces."""

    def test_conversation_has_expected_roles(self):
        """
        Verify gen_convs contains messages with roles: user, assistant, observation.

        The conversation trace should have a structured sequence of messages
        with defined roles for rendering in the chat UI.
        """
        index = build_case_index("test_ace", "20260203_1039")
        assert index, "Need at least one case"

        case_id = next(iter(index))
        trace = index[case_id][0]
        gen_convs = trace.case.get("gen_convs", [])

        assert len(gen_convs) > 0
        roles = {msg.get("role") for msg in gen_convs}
        # Should have at least user and assistant
        assert "user" in roles or "assistant" in roles

    def test_assistant_messages_have_function_calls(self):
        """
        Verify assistant messages contain function_call array when making tool calls.

        Assistant messages that invoke tools should have a function_call array
        with name and arguments for each tool invocation.
        """
        index = build_case_index("test_ace", "20260203_1039")
        case_id = next(iter(index))
        trace = index[case_id][0]
        gen_convs = trace.case.get("gen_convs", [])

        # Find an assistant message with function calls
        assistant_msgs = [m for m in gen_convs if m.get("role") == "assistant"]
        func_call_msgs = [m for m in assistant_msgs if m.get("function_call")]

        if func_call_msgs:
            msg = func_call_msgs[0]
            func_calls = msg["function_call"]
            assert isinstance(func_calls, list)
            for fc in func_calls:
                assert "name" in fc
                assert "arguments" in fc

    def test_observation_messages_have_content(self):
        """
        Verify observation messages contain tool execution results.

        Observation messages should have a content field with the results
        of tool execution, including status, message, and data.
        """
        index = build_case_index("test_ace", "20260203_1039")
        case_id = next(iter(index))
        trace = index[case_id][0]
        gen_convs = trace.case.get("gen_convs", [])

        observation_msgs = [m for m in gen_convs if m.get("role") == "observation"]

        if observation_msgs:
            msg = observation_msgs[0]
            assert "content" in msg
            content = msg["content"]
            # Content is typically a list of results
            assert isinstance(content, list)
