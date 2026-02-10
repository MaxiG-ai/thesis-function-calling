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
from pathlib import Path

import pytest

from tools.trace_viewer import (
    RESULTS_ROOT,
    LoadedTrace,
    build_case_index,
    discover_configurations,
    list_experiments,
    list_timestamps,
    load_metrics_file,
    load_trace_file,
    system_message_markdown_content,
)


# === FIXTURES ===
@pytest.fixture
def sample_experiment():
    """Known experiment name for testing."""
    return "test_ace"


@pytest.fixture
def sample_timestamp():
    """Known timestamp for testing."""
    return "20260203_1039"


@pytest.fixture
def sample_configs(sample_experiment, sample_timestamp):
    """Load configurations from test_ace experiment."""
    return discover_configurations(sample_experiment, sample_timestamp)


@pytest.fixture
def sample_trace_path(sample_configs):
    """Get first available trace path for testing."""
    if not sample_configs:
        pytest.skip("No configurations available for testing")
    strategy = next(iter(sample_configs))
    model = next(iter(sample_configs[strategy]))
    return sample_configs[strategy][model]["trace_path"]


@pytest.fixture
def sample_case_index(sample_experiment, sample_timestamp):
    """Build case index from test_ace experiment."""
    return build_case_index(sample_experiment, sample_timestamp)


@pytest.fixture
def tmp_results_root(tmp_path):
    """
    Create a temporary results/cfb directory tree for isolated tests.

    This fixture builds a dedicated, empty root directory that mirrors the
    expected layout (results/cfb) so tests can create synthetic trace files
    without relying on the repository's real data.
    """
    root = tmp_path / "results" / "cfb"
    root.mkdir(parents=True, exist_ok=True)
    return root


# === DIRECTORY DISCOVERY TESTS ===
def test_list_experiments_returns_folder_names():
    """
    Verify list_experiments() returns experiment folder names from results/cfb.

    The function should scan the RESULTS_ROOT directory and return a list of
    subdirectory names that represent different experiments (e.g., 'test_ace',
    'test_mcpbench'). It should exclude files and hidden directories.
    """
    experiments = list_experiments()
    assert isinstance(experiments, list)
    assert len(experiments) > 0
    assert "test_ace" in experiments


def test_list_experiments_excludes_files():
    """
    Verify list_experiments() only returns directories, not files.

    The results/cfb directory may contain log files or other non-directory
    entries. These should be filtered out.
    """
    experiments = list_experiments()
    for exp in experiments:
        exp_path = RESULTS_ROOT / exp
        assert exp_path.is_dir(), f"{exp} should be a directory"


def test_list_timestamps_returns_sorted_timestamps(sample_experiment):
    """
    Verify list_timestamps() returns timestamp folders in descending order.

    Timestamps follow the format YYYYMMDD_HHMM. The function should return
    them sorted newest-first for easier navigation in the UI.
    """
    timestamps = list_timestamps(sample_experiment)
    assert isinstance(timestamps, list)
    assert len(timestamps) > 0
    # Verify format: YYYYMMDD_HHMM
    for ts in timestamps:
        assert len(ts) == 13, f"Timestamp {ts} should be 13 chars (YYYYMMDD_HHMM)"
        assert ts[8] == "_", f"Timestamp {ts} should have underscore at position 8"
    # Verify sorted descending (newest first)
    assert timestamps == sorted(timestamps, reverse=True)


def test_list_timestamps_invalid_experiment_returns_empty():
    """
    Verify list_timestamps() returns empty list for non-existent experiment.

    Rather than raising an exception, the function should gracefully return
    an empty list when the experiment folder doesn't exist.
    """
    timestamps = list_timestamps("nonexistent_experiment_xyz")
    assert timestamps == []


def test_list_timestamps_excludes_temp_folders(sample_experiment):
    """
    Verify list_timestamps() excludes 'temp' and other non-timestamp folders.

    Some timestamp directories contain a 'temp' subfolder for intermediate
    files. This should not appear in the timestamps list.
    """
    timestamps = list_timestamps(sample_experiment)
    assert "temp" not in timestamps
    for ts in timestamps:
        assert ts[0:8].isdigit(), f"{ts} should start with 8 digits"


# === CONFIGURATION DISCOVERY TESTS ===
def test_discover_configurations_structure(sample_configs):
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
    assert isinstance(sample_configs, dict)
    assert len(sample_configs) > 0
    for strategy, models in sample_configs.items():
        assert isinstance(strategy, str)
        assert isinstance(models, dict)
        for model, paths in models.items():
            assert isinstance(model, str)
            assert "trace_path" in paths
            assert "compressed_path" in paths
            assert "metrics_path" in paths


def test_discover_configurations_finds_known_strategies(sample_configs):
    """
    Verify discover_configurations() finds expected strategies.

    The 20260203_1039 timestamp should contain memory_bank, truncation,
    and progressive_summarization strategies based on the actual data.
    """
    known_strategies = {"memory_bank", "truncation", "progressive_summarization"}
    found_strategies = set(sample_configs.keys())
    assert found_strategies & known_strategies, (
        "Should find at least one known strategy"
    )


def test_discover_configurations_excludes_temp(sample_configs):
    """
    Verify discover_configurations() excludes 'temp' directory.

    The temp folder should not be treated as a strategy.
    """
    assert "temp" not in sample_configs


def test_discover_configurations_invalid_path_returns_empty():
    """Verify discover_configurations() returns empty dict for invalid paths."""
    configs = discover_configurations("nonexistent", "invalid")
    assert configs == {}


# === DATA LOADING TESTS ===
def test_load_trace_file_parses_cases(sample_trace_path):
    """
    Verify load_trace_file() correctly parses a trace JSON file.

    The trace file contains an array of case objects. Each case should have
    at minimum: id, gen_convs (conversation), status, and memory_method.
    """
    cases = load_trace_file(sample_trace_path)
    assert isinstance(cases, list)
    assert len(cases) > 0

    case = cases[0]
    assert "id" in case
    assert "gen_convs" in case
    assert isinstance(case["gen_convs"], list)


def test_load_trace_file_nonexistent_returns_empty():
    """
    Verify load_trace_file() returns empty list for missing files.

    Rather than raising FileNotFoundError, gracefully return empty list.
    """
    cases = load_trace_file(Path("/nonexistent/path.json"))
    assert cases == []


def test_load_metrics_file_parses_metrics(sample_configs):
    """
    Verify load_metrics_file() correctly parses a metrics JSON file.

    The metrics file contains key-value pairs for various metrics like
    overall_success, domain_success_rate, etc.
    """
    strategy = next(iter(sample_configs))
    model = next(iter(sample_configs[strategy]))
    metrics_path = sample_configs[strategy][model]["metrics_path"]

    if metrics_path and metrics_path.exists():
        metrics = load_metrics_file(metrics_path)
        assert isinstance(metrics, dict)
        assert len(metrics) > 0


def test_load_metrics_file_nonexistent_returns_none():
    """
    Verify load_metrics_file() returns None for missing files.

    Metrics files are optional - not all runs generate them.
    """
    metrics = load_metrics_file(Path("/nonexistent/metrics.json"))
    assert metrics is None


# === CASE INDEXING TESTS ===
def test_build_case_index_groups_by_case_id(sample_case_index):
    """
    Verify build_case_index() creates mapping from case_id to LoadedTrace list.

    The index enables side-by-side comparison by grouping traces from
    different model/strategy combinations by their case_id (e.g., 'Hotels-104').
    """
    assert isinstance(sample_case_index, dict)
    assert len(sample_case_index) > 0

    for case_id, traces in sample_case_index.items():
        assert isinstance(case_id, str)
        assert isinstance(traces, list)
        assert len(traces) > 0
        for trace in traces:
            assert isinstance(trace, LoadedTrace)


def test_build_case_index_loaded_trace_structure(sample_case_index):
    """
    Verify LoadedTrace dataclass contains expected fields.

    LoadedTrace should have: strategy, model, case (the case dict),
    and optionally metrics.
    """
    assert sample_case_index, "Need at least one case"

    case_id = next(iter(sample_case_index))
    trace = sample_case_index[case_id][0]

    assert hasattr(trace, "strategy")
    assert hasattr(trace, "model")
    assert hasattr(trace, "case")
    assert hasattr(trace, "metrics")

    assert isinstance(trace.strategy, str)
    assert isinstance(trace.model, str)
    assert isinstance(trace.case, dict)
    assert trace.metrics is None or isinstance(trace.metrics, dict)


def test_build_case_index_multiple_models_same_case(sample_case_index):
    """
    Verify build_case_index() groups same case_id from different models.

    If both memory_bank/claude and truncation/claude processed 'Hotels-104',
    both should appear in index['Hotels-104'].
    """
    multi_trace_cases = [
        cid for cid, traces in sample_case_index.items() if len(traces) > 1
    ]

    if multi_trace_cases:
        case_id = multi_trace_cases[0]
        traces = sample_case_index[case_id]
        configs = [(t.strategy, t.model) for t in traces]
        assert len(configs) == len(set(configs)), "Each trace should be unique config"


def test_build_case_index_invalid_path_returns_empty():
    """Verify build_case_index() returns empty dict for invalid experiment/timestamp."""
    index = build_case_index("nonexistent", "invalid")
    assert index == {}


def test_discover_configurations_includes_compressed_trace(tmp_results_root):
    """
    Verify discover_configurations() captures both cfb and compressed files.

    The trace viewer should be able to switch between standard eval traces
    (cfb_*.json) and compressed traces (compressed_*.json). This test creates
    both files in a synthetic strategy/model directory and ensures their
    paths are returned in the discovered configuration structure.
    """
    experiment = "demo"
    timestamp = "20260210_1206"
    strategy = "memory_bank"
    model = "gpt-4-1-mini"

    model_dir = tmp_results_root / experiment / timestamp / strategy / model
    model_dir.mkdir(parents=True, exist_ok=True)

    trace_path = model_dir / "cfb_demo.json"
    compressed_path = model_dir / "compressed_demo.json"
    trace_path.write_text("[]", encoding="utf-8")
    compressed_path.write_text("[]", encoding="utf-8")

    configs = discover_configurations(experiment, timestamp, root=tmp_results_root)
    assert configs[strategy][model]["trace_path"] == trace_path
    assert configs[strategy][model]["compressed_path"] == compressed_path


def test_build_case_index_uses_eval_trace(tmp_results_root):
    """
    Verify build_case_index() loads eval conversations from cfb files.

    When the conversation source is set to eval, the trace viewer should load
    cases from cfb_*.json and preserve the gen_convs content without using
    any compressed data, even if a compressed file is present.
    """
    experiment = "demo"
    timestamp = "20260210_1206"
    strategy = "memory_bank"
    model = "gpt-4-1-mini"
    model_dir = tmp_results_root / experiment / timestamp / strategy / model
    model_dir.mkdir(parents=True, exist_ok=True)

    eval_case = {
        "id": "Case-1",
        "gen_convs": [{"role": "user", "content": "eval message"}],
        "status": "Success",
        "memory_method": "memory_bank",
    }
    compressed_case = {
        "id": "Case-1",
        "memory_method": "memory_bank",
        "compressed_trace": [
            {
                "step": 1,
                "compressed_messages": [
                    {"role": "user", "content": "compressed message"}
                ],
            }
        ],
    }

    (model_dir / "cfb_demo.json").write_text(json.dumps([eval_case]), encoding="utf-8")
    (model_dir / "compressed_demo.json").write_text(
        json.dumps([compressed_case]), encoding="utf-8"
    )

    index = build_case_index(
        experiment, timestamp, root=tmp_results_root, conversation_source="eval"
    )
    trace = index["Case-1"][0]
    assert trace.case["gen_convs"][0]["content"] == "eval message"


def test_build_case_index_uses_compressed_trace(tmp_results_root):
    """
    Verify build_case_index() loads real conversations from compressed files.

    When the conversation source is set to real, the trace viewer should load
    cases from compressed_*.json and expand compressed_trace into gen_convs
    so the UI can render the actual messages sent to the model.
    """
    experiment = "demo"
    timestamp = "20260210_1206"
    strategy = "memory_bank"
    model = "gpt-4-1-mini"
    model_dir = tmp_results_root / experiment / timestamp / strategy / model
    model_dir.mkdir(parents=True, exist_ok=True)

    eval_case = {
        "id": "Case-1",
        "gen_convs": [{"role": "user", "content": "eval message"}],
    }
    compressed_case = {
        "id": "Case-1",
        "memory_method": "memory_bank",
        "compressed_trace": [
            {
                "step": 1,
                "compressed_messages": [
                    {"role": "user", "content": "compressed message"}
                ],
            }
        ],
    }

    (model_dir / "cfb_demo.json").write_text(json.dumps([eval_case]), encoding="utf-8")
    (model_dir / "compressed_demo.json").write_text(
        json.dumps([compressed_case]), encoding="utf-8"
    )

    index = build_case_index(
        experiment, timestamp, root=tmp_results_root, conversation_source="real"
    )
    trace = index["Case-1"][0]
    first_message = trace.case["gen_convs"][0]
    assert first_message["role"] == "system"
    assert "Compressed step" in first_message["content"]
    assert trace.case["gen_convs"][1]["content"] == "compressed message"


# === CONVERSATION PARSING TESTS ===
def test_system_message_markdown_content_prefers_content_key():
    """
    Verify system_message_markdown_content() extracts markdown from content key.

    System messages in traces can encode content as a JSON object with a
    nested "content" field. The helper should return that nested string so
    the UI renders markdown instead of dumping raw JSON.
    """
    msg = {"role": "system", "content": {"content": "# Heading\n\nBody"}}
    assert system_message_markdown_content(msg) == "# Heading\n\nBody"


def test_system_message_markdown_content_handles_plain_string():
    """
    Verify system_message_markdown_content() returns string content unchanged.

    When the system message content is already a plain string, the helper
    should pass it through so rendering does not alter the text.
    """
    msg = {"role": "system", "content": "Plain system content"}
    assert system_message_markdown_content(msg) == "Plain system content"


def test_system_message_markdown_content_parses_json_string():
    """
    Verify system_message_markdown_content() parses JSON strings when needed.

    Some traces store system content as a JSON string containing a content
    field. The helper should decode it and return the nested markdown so the
    UI renders it without showing raw JSON.
    """
    msg = {"role": "system", "content": '{"content": "## Title\nText"}'}
    assert system_message_markdown_content(msg) == "## Title\nText"


def test_conversation_has_expected_roles(sample_case_index):
    """
    Verify gen_convs contains messages with roles: user, assistant, observation.

    The conversation trace should have a structured sequence of messages
    with defined roles for rendering in the chat UI.
    """
    assert sample_case_index, "Need at least one case"

    case_id = next(iter(sample_case_index))
    trace = sample_case_index[case_id][0]
    gen_convs = trace.case.get("gen_convs", [])

    assert len(gen_convs) > 0
    roles = {msg.get("role") for msg in gen_convs}
    assert "user" in roles or "assistant" in roles


def test_assistant_messages_have_function_calls(sample_case_index):
    """
    Verify assistant messages contain function_call array when making tool calls.

    Assistant messages that invoke tools should have a function_call array
    with name and arguments for each tool invocation.
    """
    case_id = next(iter(sample_case_index))
    trace = sample_case_index[case_id][0]
    gen_convs = trace.case.get("gen_convs", [])

    assistant_msgs = [m for m in gen_convs if m.get("role") == "assistant"]
    func_call_msgs = [m for m in assistant_msgs if m.get("function_call")]

    if func_call_msgs:
        msg = func_call_msgs[0]
        func_calls = msg["function_call"]
        assert isinstance(func_calls, list)
        for fc in func_calls:
            assert "name" in fc
            assert "arguments" in fc


def test_observation_messages_have_content(sample_case_index):
    """
    Verify observation messages contain tool execution results.

    Observation messages should have a content field with the results
    of tool execution, including status, message, and data.
    """
    case_id = next(iter(sample_case_index))
    trace = sample_case_index[case_id][0]
    gen_convs = trace.case.get("gen_convs", [])

    observation_msgs = [m for m in gen_convs if m.get("role") == "observation"]

    if observation_msgs:
        msg = observation_msgs[0]
        assert "content" in msg
        content = msg["content"]
        assert isinstance(content, list)
