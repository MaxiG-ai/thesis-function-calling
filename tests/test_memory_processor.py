from types import SimpleNamespace

from src.memory_processing import MemoryProcessor
from src.utils.config import ExperimentConfig, MemoryDef
from src.utils.trace_processing import detect_tail_loop


def _build_config(threshold: int) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name="test",
        results_dir=".",
        log_dir=".",
        logging_level="DEBUG",
        debug=False,
        input_file="input",
        enabled_models=[],
        enabled_memory_methods=["progressive"],
        max_tokens=8000,
        memory_strategies={
            "progressive": MemoryDef(
                type="progressive_summarization",
                auto_compact_threshold=threshold,
                summarizer_model="gpt-5-mini",
            )
        },
        model_registry={},
    )


def _build_messages() -> list[dict]:
    return [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "First ask."},
        {"role": "assistant", "content": "Working through."},
        {"role": "user", "content": "Second ask."},
    ]


class FakeOrchestrator:
    def __init__(self, summary: str):
        self.summary = summary
        self.call_count = 0

    def generate_plain(self, input_messages, **kwargs):
        self.call_count += 1
        return SimpleNamespace(choices=[SimpleNamespace(message={"content": self.summary})])


class FailingOrchestrator:
    def generate_plain(self, *args, **kwargs):
        raise RuntimeError("model down")


def test_progressive_summarization_injects_summary_and_is_idempotent() -> None:
    config = _build_config(threshold=1)
    processor = MemoryProcessor(config)
    orchestrator = FakeOrchestrator("compacted summary")
    messages = _build_messages()

    # Get settings and call directly since we're testing the method itself
    settings = config.memory_strategies["progressive"]
    first_view = processor._apply_progressive_summarization(
        messages, token_count=1000, settings=settings, llm_client=orchestrator
    )
    
    # Check structure: system message + summary message + working memory
    assert first_view[0]["role"] == "system"
    assert first_view[0]["content"] == "System"
    assert first_view[1]["role"] == "system"
    assert first_view[1]["content"] == "compacted summary"
    assert first_view[2]["content"] == "Second ask."

    # Call again - should create new summary (no idempotency in full re-summarization)
    second_view = processor._apply_progressive_summarization(
        messages, token_count=1000, settings=settings, llm_client=orchestrator
    )
    # Called twice now (once per invocation)
    assert orchestrator.call_count == 2
    assert second_view[1]["content"] == "compacted summary"


def test_progressive_summarization_fails_back_to_full_history() -> None:
    config = _build_config(threshold=1)
    processor = MemoryProcessor(config)
    failing_orchestrator = FailingOrchestrator()
    messages = _build_messages()

    # Now exceptions propagate instead of silent fallback
    settings = config.memory_strategies["progressive"]
    try:
        processor._apply_progressive_summarization(
            messages, token_count=1000, settings=settings, llm_client=failing_orchestrator
        )
        assert False, "Expected RuntimeError to propagate"
    except RuntimeError as e:
        assert "model down" in str(e)


def test_detect_tail_loop_with_dict_style_tool_calls() -> None:
    """Test that detect_tail_loop correctly handles dict-style tool_calls (litellm format)."""
    # Create messages with dict-style tool_calls
    messages = [
        {"role": "user", "content": "Please help"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_data",
                        "arguments": '{"query": "test"}'
                    }
                }
            ]
        },
        {"role": "tool", "content": "result", "tool_call_id": "call_1"},
        {"role": "user", "content": "Please help"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_data",
                        "arguments": '{"query": "test"}'
                    }
                }
            ]
        },
        {"role": "tool", "content": "result", "tool_call_id": "call_2"},
        {"role": "user", "content": "Please help"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_3",
                    "type": "function",
                    "function": {
                        "name": "get_data",
                        "arguments": '{"query": "test"}'
                    }
                }
            ]
        },
        {"role": "tool", "content": "result", "tool_call_id": "call_3"},
        {"role": "user", "content": "Please help"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_4",
                    "type": "function",
                    "function": {
                        "name": "get_data",
                        "arguments": '{"query": "test"}'
                    }
                }
            ]
        },
        {"role": "tool", "content": "result", "tool_call_id": "call_4"},
    ]
    
    # Should detect the loop (pattern of 3 messages repeating 4 times)
    assert detect_tail_loop(messages, threshold=4, max_pattern_len=5) is True


def test_detect_tail_loop_no_loop_with_dict_tool_calls() -> None:
    """Test that detect_tail_loop returns False when there's no loop."""
    messages = [
        {"role": "user", "content": "First question"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_data",
                        "arguments": '{"query": "first"}'
                    }
                }
            ]
        },
        {"role": "tool", "content": "result1", "tool_call_id": "call_1"},
        {"role": "user", "content": "Second question"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_other_data",
                        "arguments": '{"query": "second"}'
                    }
                }
            ]
        },
        {"role": "tool", "content": "result2", "tool_call_id": "call_2"},
    ]
    
    # Should not detect a loop
    assert detect_tail_loop(messages, threshold=4, max_pattern_len=5) is False


def test_detect_tail_loop_short_history() -> None:
    """Test that detect_tail_loop handles short message history correctly."""
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    
    # Too short to contain a loop
    assert detect_tail_loop(messages, threshold=4, max_pattern_len=5) is False
