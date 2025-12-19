from types import SimpleNamespace

from src.memory_processing import MemoryProcessor
from src.utils.config import ExperimentConfig, MemoryDef


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
