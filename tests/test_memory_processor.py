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
    processor = MemoryProcessor(_build_config(threshold=1))
    orchestrator = FakeOrchestrator("compacted summary")
    messages = _build_messages()

    first_view = processor.apply_strategy(
        messages, "progressive", llm_client=orchestrator
    )
    assert first_view[0]["role"] == "system"
    assert first_view[1]["content"] == "compacted summary"

    second_view = processor.apply_strategy(
        messages, "progressive", llm_client=orchestrator
    )
    assert orchestrator.call_count == 1
    assert second_view == first_view


def test_progressive_summarization_fails_back_to_full_history() -> None:
    processor = MemoryProcessor(_build_config(threshold=1))
    failing_orchestrator = FailingOrchestrator()
    messages = _build_messages()

    result = processor.apply_strategy(
        messages, "progressive", llm_client=failing_orchestrator
    )
    assert result == messages
