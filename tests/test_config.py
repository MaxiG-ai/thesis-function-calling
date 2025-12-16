import pytest

from src.utils.config import MemoryDef


def test_memory_def_accepts_auto_compact_threshold():
    definition = MemoryDef(
        type="progressive_summarization",
        auto_compact_threshold=4000,
        summarizer_model="gpt-5-mini",
    )

    assert definition.auto_compact_threshold == 4000
    assert definition.summarizer_model == "gpt-5-mini"
