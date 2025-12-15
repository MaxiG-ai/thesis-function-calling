from typing import List

from src.utils.history import segment_message_history


def _make_message(role: str, content: str, **extras) -> dict:
    message = {
        "role": role,
        "content": content,
    }
    message.update(extras)
    return message


def test_working_memory_preserves_tool_chain_and_last_user() -> None:
    messages: List[dict] = [
        _make_message("system", "System prompt"),
        _make_message("user", "Initial ask"),
        _make_message(
            "assistant",
            "Calling data tool",
            tool_calls=[{"type": "tool_call", "function": {"name": "fetch"}, "arguments": {"query": "latest"}}],
        ),
        _make_message("tool", "Tool result", tool_call_id="tc-one"),
    ]

    result = segment_message_history(messages)
    working_memory = result["working_memory"]

    assert working_memory[0]["role"] == "user"
    assert working_memory[0]["content"] == "Initial ask"
    assert any(msg["role"] == "assistant" and "tool_calls" in msg for msg in working_memory)
    assert any(msg["role"] == "tool" for msg in working_memory)


def test_archived_context_excludes_system_and_tail() -> None:
    messages = [
        _make_message("system", "System prompt"),
        _make_message("user", "Background"),
        _make_message("assistant", "Reviewing prior information"),
        _make_message("user", "Current ask"),
    ]

    result = segment_message_history(messages)

    assert result["system_message"][0]["role"] == "system"
    assert all(msg in messages for msg in result["archived_context"])
    assert all(msg in messages for msg in result["working_memory"])
    assert result["working_memory"][-1]["role"] == "user"
    assert all(msg not in result["working_memory"] for msg in result["archived_context"])
