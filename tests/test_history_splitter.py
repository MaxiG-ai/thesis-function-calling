from typing import List

from src.utils.split_trace import (
    get_user_message,
    get_last_tool_interaction,
    process_and_split_trace_user,
    process_and_split_trace_user_tool,
)


def _make_message(role: str, content: str, **extras) -> dict:
    message = {
        "role": role,
        "content": content,
    }
    message.update(extras)
    return message


# Tests for get_user_message
def test_get_user_message_single_user() -> None:
    """Test extracting a single user message from trace."""
    messages = [
        _make_message("system", "System prompt"),
        _make_message("user", "Hello"),
        _make_message("assistant", "Hi there!"),
    ]
    
    result = get_user_message(messages)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "Hello"


def test_get_user_message_no_user() -> None:
    """Test when no user message exists."""
    messages = [
        _make_message("system", "System prompt"),
        _make_message("assistant", "Hi there!"),
    ]
    
    result = get_user_message(messages)
    assert result == []


def test_get_user_message_empty() -> None:
    """Test with empty messages list."""
    result = get_user_message([])
    assert result == []


def test_get_user_message_multiple_users() -> None:
    """Test extracting all user messages from trace."""
    messages = [
        _make_message("user", "First question"),
        _make_message("assistant", "First answer"),
        _make_message("user", "Second question"),
        _make_message("assistant", "Second answer"),
    ]
    
    result = get_user_message(messages)
    assert len(result) == 2
    assert result[0]["content"] == "First question"
    assert result[1]["content"] == "Second question"


# Tests for get_last_tool_interaction
def test_get_last_tool_interaction_valid() -> None:
    """Test extracting a valid tool episode."""
    messages = [
        _make_message("user", "Get data"),
        _make_message(
            "assistant",
            "Fetching...",
            tool_calls=[{"id": "tc-1", "type": "function", "function": {"name": "fetch"}}],
        ),
        _make_message("tool", "Result 1", tool_call_id="tc-1"),
    ]
    
    result = get_last_tool_interaction(messages)
    assert len(result) == 2  # assistant + 1 tool response
    assert result[0]["role"] == "assistant"
    assert "tool_calls" in result[0]
    assert result[1]["role"] == "tool"


def test_get_last_tool_interaction_no_tools() -> None:
    """Test when no tool messages exist at end."""
    messages = [
        _make_message("user", "Hello"),
        _make_message("assistant", "Hi there!"),
    ]
    
    result = get_last_tool_interaction(messages)
    assert result == []


def test_get_last_tool_interaction_empty() -> None:
    """Test with empty messages list."""
    result = get_last_tool_interaction([])
    assert result == []


def test_get_last_tool_interaction_corrupted_type_field() -> None:
    """Test handling corrupted tool_calls with _type field."""
    messages = [
        _make_message("user", "Get data"),
        _make_message(
            "assistant",
            "Fetching...",
            tool_calls=[{"_type": "ToolCall", "id": "tc-1", "function": {"name": "fetch"}}],
        ),
        _make_message("tool", "Result 1", tool_call_id="tc-1"),
    ]
    
    result = get_last_tool_interaction(messages)
    assert result == []  # Should treat as invalid


def test_get_last_tool_interaction_multiple_parallel() -> None:
    """Test multiple parallel tool calls from single assistant message."""
    messages = [
        _make_message("user", "Get multiple data"),
        _make_message(
            "assistant",
            "Fetching multiple...",
            tool_calls=[
                {"id": "tc-1", "type": "function", "function": {"name": "fetch1"}},
                {"id": "tc-2", "type": "function", "function": {"name": "fetch2"}},
            ],
        ),
        _make_message("tool", "Result 1", tool_call_id="tc-1"),
        _make_message("tool", "Result 2", tool_call_id="tc-2"),
    ]
    
    result = get_last_tool_interaction(messages)
    assert len(result) == 3  # 1 assistant + 2 tool responses
    assert result[0]["role"] == "assistant"
    assert len(result[0]["tool_calls"]) == 2
    assert result[1]["role"] == "tool"
    assert result[2]["role"] == "tool"


def test_get_last_tool_interaction_tool_id_mismatch() -> None:
    """Test when tool response IDs don't match assistant's tool_call IDs."""
    messages = [
        _make_message("user", "Get data"),
        _make_message(
            "assistant",
            "Fetching...",
            tool_calls=[{"id": "tc-1", "type": "function", "function": {"name": "fetch"}}],
        ),
        _make_message("tool", "Result", tool_call_id="tc-999"),  # Mismatched ID
    ]
    
    result = get_last_tool_interaction(messages)
    assert result == []  # Should treat as invalid


def test_get_last_tool_interaction_no_preceding_assistant() -> None:
    """Test when tool messages have no preceding assistant message."""
    messages = [
        _make_message("user", "Get data"),
        _make_message("tool", "Orphan tool result", tool_call_id="tc-1"),
    ]
    
    result = get_last_tool_interaction(messages)
    assert result == []


# Tests for process_and_split_trace_user
def test_process_and_split_trace_user_simple() -> None:
    """Test simple 2-way split (user + rest)."""
    messages = [
        _make_message("system", "System prompt"),
        _make_message("user", "Hello"),
        _make_message("assistant", "Hi there!"),
        _make_message("assistant", "How can I help?"),
    ]
    
    user_messages, rest = process_and_split_trace_user(messages)
    
    assert len(user_messages) == 1
    assert user_messages[0]["role"] == "user"
    assert len(rest) == 3  # system + 2 assistants


def test_process_and_split_trace_user_no_user() -> None:
    """Test when no user message exists."""
    messages = [
        _make_message("system", "System prompt"),
        _make_message("assistant", "Hello"),
    ]
    
    user_messages, rest = process_and_split_trace_user(messages)
    
    assert user_messages == []
    assert len(rest) == 2


def test_process_and_split_trace_user_empty() -> None:
    """Test with empty messages."""
    user_messages, rest = process_and_split_trace_user([])
    
    assert user_messages == []
    assert rest == []


# Tests for process_and_split_trace_user_tool
def test_process_and_split_trace_user_tool_3way() -> None:
    """Test 3-way split (user + intermediate + last tool episode)."""
    messages = [
        _make_message("system", "System prompt"),
        _make_message("user", "First question"),
        _make_message("assistant", "First answer"),
        _make_message("user", "Get data"),
        _make_message(
            "assistant",
            "Fetching...",
            tool_calls=[{"id": "tc-1", "type": "function", "function": {"name": "fetch"}}],
        ),
        _make_message("tool", "Result", tool_call_id="tc-1"),
    ]
    
    user_messages, intermediate, tool_episode = process_and_split_trace_user_tool(messages)
    
    assert len(user_messages) == 2
    assert user_messages[0]["content"] == "First question"
    assert user_messages[1]["content"] == "Get data"
    assert len(intermediate) == 2  # system + first assistant
    assert len(tool_episode) == 2  # assistant with tool_calls + tool response


def test_process_and_split_trace_user_tool_no_tools() -> None:
    """Test 3-way split when no tools at end."""
    messages = [
        _make_message("user", "Hello"),
        _make_message("assistant", "Hi there!"),
    ]
    
    user_messages, intermediate, tool_episode = process_and_split_trace_user_tool(messages)
    
    assert len(user_messages) == 1
    assert len(intermediate) == 1
    assert tool_episode == []


def test_process_and_split_trace_user_tool_empty() -> None:
    """Test 3-way split with empty messages."""
    user_messages, intermediate, tool_episode = process_and_split_trace_user_tool([])
    
    assert user_messages == []
    assert intermediate == []
    assert tool_episode == []


def test_process_and_split_trace_user_tool_corrupted_tools() -> None:
    """Test 3-way split with corrupted tool calls."""
    messages = [
        _make_message("user", "Get data"),
        _make_message(
            "assistant",
            "Fetching...",
            tool_calls=[{"_type": "ToolCall", "id": "tc-1"}],  # Corrupted
        ),
        _make_message("tool", "Result", tool_call_id="tc-1"),
    ]
    
    user_messages, intermediate, tool_episode = process_and_split_trace_user_tool(messages)
    
    assert len(user_messages) == 1
    assert len(intermediate) == 2  # user already counted in user_messages, so assistant + tool
    assert tool_episode == []  # Corrupted tools should be excluded
