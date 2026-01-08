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
    
    result, idxs = get_user_message(messages)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "Hello"
    assert len(idxs) == 1
    assert idxs[0] == 1


def test_get_user_message_no_user() -> None:
    """Test when no user message exists."""
    messages = [
        _make_message("system", "System prompt"),
        _make_message("assistant", "Hi there!"),
    ]
    
    result, idxs = get_user_message(messages)
    assert result == []
    assert idxs == []


def test_get_user_message_empty() -> None:
    """Test with empty messages list."""
    result, idxs = get_user_message([])
    assert result == []
    assert idxs == []


def test_get_user_message_multiple_users() -> None:
    """Test extracting all user messages from trace."""
    messages = [
        _make_message("user", "First question"),
        _make_message("assistant", "First answer"),
        _make_message("user", "Second question"),
        _make_message("assistant", "Second answer"),
    ]
    
    result, idxs = get_user_message(messages)
    assert len(result) == 2
    assert result[0]["content"] == "First question"
    assert result[1]["content"] == "Second question"
    assert len(idxs) == 2


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
    
    interaction, idx = get_last_tool_interaction(messages)
    assert len(interaction) == 2  # assistant + 1 tool response
    assert interaction[0]["role"] == "assistant"
    assert "tool_calls" in interaction[0]
    assert interaction[1]["role"] == "tool"
    assert idx == 1


def test_get_last_tool_interaction_no_tools() -> None:
    """Test when no tool messages exist at end."""
    messages = [
        _make_message("user", "Hello"),
        _make_message("assistant", "Hi there!"),
    ]
    
    interaction, idx = get_last_tool_interaction(messages)
    assert interaction == []
    assert idx == len(messages)


def test_get_last_tool_interaction_empty() -> None:
    """Test with empty messages list."""
    interaction, idx = get_last_tool_interaction([])
    assert interaction == []
    assert idx == 0


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
    
    interaction, idx = get_last_tool_interaction(messages)
    assert len(interaction) == 3  # 1 assistant + 2 tool responses
    assert interaction[0]["role"] == "assistant"
    assert len(interaction[0]["tool_calls"]) == 2
    assert interaction[1]["role"] == "tool"
    assert interaction[2]["role"] == "tool"
    assert idx == 1


def test_get_last_tool_interaction_tool_id_mismatch() -> None:
    """Test when tool response IDs don't match assistant's tool_call IDs."""
    # Structure-based parsing accepts this
    messages = [
        _make_message("user", "Get data"),
        _make_message(
            "assistant",
            "Fetching...",
            tool_calls=[{"id": "tc-1", "type": "function", "function": {"name": "fetch"}}],
        ),
        _make_message("tool", "Result", tool_call_id="tc-999"),  # Mismatched ID
    ]
    
    interaction, idx = get_last_tool_interaction(messages)
    assert len(interaction) == 2
    assert interaction[0]["role"] == "assistant"


def test_get_last_tool_interaction_no_preceding_assistant() -> None:
    """Test when tool messages have no preceding assistant message."""
    messages = [
        _make_message("user", "Get data"),
        _make_message("tool", "Orphan tool result", tool_call_id="tc-1"),
    ]
    
    interaction, idx = get_last_tool_interaction(messages)
    assert interaction == []
    assert idx == len(messages)


# Tests for process_and_split_trace_user
def test_process_and_split_trace_user_simple() -> None:
    """Test simple 2-way split (last user + messages before)."""
    messages = [
        _make_message("system", "System prompt"),
        _make_message("user", "First question"),
        _make_message("assistant", "First answer"),
        _make_message("user", "Second question"),
    ]
    
    user_messages, rest = process_and_split_trace_user(messages)
    
    # process_and_split_trace_user returns [user_messages[-1]]
    assert len(user_messages) == 1
    assert user_messages[0]["role"] == "user"
    assert user_messages[0]["content"] == "Second question"
    assert len(rest) == 3  # system + first user + first assistant


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


def test_process_and_split_trace_user_only_user() -> None:
    """Test with only a user message."""
    messages = [_make_message("user", "Hello")]
    
    user_messages, rest = process_and_split_trace_user(messages)
    
    assert len(user_messages) == 1
    assert user_messages[0]["content"] == "Hello"
    assert rest == []


# Tests for process_and_split_trace_user_tool
def test_process_and_split_trace_user_tool_3way() -> None:
    """Test 3-way split (last user + intermediate + last tool episode)."""
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
    
    # According to new logic, returns First User Message
    assert len(user_messages) == 1
    assert user_messages[0]["content"] == "First question"
    # Intermediate: between first user and tool start (idx=4)
    # First user at 1. Intermediate start 2. End 4.
    # [Assistant "First answer", User "Get data"]
    assert len(intermediate) == 2 
    assert len(tool_episode) == 2


def test_process_and_split_trace_user_tool_no_tools() -> None:
    """Test 3-way split when no tools at end."""
    messages = [
        _make_message("system", "System prompt"),
        _make_message("user", "Hello"),
        _make_message("assistant", "Hi there!"),
    ]
    
    user_messages, intermediate, tool_episode = process_and_split_trace_user_tool(messages)
    
    assert len(user_messages) == 1
    assert user_messages[0]["content"] == "Hello"
    assert len(intermediate) == 1  # just assistant
    assert tool_episode == []


def test_process_and_split_trace_user_tool_empty() -> None:
    """Test 3-way split with empty messages."""
    user_messages, intermediate, tool_episode = process_and_split_trace_user_tool([])
    
    assert user_messages == []
    assert intermediate == []
    assert tool_episode == []


def test_process_and_split_trace_user_tool_corrupted_tools() -> None:
    """Test 3-way split with previously corrupted tool calls (now accepted)."""
    messages = [
        _make_message("system", "System prompt"),
        _make_message("user", "Get data"),
        _make_message(
            "assistant",
            "Fetching...",
            tool_calls=[{"_type": "ToolCall", "id": "tc-1"}],
        ),
        _make_message("tool", "Result", tool_call_id="tc-1"),
    ]
    
    user_messages, intermediate, tool_episode = process_and_split_trace_user_tool(messages)
    
    assert len(user_messages) == 1
    assert user_messages[0]["content"] == "Get data"
    # Intermediate = [] (User at 1, Tool at 2. 2:2)
    assert len(intermediate) == 0
    assert len(tool_episode) == 2


def test_process_and_split_trace_user_tool_no_user_before_tools() -> None:
    """Test 3-way split when no user message before tool episode."""
    messages = [
        _make_message("system", "System prompt"),
        _make_message(
            "assistant",
            "Fetching...",
            tool_calls=[{"id": "tc-1", "type": "function", "function": {"name": "fetch"}}],
        ),
        _make_message("tool", "Result", tool_call_id="tc-1"),
    ]
    
    user_messages, intermediate, tool_episode = process_and_split_trace_user_tool(messages)
    
    assert user_messages == []
    # No user. Intermediate = messages[:tool_start] = [System]
    assert len(intermediate) == 1
    assert intermediate[0]["role"] == "system"
    assert len(tool_episode) == 2
