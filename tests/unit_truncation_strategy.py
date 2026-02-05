"""
Tests for the truncation memory strategy.

The truncation strategy removes conversation history between the user query
and the last tool interaction, keeping only the essential context for the LLM.
"""

import pytest

from memorch.strategies.truncation.truncation import truncate_messages


def _make_message(role: str, content: str, **extras) -> dict:
    """Helper to create message dicts for testing."""
    message = {"role": role, "content": content}
    message.update(extras)
    return message


class TestTruncateMessages:
    """Tests for the truncate_messages function."""

    def test_truncate_removes_intermediate_history(self) -> None:
        """
        Test that intermediate conversation history is removed.

        The truncation strategy should keep the first user message and the
        last tool interaction, discarding everything in between to reduce
        context size while preserving essential information.
        """
        messages = [
            _make_message("system", "System prompt"),
            _make_message("user", "First question - this is the task"),
            _make_message("assistant", "First answer - should be removed"),
            _make_message("user", "Follow-up question - should be removed"),
            _make_message("assistant", "Second answer - should be removed"),
            _make_message(
                "assistant",
                "Calling tool",
                tool_calls=[
                    {"id": "tc-1", "type": "function", "function": {"name": "get_data"}}
                ],
            ),
            _make_message("tool", '{"result": "data"}', tool_call_id="tc-1"),
        ]

        result = truncate_messages(messages)

        # Should have: first user message + tool interaction (assistant + tool)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "First question - this is the task"
        assert result[1]["role"] == "assistant"
        assert "tool_calls" in result[1]
        assert result[2]["role"] == "tool"

    def test_truncate_no_tool_interaction(self) -> None:
        """
        Test truncation when there's no tool interaction at the end.

        When there's no tool interaction, the strategy keeps only the user
        query since there's no tool context to preserve. The assistant
        response is considered intermediate history and is discarded.
        """
        messages = [
            _make_message("user", "Simple question"),
            _make_message("assistant", "Simple answer"),
        ]

        result = truncate_messages(messages)

        # Only user query is kept (no tool interaction to preserve)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Simple question"

    def test_truncate_empty_messages(self) -> None:
        """
        Test truncation with empty message list.

        Should handle edge case gracefully without raising exceptions.
        """
        result = truncate_messages([])

        assert result == []

    def test_truncate_only_user_message(self) -> None:
        """
        Test truncation with only a user message.

        The simplest valid conversation should be preserved intact.
        """
        messages = [_make_message("user", "Hello")]

        result = truncate_messages(messages)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello"

    def test_truncate_preserves_tool_call_details(self) -> None:
        """
        Test that tool call details are preserved in truncated output.

        The tool_calls structure and tool response content should be
        kept intact for the LLM to process correctly.
        """
        tool_calls = [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {"name": "search", "arguments": '{"query": "test"}'},
            }
        ]
        messages = [
            _make_message("user", "Search for test"),
            _make_message("assistant", "Searching...", tool_calls=tool_calls),
            _make_message(
                "tool", '{"results": ["item1", "item2"]}', tool_call_id="call_abc123"
            ),
        ]

        result = truncate_messages(messages)

        # Find the assistant message with tool_calls
        assistant_msg = next((m for m in result if m.get("tool_calls")), None)
        assert assistant_msg is not None
        assert assistant_msg["tool_calls"] == tool_calls

        # Find the tool response
        tool_msg = next((m for m in result if m["role"] == "tool"), None)
        assert tool_msg is not None
        assert tool_msg["tool_call_id"] == "call_abc123"

    def test_truncate_multiple_parallel_tool_calls(self) -> None:
        """
        Test truncation preserves multiple parallel tool calls.

        When an assistant makes multiple tool calls in one turn, all
        should be preserved in the truncated output.
        """
        tool_calls = [
            {"id": "tc-1", "type": "function", "function": {"name": "get_weather"}},
            {"id": "tc-2", "type": "function", "function": {"name": "get_time"}},
        ]
        messages = [
            _make_message("user", "What's the weather and time?"),
            _make_message("assistant", "Getting both...", tool_calls=tool_calls),
            _make_message("tool", '{"temp": 72}', tool_call_id="tc-1"),
            _make_message("tool", '{"time": "3pm"}', tool_call_id="tc-2"),
        ]

        result = truncate_messages(messages)

        # Should have user + assistant + 2 tool responses
        assert len(result) == 4
        tool_responses = [m for m in result if m["role"] == "tool"]
        assert len(tool_responses) == 2

    def test_truncate_with_system_message(self) -> None:
        """
        Test that system messages are handled correctly.

        System messages typically come before user messages and may or
        may not be included depending on the split_trace implementation.
        """
        messages = [
            _make_message("system", "You are a helpful assistant"),
            _make_message("user", "Hello"),
            _make_message("assistant", "Hi there!"),
        ]

        result = truncate_messages(messages)

        # The first user message should be preserved
        user_msgs = [m for m in result if m["role"] == "user"]
        assert len(user_msgs) >= 1

    def test_truncate_long_conversation(self) -> None:
        """
        Test truncation on a longer conversation with multiple turns.

        This simulates a real-world scenario where context has grown
        large and needs to be compressed.
        """
        messages = [
            _make_message("system", "System prompt"),
            _make_message("user", "Initial task: analyze this data"),
            # Many intermediate turns
            _make_message("assistant", "I'll start by..."),
            _make_message("user", "Good, continue"),
            _make_message("assistant", "Next step..."),
            _make_message("user", "What about X?"),
            _make_message("assistant", "For X, I'll..."),
            _make_message("user", "Ok proceed"),
            _make_message("assistant", "Analyzing..."),
            # Final tool interaction
            _make_message(
                "assistant",
                "Running analysis",
                tool_calls=[
                    {
                        "id": "tc-final",
                        "type": "function",
                        "function": {"name": "analyze"},
                    }
                ],
            ),
            _make_message("tool", '{"analysis": "complete"}', tool_call_id="tc-final"),
        ]

        result = truncate_messages(messages)

        # Should be much shorter than original
        assert len(result) < len(messages)
        # Should preserve the first user message (task)
        assert result[0]["role"] == "user"
        assert "Initial task" in result[0]["content"]
        # Should preserve the final tool interaction
        assert result[-1]["role"] == "tool"
