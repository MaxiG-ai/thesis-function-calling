"""
Tests for haystack injection in FunctionCallSAPGPT and SAPGPTRunner.

These tests verify that pre-computed haystack messages (distractor context)
are correctly injected into the model's message list before the first LLM
call, without affecting the logical evaluation message list.

Test categories:
1. FunctionCallSAPGPT injection - haystack messages prepended on first call
2. SAPGPTRunner passthrough     - runner reads haystack_messages from data
                                  and passes them to the model
3. Evaluation integrity         - haystack never leaks into logical messages
"""

import copy
import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from benchmarks.complex_func_bench.models.sap_gpt import FunctionCallSAPGPT
from benchmarks.complex_func_bench.runner.sap_gpt_runner import SAPGPTRunner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_haystack_messages(n=2):
    """Create n synthetic haystack message pairs (assistant + tool response).

    Returns a flat list of OpenAI-format messages suitable for prepending
    to a model's message list. Each pair is one assistant message with a
    tool_call followed by one tool response.
    """
    msgs = []
    for i in range(n):
        tc_id = f"haystack_{i:04d}_0000"
        msgs.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": f"Distractor_Func_{i}",
                            "arguments": json.dumps({"key": f"value_{i}"}),
                        },
                    }
                ],
            }
        )
        msgs.append(
            {
                "role": "tool",
                "tool_call_id": tc_id,
                "name": f"Distractor_Func_{i}",
                "content": json.dumps({"status": True, "data": f"distractor_{i}"}),
            }
        )
    return msgs


def _make_mock_orchestrator(compressed_view=None):
    """Create a mock LLMOrchestrator with required attributes.

    Args:
        compressed_view: Optional list of messages to return as last_compressed_view,
            simulating a strategy that has compressed the input. When None the
            orchestrator behaves like no_strategy (write-back is a no-op guard).
    """
    orchestrator = MagicMock()
    orchestrator.active_model_key = "test-model"
    orchestrator.last_compressed_view = compressed_view
    return orchestrator


def _make_runner_test_data(haystack_messages=None):
    """Create a minimal but valid benchmark case for runner tests.

    The case must have at least one function_call + observation pair so
    init_golden() in base_runner.py can index into fc_chain/obs_chain.
    The conversation ends with a user query that the LLM will respond to.

    Args:
        haystack_messages: Optional list of haystack messages to include.

    Returns:
        A dict matching ComplexFuncBench.jsonl schema, optionally with
        haystack_messages.
    """
    data = {
        "id": "Flights-1",
        "conversations": [
            {"role": "user", "content": "Find flights from NYC to LAX"},
            {
                "role": "assistant",
                "function_call": [
                    {
                        "name": "Search_Flight_Location",
                        "arguments": {"query": "New York"},
                    }
                ],
            },
            {
                "role": "observation",
                "content": [
                    {"status": True, "data": [{"id": "NYC", "name": "New York"}]}
                ],
            },
            {"role": "assistant", "content": "I found flights for you."},
        ],
        "functions": [
            {
                "name": "Search_Flight_Location",
                "description": "Search for flight locations",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ],
    }
    if haystack_messages is not None:
        data["haystack_messages"] = haystack_messages
    return data


# ---------------------------------------------------------------------------
# 1. FunctionCallSAPGPT injection
# ---------------------------------------------------------------------------


class TestFunctionCallSAPGPTInjection:
    """Tests for haystack message injection into FunctionCallSAPGPT."""

    def test_haystack_prepended_on_first_call(self):
        """
        When haystack_messages is set on the model, the first call to
        generate_response must prepend them to self.messages before the
        user's conversation messages. This ensures the LLM sees the
        distractor context first, then the actual task.
        """
        orchestrator = _make_mock_orchestrator()
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.content = "test response"
        orchestrator.generate_with_memory_applied.return_value = mock_response

        model = FunctionCallSAPGPT("test-model", orchestrator=orchestrator)

        haystack = _make_haystack_messages(2)  # 4 messages total
        model.haystack_messages = haystack

        # First call - messages without function_call triggers self.messages init
        user_messages = [{"role": "user", "content": "Hello"}]
        model.generate_response(user_messages, tools=[])

        # self.messages should start with haystack, then user messages
        assert len(model.messages) == 5  # 4 haystack + 1 user
        assert model.messages[0]["role"] == "assistant"
        assert model.messages[0]["content"] is None
        assert "tool_calls" in model.messages[0]
        assert model.messages[1]["role"] == "tool"
        assert model.messages[2]["role"] == "assistant"
        assert model.messages[3]["role"] == "tool"
        assert model.messages[4]["role"] == "user"
        assert model.messages[4]["content"] == "Hello"

    def test_haystack_cleared_after_injection(self):
        """
        After the haystack messages are prepended on the first call,
        haystack_messages must be set to None so subsequent calls do not
        re-inject them. This prevents duplication on multi-turn conversations.
        """
        orchestrator = _make_mock_orchestrator()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.content = "test"
        orchestrator.generate_with_memory_applied.return_value = mock_response

        model = FunctionCallSAPGPT("test-model", orchestrator=orchestrator)
        model.haystack_messages = _make_haystack_messages(1)

        user_messages = [{"role": "user", "content": "Hello"}]
        model.generate_response(user_messages, tools=[])

        assert model.haystack_messages is None

    def test_no_haystack_when_not_set(self):
        """
        When haystack_messages is None (the default), generate_response must
        behave identically to the original code — self.messages equals the
        deep copy of the input messages with no extra prefix.
        """
        orchestrator = _make_mock_orchestrator()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.content = "test"
        orchestrator.generate_with_memory_applied.return_value = mock_response

        model = FunctionCallSAPGPT("test-model", orchestrator=orchestrator)
        # haystack_messages defaults to None — do not set it

        user_messages = [{"role": "user", "content": "Hello"}]
        model.generate_response(user_messages, tools=[])

        assert len(model.messages) == 1
        assert model.messages[0]["role"] == "user"

    def test_haystack_not_injected_on_subsequent_calls(self):
        """
        On subsequent calls (where messages contain function_call), the
        haystack should already be part of self.messages and must not be
        re-injected. The 'function_call' check skips the init branch,
        so haystack_messages (now None) is never touched again.
        """
        orchestrator = _make_mock_orchestrator()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.content = "test"
        orchestrator.generate_with_memory_applied.return_value = mock_response

        model = FunctionCallSAPGPT("test-model", orchestrator=orchestrator)
        model.haystack_messages = _make_haystack_messages(1)

        # First call - injects haystack
        user_messages = [{"role": "user", "content": "Hello"}]
        model.generate_response(user_messages, tools=[])
        msg_count_after_first = len(model.messages)

        # Simulate runner appending tool interaction to self.messages
        model.messages.append({"role": "assistant", "content": None, "tool_calls": []})
        model.messages.append(
            {"role": "tool", "tool_call_id": "tc_1", "content": "obs"}
        )

        # Second call - messages now contain tool data (function_call branch)
        messages_with_fc = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "function_call": [{"name": "f", "arguments": {}}]},
            {"role": "observation", "content": [{}]},
        ]
        model.generate_response(messages_with_fc, tools=[])

        # self.messages should NOT have been reset or re-injected
        assert len(model.messages) == msg_count_after_first + 2  # 2 appended above


# ---------------------------------------------------------------------------
# 2. SAPGPTRunner passthrough
# ---------------------------------------------------------------------------


class TestSAPGPTRunnerHaystackPassthrough:
    """Tests for SAPGPTRunner passing haystack data to the model."""

    def test_runner_sets_haystack_on_model(self):
        """
        When data contains 'haystack_messages', the runner must assign
        them to self.model.haystack_messages before the first LLM call.
        This ensures the model has the haystack ready for injection.
        """
        orchestrator = _make_mock_orchestrator()
        mock_args = MagicMock()
        mock_compare = MagicMock()
        mock_compare.free_function_list = []

        runner = SAPGPTRunner(
            mock_args,
            orchestrator=orchestrator,
            compare_class=mock_compare,
        )

        haystack = _make_haystack_messages(2)

        # Mock the model's generate_response to capture the haystack state
        captured_haystack = []

        original_generate = runner.model.generate_response

        def mock_generate(messages, tools=None, **kwargs):
            captured_haystack.append(copy.deepcopy(runner.model.haystack_messages))
            # Return a final text response to end the loop
            mock_msg = MagicMock()
            mock_msg.tool_calls = None
            mock_msg.content = "Done."
            return mock_msg

        runner.model.generate_response = mock_generate

        data = _make_runner_test_data(haystack_messages=haystack)

        runner.run(data)

        # The model should have had haystack_messages set before generate_response
        assert captured_haystack[0] == haystack

    def test_runner_no_haystack_in_data(self):
        """
        When data does NOT contain 'haystack_messages', the model's
        haystack_messages must be None, preserving backward compatibility
        with the original ComplexFuncBench.jsonl format.
        """
        orchestrator = _make_mock_orchestrator()
        mock_args = MagicMock()
        mock_compare = MagicMock()
        mock_compare.free_function_list = []

        runner = SAPGPTRunner(
            mock_args,
            orchestrator=orchestrator,
            compare_class=mock_compare,
        )

        captured_haystack = []

        def mock_generate(messages, tools=None, **kwargs):
            captured_haystack.append(runner.model.haystack_messages)
            mock_msg = MagicMock()
            mock_msg.tool_calls = None
            mock_msg.content = "Done."
            return mock_msg

        runner.model.generate_response = mock_generate

        data = _make_runner_test_data(haystack_messages=None)

        runner.run(data)

        assert captured_haystack[0] is None


# ---------------------------------------------------------------------------
# 3. Evaluation integrity
# ---------------------------------------------------------------------------


class TestEvaluationIntegrity:
    """Tests ensuring haystack never leaks into the evaluation messages."""

    def test_logical_messages_exclude_haystack(self):
        """
        The logical 'messages' list used for step evaluation must never
        contain haystack messages. Haystack is only in self.model.messages
        (the LLM-facing list). The runner's local 'messages' variable
        starts with the user query and contains only actual conversation
        turns, not distractor context.
        """
        orchestrator = _make_mock_orchestrator()
        mock_args = MagicMock()
        mock_compare = MagicMock()
        mock_compare.free_function_list = []

        runner = SAPGPTRunner(
            mock_args,
            orchestrator=orchestrator,
            compare_class=mock_compare,
        )

        haystack = _make_haystack_messages(3)  # 6 messages

        # Track the messages list passed to return_result
        captured_messages = []
        original_return_result = runner.return_result

        def mock_return_result(messages, error_info=None):
            captured_messages.append(copy.deepcopy(messages))
            return messages, "Success.", 1, 0

        runner.return_result = mock_return_result

        def mock_generate(messages, tools=None, **kwargs):
            mock_msg = MagicMock()
            mock_msg.tool_calls = None
            mock_msg.content = "Final answer."
            return mock_msg

        runner.model.generate_response = mock_generate

        data = _make_runner_test_data(haystack_messages=haystack)

        runner.run(data)

        # The logical messages should only contain user + assistant, no haystack
        assert len(captured_messages) == 1
        logical = captured_messages[0]
        # First message is the user query
        assert logical[0]["role"] == "user"
        assert logical[0]["content"] == "Find flights from NYC to LAX"
        # No haystack function names should appear in the logical messages
        logical_str = json.dumps(logical)
        assert "Distractor_Func" not in logical_str


# ---------------------------------------------------------------------------
# 4. Compressed-view write-back
# ---------------------------------------------------------------------------


class TestCompressedViewWriteBack:
    """Tests verifying that generate_response replaces self.messages with the
    orchestrator's compressed_view after each call.

    This is the core mechanism that prevents redundant recompression: once a
    strategy has processed the raw buffer (e.g. summarised haystack + history),
    the next turn starts from the already-compressed state rather than the
    original growing buffer.
    """

    def test_write_back_replaces_messages_with_compressed_view(self):
        """
        After generate_response() succeeds, self.messages must equal a deep
        copy of orchestrator.last_compressed_view.

        The orchestrator simulates a compressing strategy by returning a
        compressed_view that is shorter than the original input. After the
        call, model.messages should reflect the compressed state, not the
        original injected messages.
        """
        # Simulate what a strategy like progressive_summarization returns:
        # the full haystack + user input is compressed to a single summary msg.
        compressed = [{"role": "system", "content": "Summary of prior context."}]
        orchestrator = _make_mock_orchestrator(compressed_view=compressed)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.content = "answer"
        orchestrator.generate_with_memory_applied.return_value = mock_response

        model = FunctionCallSAPGPT("test-model", orchestrator=orchestrator)
        model.haystack_messages = _make_haystack_messages(3)  # 6 distractor msgs

        user_messages = [{"role": "user", "content": "Do something"}]
        model.generate_response(user_messages, tools=[])

        # model.messages must now be a copy of the compressed view, not the
        # original 7-message buffer (6 haystack + 1 user).
        assert model.messages == compressed
        assert model.messages is not compressed, (
            "must be a deep copy, not the same object"
        )

    def test_write_back_is_deep_copy(self):
        """
        The write-back must store a deep copy of last_compressed_view, not a
        reference. Mutating model.messages after the call (e.g. appending new
        tool turns) must not affect orchestrator.last_compressed_view.
        """
        compressed = [{"role": "system", "content": "Summary."}]
        orchestrator = _make_mock_orchestrator(compressed_view=compressed)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.content = "ok"
        orchestrator.generate_with_memory_applied.return_value = mock_response

        model = FunctionCallSAPGPT("test-model", orchestrator=orchestrator)
        model.generate_response([{"role": "user", "content": "hi"}], tools=[])

        # Append a new tool turn (as the runner would do between turns)
        model.messages.append({"role": "assistant", "content": None, "tool_calls": []})

        # The original compressed list must be unaffected
        assert len(orchestrator.last_compressed_view) == 1

    def test_no_write_back_when_compressed_view_is_none(self):
        """
        When orchestrator.last_compressed_view is None (e.g. for no_strategy
        or a mock that does not set it), self.messages must be left untouched
        after the call. This guards against overwriting messages with None.
        """
        orchestrator = _make_mock_orchestrator(compressed_view=None)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.content = "ok"
        orchestrator.generate_with_memory_applied.return_value = mock_response

        model = FunctionCallSAPGPT("test-model", orchestrator=orchestrator)
        user_messages = [{"role": "user", "content": "hello"}]
        model.generate_response(user_messages, tools=[])

        # With no compressed view, messages stays as the deep-copied input
        assert model.messages == user_messages

    def test_second_turn_builds_on_compressed_state(self):
        """
        After a first call with write-back, the second call must send the
        compressed buffer (plus any new turns appended by the runner) to the
        orchestrator — not the original raw buffer.

        This is the core guarantee of the write-back mechanism: subsequent
        turns never see the original haystack again.
        """
        compressed_after_turn1 = [
            {"role": "system", "content": "Compressed history."},
            {"role": "user", "content": "Do something"},
        ]
        orchestrator = _make_mock_orchestrator(compressed_view=compressed_after_turn1)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].message.content = "done"
        orchestrator.generate_with_memory_applied.return_value = mock_response

        model = FunctionCallSAPGPT("test-model", orchestrator=orchestrator)
        model.haystack_messages = _make_haystack_messages(5)  # 10 distractor msgs

        # Turn 1
        model.generate_response([{"role": "user", "content": "Do something"}], tools=[])
        # model.messages is now compressed_after_turn1 (2 messages)

        # Runner appends new tool-call turn (as sap_gpt_runner.py lines 114-173 do)
        new_tool_call = {"role": "assistant", "content": None, "tool_calls": []}
        new_tool_result = {"role": "tool", "tool_call_id": "tc_1", "content": "result"}
        model.messages.append(new_tool_call)
        model.messages.append(new_tool_result)

        # Turn 2 — capture what the orchestrator actually receives
        captured_input = []
        original_call = orchestrator.generate_with_memory_applied

        def capturing_call(input_messages, **kwargs):
            captured_input.append(copy.deepcopy(input_messages))
            return original_call(input_messages, **kwargs)

        orchestrator.generate_with_memory_applied.side_effect = capturing_call

        model.generate_response([{"role": "user", "content": "Do something"}], tools=[])

        # The orchestrator must have received compressed(2) + 2 new turns = 4 msgs,
        # NOT the original 10 haystack + 1 user + 2 new turns = 13 msgs.
        # (captured_input may contain multiple entries due to the @retry decorator,
        # but every entry must reflect the compressed state.)
        assert len(captured_input) >= 1
        first_call_input = captured_input[0]
        assert len(first_call_input) == 4
        # Haystack content must not appear in what was sent
        sent_str = json.dumps(first_call_input)
        assert "Distractor_Func" not in sent_str
