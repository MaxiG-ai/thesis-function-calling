from unittest.mock import Mock

from src.strategies.memory_bank.memory_bank import (
    MemoryBank,
    ToolExecution,
    extract_tool_executions,
    extract_retrieval_query,
)


def _mock_llm(summary_text: str) -> Mock:
    response = Mock()
    message = Mock()
    message.content = summary_text
    response.choices = [Mock(message=message)]
    client = Mock()
    client.generate_plain.return_value = response
    return client


def test_extract_tool_executions_from_messages_parallel() -> None:
    messages = [
        {"role": "user", "content": "Find coordinates"},
        {
            "role": "assistant",
            "content": "Calling tools",
            "tool_calls": [
                {
                    "id": "tc-1",
                    "type": "function",
                    "function": {"name": "search_api", "arguments": '{"query": "Berlin"}'},
                },
                {
                    "id": "tc-2",
                    "type": "function",
                    "function": {"name": "search_api", "arguments": '{"query": "Paris"}'},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "tc-1", "content": '{"id": "loc_123"}'},
        {"role": "tool", "tool_call_id": "tc-2", "content": '{"id": "loc_456"}'},
    ]

    executions = extract_tool_executions(messages)

    assert len(executions) == 2
    assert executions[0].tool_name == "search_api"
    assert executions[0].raw_input == {"query": "Berlin"}
    assert executions[0].raw_output == {"id": "loc_123"}
    assert executions[1].raw_input == {"query": "Paris"}
    assert executions[1].raw_output == {"id": "loc_456"}


def test_memory_bank_ingests_and_retrieves_records() -> None:
    memory_bank = MemoryBank(top_k=1)
    llm_client = _mock_llm("Tool 'search_api' found loc_123")

    tool_execution = ToolExecution(
        tool_name="search_api",
        raw_input={"query": "Berlin"},
        raw_output={"id": "loc_123", "lat": 52.52},
        tool_call_id="tc-1",
    )

    trace_ids = memory_bank.ingest_tool_executions(
        user_query="Find Berlin",
        tool_executions=[tool_execution],
        llm_client=llm_client,
    )

    assert len(trace_ids) == 1
    assert trace_ids[0] in memory_bank.fact_store
    assert memory_bank.insight_store[0].summary == "Tool 'search_api' found loc_123"

    output = memory_bank.retrieve_relevant_history("Berlin")
    assert "[RETRIEVED RECORD 1]" in output
    assert "Summary: Tool 'search_api' found loc_123" in output
    assert '"id": "loc_123"' in output


def test_memory_bank_truncates_raw_data_output() -> None:
    memory_bank = MemoryBank(top_k=1, record_char_limit=50)
    llm_client = _mock_llm("Result stored")

    tool_execution = ToolExecution(
        tool_name="search_api",
        raw_input={"query": "Berlin"},
        raw_output={"blob": "x" * 200},
        tool_call_id="tc-1",
    )

    memory_bank.ingest_tool_executions(
        user_query="Find Berlin",
        tool_executions=[tool_execution],
        llm_client=llm_client,
    )

    output = memory_bank.retrieve_relevant_history("Berlin")
    raw_data_line = output.split("Raw Data: ")[1].split("\n")[0]
    assert len(raw_data_line) <= 50


def test_memory_bank_skips_duplicate_tool_call_ids() -> None:
    memory_bank = MemoryBank(top_k=1)
    llm_client = _mock_llm("Duplicate tool call")

    tool_execution = ToolExecution(
        tool_name="search_api",
        raw_input={"query": "Berlin"},
        raw_output={"id": "loc_123"},
        tool_call_id="tc-1",
    )

    memory_bank.ingest_tool_executions(
        user_query="Find Berlin",
        tool_executions=[tool_execution, tool_execution],
        llm_client=llm_client,
    )

    assert len(memory_bank.fact_store) == 1
    assert len(memory_bank.insight_store) == 1


def test_extract_retrieval_query_from_tool_call() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "Need memory",
            "tool_calls": [
                {
                    "id": "tc-1",
                    "type": "function",
                    "function": {"name": "retrieve_relevant_history", "arguments": '{"query": "Berlin"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "tc-1", "content": "result"},
    ]

    assert extract_retrieval_query(messages) == "Berlin"
