"""
Tests for the Memory Bank strategy.

Tests follow TDD approach - covering all components of the vector-based
retrieval system for agent interaction history.
"""

import pytest
import numpy as np
from unittest.mock import Mock

from memorch.strategies.memory_bank.models import InteractionRecord
from memorch.strategies.memory_bank.fact_store import FactStore
from memorch.strategies.memory_bank.insight_store import InsightStore
from memorch.strategies.memory_bank.retrieval import (
    retrieve_and_format,
    format_retrieved_memory,
)


# =============================================================================
# InteractionRecord Tests
# =============================================================================


class TestInteractionRecord:
    """Tests for the InteractionRecord data model."""

    def test_create_generates_trace_id(self):
        """
        InteractionRecord.create() should auto-generate a unique trace_id (UUID).
        This ensures each record can be uniquely identified for retrieval.
        """
        record = InteractionRecord.create(
            step_id=1,
            tool_name="search_api",
            raw_input={"query": "Berlin hotels"},
            raw_output={"results": [{"id": "hotel_1"}]},
        )

        assert record.trace_id is not None
        assert len(record.trace_id) == 36  # UUID format: 8-4-4-4-12
        assert record.step_id == 1
        assert record.tool_name == "search_api"

    def test_create_generates_unique_trace_ids(self):
        """
        Multiple calls to create() should generate different trace_ids.
        Critical for deduplication and correct retrieval.
        """
        record1 = InteractionRecord.create(
            step_id=1, tool_name="api_a", raw_input={}, raw_output={}
        )
        record2 = InteractionRecord.create(
            step_id=1, tool_name="api_a", raw_input={}, raw_output={}
        )

        assert record1.trace_id != record2.trace_id

    def test_create_sets_timestamp(self):
        """
        InteractionRecord.create() should set a timestamp automatically.
        Used for debugging and potential time-based ordering.
        """
        record = InteractionRecord.create(
            step_id=1, tool_name="api", raw_input={}, raw_output={}
        )

        assert record.timestamp > 0


# =============================================================================
# FactStore Tests
# =============================================================================


class TestFactStore:
    """Tests for the in-memory key-value Fact Store."""

    def test_store_and_retrieve_single_record(self):
        """
        FactStore should store a record and retrieve it by trace_id.
        This is the basic CRUD operation for ground truth storage.
        """
        store = FactStore()
        record = InteractionRecord.create(
            step_id=1,
            tool_name="search_api",
            raw_input={"query": "Berlin"},
            raw_output={"lat": 52.52, "lon": 13.40},
        )

        store.store(record)
        retrieved = store.get(record.trace_id)

        assert retrieved is not None
        assert retrieved.trace_id == record.trace_id
        assert retrieved.tool_name == "search_api"
        assert retrieved.raw_output == {"lat": 52.52, "lon": 13.40}

    def test_get_returns_none_for_missing_key(self):
        """
        FactStore.get() should return None for non-existent trace_id.
        Ensures graceful handling of missing records.
        """
        store = FactStore()

        result = store.get("non-existent-id")

        assert result is None

    def test_get_many_returns_multiple_records(self):
        """
        FactStore.get_many() should return records for multiple trace_ids.
        Used when retrieving top-K results from vector search.
        """
        store = FactStore()
        records = [
            InteractionRecord.create(
                step_id=i, tool_name=f"api_{i}", raw_input={}, raw_output={"id": i}
            )
            for i in range(3)
        ]
        for r in records:
            store.store(r)

        trace_ids = [records[0].trace_id, records[2].trace_id]
        retrieved = store.get_many(trace_ids)

        assert len(retrieved) == 2
        assert retrieved[0].raw_output == {"id": 0}
        assert retrieved[1].raw_output == {"id": 2}

    def test_get_many_skips_missing_ids(self):
        """
        FactStore.get_many() should skip non-existent trace_ids gracefully.
        Prevents errors when vector store returns stale references.
        """
        store = FactStore()
        record = InteractionRecord.create(
            step_id=1, tool_name="api", raw_input={}, raw_output={}
        )
        store.store(record)

        retrieved = store.get_many([record.trace_id, "missing-id"])

        assert len(retrieved) == 1

    def test_clear_removes_all_records(self):
        """
        FactStore.clear() should remove all stored records.
        Called on session reset between benchmark tasks.
        """
        store = FactStore()
        for i in range(3):
            record = InteractionRecord.create(
                step_id=i, tool_name="api", raw_input={}, raw_output={}
            )
            store.store(record)

        store.clear()

        assert store.size() == 0

    def test_size_returns_record_count(self):
        """
        FactStore.size() should return the number of stored records.
        Useful for debugging and state inspection.
        """
        store = FactStore()
        assert store.size() == 0

        for i in range(5):
            record = InteractionRecord.create(
                step_id=i, tool_name="api", raw_input={}, raw_output={}
            )
            store.store(record)

        assert store.size() == 5


# =============================================================================
# InsightStore Tests
# =============================================================================


class TestInsightStore:
    """Tests for the vector-based Insight Store."""

    @pytest.fixture
    def mock_embedding_model(self):
        """
        Creates a mock FlagModel that returns predictable embeddings.
        Embedding dimension is 4 for simplicity in tests.
        """
        mock_model = Mock()

        # Return different embeddings based on input content
        def mock_encode(texts):
            embeddings = []
            for text in texts:
                if "Berlin" in text:
                    embeddings.append([1.0, 0.0, 0.0, 0.0])
                elif "Paris" in text:
                    embeddings.append([0.0, 1.0, 0.0, 0.0])
                elif "proceed" in text:
                    embeddings.append([0.5, 0.5, 0.0, 0.0])  # Similar to both
                else:
                    embeddings.append([0.0, 0.0, 1.0, 0.0])
            return np.array(embeddings)

        mock_model.encode = mock_encode
        return mock_model

    def test_add_and_search_returns_trace_ids(self, mock_embedding_model):
        """
        InsightStore should store summaries with embeddings and return
        matching trace_ids on search. The core retrieval mechanism.
        """
        store = InsightStore(embedding_model=mock_embedding_model)

        store.add("trace-1", "Found coordinates for Berlin")
        store.add("trace-2", "Found coordinates for Paris")

        results = store.search("how to proceed", top_k=2)

        assert len(results) == 2
        assert "trace-1" in results
        assert "trace-2" in results

    def test_search_returns_most_similar_first(self, mock_embedding_model):
        """
        InsightStore.search() should return results ordered by similarity.
        Top-K should prioritize the most semantically relevant records.
        """
        store = InsightStore(embedding_model=mock_embedding_model)

        store.add("trace-berlin", "Found coordinates for Berlin")
        store.add("trace-paris", "Found coordinates for Paris")
        store.add("trace-other", "Unrelated tool output")

        # "proceed" is more similar to Berlin and Paris in our mock
        results = store.search("how to proceed", top_k=2)

        assert len(results) == 2
        # Both Berlin and Paris should be retrieved, not "other"
        assert "trace-other" not in results

    def test_search_respects_top_k_limit(self, mock_embedding_model):
        """
        InsightStore.search() should return at most top_k results.
        Ensures we don't blow up context window with too many records.
        """
        store = InsightStore(embedding_model=mock_embedding_model)

        for i in range(10):
            store.add(f"trace-{i}", f"Summary {i} for Berlin")

        results = store.search("Berlin", top_k=3)

        assert len(results) == 3

    def test_search_returns_empty_when_no_records(self, mock_embedding_model):
        """
        InsightStore.search() should return empty list when store is empty.
        Handles first-step case gracefully.
        """
        store = InsightStore(embedding_model=mock_embedding_model)

        results = store.search("anything", top_k=5)

        assert results == []

    def test_is_empty(self, mock_embedding_model):
        """
        InsightStore.is_empty() should correctly report store state.
        Used to determine if we should pass through on first step.
        """
        store = InsightStore(embedding_model=mock_embedding_model)

        assert store.is_empty() is True

        store.add("trace-1", "Some summary")

        assert store.is_empty() is False

    def test_clear_removes_all_entries(self, mock_embedding_model):
        """
        InsightStore.clear() should remove all summaries and embeddings.
        Called on session reset between benchmark tasks.
        """
        store = InsightStore(embedding_model=mock_embedding_model)
        store.add("trace-1", "Summary 1")
        store.add("trace-2", "Summary 2")

        store.clear()

        assert store.is_empty() is True

    def test_get_summary_returns_stored_summary(self, mock_embedding_model):
        """
        InsightStore.get_summary() should return the summary for a trace_id.
        Needed for formatting retrieval results.
        """
        store = InsightStore(embedding_model=mock_embedding_model)
        store.add("trace-1", "Tool found Berlin coordinates")

        summary = store.get_summary("trace-1")

        assert summary == "Tool found Berlin coordinates"

    def test_get_summary_returns_none_for_missing(self, mock_embedding_model):
        """
        InsightStore.get_summary() should return None for non-existent trace_id.
        """
        store = InsightStore(embedding_model=mock_embedding_model)

        summary = store.get_summary("non-existent")

        assert summary is None


# =============================================================================
# Retrieval & Formatting Tests
# =============================================================================


class TestRetrieval:
    """Tests for the retrieval and formatting functions."""

    @pytest.fixture
    def populated_stores(self):
        """
        Creates fact and insight stores with test data.
        Uses mock embedding model for predictable behavior.
        """
        mock_model = Mock()
        mock_model.encode = lambda texts: np.array([[1.0, 0.0] for _ in texts])

        fact_store = FactStore()
        insight_store = InsightStore(embedding_model=mock_model)

        # Add test records
        record1 = InteractionRecord(
            trace_id="trace-1",
            step_id=1,
            tool_name="search_api",
            raw_input={"query": "Berlin"},
            raw_output={"lat": 52.52, "lon": 13.40, "id": "loc_123"},
        )
        record2 = InteractionRecord(
            trace_id="trace-2",
            step_id=2,
            tool_name="hotel_api",
            raw_input={"location_id": "loc_123"},
            raw_output={"hotels": [{"name": "Hotel A", "id": "h1"}]},
        )

        fact_store.store(record1)
        fact_store.store(record2)
        insight_store.add(
            "trace-1", "Found coordinates for Berlin: lat 52.52, lon 13.40, id loc_123"
        )
        insight_store.add("trace-2", "Found hotels in Berlin: Hotel A (id: h1)")

        return fact_store, insight_store

    def test_retrieve_and_format_returns_records(self, populated_stores):
        """
        retrieve_and_format() should return formatted records with summaries
        and raw data from both stores.
        """
        fact_store, insight_store = populated_stores

        results = retrieve_and_format(
            query="how to proceed",
            fact_store=fact_store,
            insight_store=insight_store,
            top_k=2,
            max_chars=2000,
        )

        assert len(results) == 2
        assert "summary" in results[0]
        assert "raw_data" in results[0]

    def test_retrieve_and_format_truncates_raw_data(self, populated_stores):
        """
        retrieve_and_format() should truncate raw_data to max_chars.
        Prevents context window overflow from large API responses.
        """
        fact_store, insight_store = populated_stores

        # Add a record with large output
        large_record = InteractionRecord(
            trace_id="trace-large",
            step_id=3,
            tool_name="large_api",
            raw_input={},
            raw_output={"data": "x" * 5000},  # 5000+ chars when serialized
        )
        fact_store.store(large_record)
        insight_store.add("trace-large", "Large data returned")

        results = retrieve_and_format(
            query="how to proceed",
            fact_store=fact_store,
            insight_store=insight_store,
            top_k=3,
            max_chars=100,
        )

        # Find the large record result
        large_result = next(
            (r for r in results if "trace-large" in str(r.get("trace_id", ""))), None
        )
        if large_result:
            assert len(large_result["raw_data"]) <= 100 + 3  # +3 for "..."

    def test_format_retrieved_memory_produces_expected_format(self):
        """
        format_retrieved_memory() should produce the spec-defined format:
        [RETRIEVED RECORD N]
        Summary: ...
        Raw Data: ...
        -------------------
        """
        records = [
            {
                "trace_id": "trace-1",
                "summary": "Found coordinates for Berlin",
                "raw_data": '{"lat": 52.52, "lon": 13.40}',
            },
            {
                "trace_id": "trace-2",
                "summary": "Found hotels",
                "raw_data": '{"hotels": []}',
            },
        ]

        formatted = format_retrieved_memory(records)

        assert "[RETRIEVED RECORD 1]" in formatted
        assert "[RETRIEVED RECORD 2]" in formatted
        assert "Summary: Found coordinates for Berlin" in formatted
        assert "Raw Data:" in formatted
        assert "-------------------" in formatted

    def test_format_retrieved_memory_handles_empty_list(self):
        """
        format_retrieved_memory() should return empty string for empty input.
        Handles first-step case where no history exists.
        """
        formatted = format_retrieved_memory([])

        assert formatted == ""


# =============================================================================
# Observer Tests (with mocked LLM)
# =============================================================================


class TestObserver:
    """Tests for the Observer LLM component."""

    def test_observe_tool_output_calls_llm(self):
        """
        observe_tool_output() should call the LLM with proper prompt
        and return the summary from the response.
        """
        from memorch.strategies.memory_bank.observer import observe_tool_output

        mock_llm = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Tool found 3 hotels in Berlin"
        mock_llm.generate_plain.return_value = mock_response

        summary = observe_tool_output(
            user_query="Find hotels in Berlin",
            tool_name="search_hotel",
            raw_output={"hotels": [{"id": 1}, {"id": 2}, {"id": 3}]},
            llm_client=mock_llm,
            model="gpt-4-1-mini",
        )

        assert summary == "Tool found 3 hotels in Berlin"
        mock_llm.generate_plain.assert_called_once()

    def test_observe_tool_output_truncates_large_output(self):
        """
        observe_tool_output() should truncate raw_output >10k chars.
        Prevents LLM context overflow during observation.
        """
        from memorch.strategies.memory_bank.observer import (
            observe_tool_output,
            MAX_OUTPUT_CHARS,
        )

        mock_llm = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Large output summarized"
        mock_llm.generate_plain.return_value = mock_response

        large_output = {"data": "x" * 15000}

        observe_tool_output(
            user_query="Task",
            tool_name="api",
            raw_output=large_output,
            llm_client=mock_llm,
            model="gpt-4-1-mini",
        )

        # Check that the call was made with truncated content
        call_args = mock_llm.generate_plain.call_args
        messages = call_args.kwargs.get("input_messages") or call_args[1].get(
            "input_messages"
        )
        user_content = messages[-1]["content"]

        # The raw output in the prompt should be truncated
        assert len(user_content) < 15000


# =============================================================================
# Ingestion Pipeline Tests
# =============================================================================


class TestIngestion:
    """Tests for the ingestion pipeline."""

    def test_ingest_tool_outputs_stores_in_both_stores(self):
        """
        ingest_tool_outputs() should store raw data in FactStore
        and summaries in InsightStore for each tool output.
        """
        from memorch.strategies.memory_bank.ingestion import ingest_tool_outputs

        mock_model = Mock()
        mock_model.encode = lambda texts: np.array([[1.0, 0.0] for _ in texts])

        fact_store = FactStore()
        insight_store = InsightStore(embedding_model=mock_model)

        mock_llm = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Summary of tool output"
        mock_llm.generate_plain.return_value = mock_response

        tool_outputs = [
            ("search_api", {"query": "Berlin"}, {"lat": 52.52}),
            ("hotel_api", {"loc": "Berlin"}, {"hotels": []}),
        ]

        trace_ids = ingest_tool_outputs(
            tool_outputs=tool_outputs,
            user_query="Find hotels",
            fact_store=fact_store,
            insight_store=insight_store,
            llm_client=mock_llm,
            observer_model="gpt-4-1-mini",
            step_id=1,
        )

        assert len(trace_ids) == 2
        assert fact_store.size() == 2
        assert not insight_store.is_empty()

    def test_ingest_handles_parallel_calls_individually(self):
        """
        ingest_tool_outputs() should create separate records for each
        parallel tool call, as per spec requirement.
        """
        from memorch.strategies.memory_bank.ingestion import ingest_tool_outputs

        mock_model = Mock()
        mock_model.encode = lambda texts: np.array([[1.0, 0.0] for _ in texts])

        fact_store = FactStore()
        insight_store = InsightStore(embedding_model=mock_model)

        mock_llm = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Summary"
        mock_llm.generate_plain.return_value = mock_response

        # Simulate 3 parallel API calls
        tool_outputs = [
            ("api_1", {}, {"result": 1}),
            ("api_2", {}, {"result": 2}),
            ("api_3", {}, {"result": 3}),
        ]

        ingest_tool_outputs(
            tool_outputs=tool_outputs,
            user_query="Task",
            fact_store=fact_store,
            insight_store=insight_store,
            llm_client=mock_llm,
            observer_model="gpt-4-1-mini",
            step_id=1,
        )

        # Each parallel call gets its own LLM observation call
        assert mock_llm.generate_plain.call_count == 3
        assert fact_store.size() == 3


# =============================================================================
# Strategy Integration Tests
# =============================================================================


class TestMemoryBankStrategy:
    """Integration tests for the full memory bank strategy."""

    @pytest.fixture
    def mock_llm_client(self):
        """Creates a mock LLM client for strategy tests."""
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Tool summary"
        mock_llm.generate_plain.return_value = mock_response
        return mock_llm

    @pytest.fixture
    def mock_settings(self):
        """Creates mock strategy settings."""
        settings = Mock()
        settings.observer_model = "gpt-4-1-mini"
        settings.embedding_model = "BAAI/bge-small-en-v1.5"
        settings.top_k = 3
        settings.max_chars_per_record = 2000
        return settings

    def test_strategy_passthrough_on_first_step(self, mock_llm_client, mock_settings):
        """
        On first step with no history, strategy should pass through
        messages unchanged. Per spec requirement.
        """
        from memorch.strategies.memory_bank.memory_bank_strategy import (
            MemoryBankState,
            apply_memory_bank_strategy,
        )

        messages = [{"role": "user", "content": "Find hotels in Berlin"}]

        # Create state with mock embedding model
        mock_model = Mock()
        mock_model.encode = lambda texts: np.array([[1.0, 0.0] for _ in texts])
        state = MemoryBankState(_embedding_model=mock_model)
        state.insight_store = InsightStore(embedding_model=mock_model)

        processed, token_count = apply_memory_bank_strategy(
            messages=messages,
            llm_client=mock_llm_client,
            settings=mock_settings,
            state=state,
        )

        # Should pass through unchanged
        assert processed == messages

    def test_strategy_constructs_context_with_retrieval(
        self, mock_llm_client, mock_settings
    ):
        """
        After first step, strategy should construct context as:
        user_query + retrieved_memory + last_tool_interaction
        """
        from memorch.strategies.memory_bank.memory_bank_strategy import (
            MemoryBankState,
            apply_memory_bank_strategy,
        )

        # Create state and pre-populate with history
        mock_model = Mock()
        mock_model.encode = lambda texts: np.array([[1.0, 0.0] for _ in texts])
        state = MemoryBankState(_embedding_model=mock_model)
        state.insight_store = InsightStore(embedding_model=mock_model)

        # Manually add some history to the stores
        record = InteractionRecord.create(
            step_id=1,
            tool_name="search_api",
            raw_input={"q": "Berlin"},
            raw_output={"lat": 52.52},
        )
        state.fact_store.store(record)
        state.insight_store.add(record.trace_id, "Found Berlin coordinates")

        # Messages with tool interaction
        messages = [
            {"role": "user", "content": "Find hotels in Berlin"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "1", "function": {"name": "api"}}],
            },
            {"role": "tool", "tool_call_id": "1", "content": '{"result": "ok"}'},
        ]

        processed, token_count = apply_memory_bank_strategy(
            messages=messages,
            llm_client=mock_llm_client,
            settings=mock_settings,
            state=state,
        )

        # Should have: user message + system (retrieved) + tool interaction
        roles = [m["role"] for m in processed]
        assert "user" in roles
        assert "system" in roles  # Retrieved context
        assert "Retrieved Context" in str(processed)

    def test_strategy_resets_state(self, mock_llm_client, mock_settings):
        """
        MemoryBankState.reset() should clear both stores.
        Called between benchmark tasks.
        """
        mock_model = Mock()
        mock_model.encode = lambda texts: np.array([[1.0, 0.0] for _ in texts])

        from memorch.strategies.memory_bank.memory_bank_strategy import MemoryBankState

        state = MemoryBankState(_embedding_model=mock_model)
        state.insight_store = InsightStore(embedding_model=mock_model)

        # Add some data
        record = InteractionRecord.create(
            step_id=1, tool_name="api", raw_input={}, raw_output={}
        )
        state.fact_store.store(record)
        state.insight_store.add(record.trace_id, "Summary")
        state.step_count = 5

        # Reset
        state.reset()

        assert state.fact_store.size() == 0
        assert state.insight_store.is_empty()
        assert state.step_count == 0


# =============================================================================
# Message Parsing Tests
# =============================================================================


class TestMessageParsing:
    """Tests for extracting tool outputs from messages."""

    def test_extract_tool_outputs_from_messages(self):
        """
        extract_tool_outputs() should parse assistant tool_calls and
        subsequent tool responses into (name, input, output) tuples.
        """
        from memorch.strategies.memory_bank.ingestion import extract_tool_outputs

        messages = [
            {"role": "user", "content": "Find hotels"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "search_api",
                            "arguments": '{"query": "Berlin"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": '{"results": [{"id": 1}]}',
            },
        ]

        outputs = extract_tool_outputs(messages)

        assert len(outputs) == 1
        assert outputs[0][0] == "search_api"  # tool_name
        assert outputs[0][1] == {"query": "Berlin"}  # raw_input
        assert outputs[0][2] == {"results": [{"id": 1}]}  # raw_output

    def test_extract_tool_outputs_handles_parallel_calls(self):
        """
        extract_tool_outputs() should handle multiple parallel tool calls
        in a single assistant message.
        """
        from memorch.strategies.memory_bank.ingestion import extract_tool_outputs

        messages = [
            {"role": "user", "content": "Task"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "api_1", "arguments": "{}"}},
                    {"id": "call_2", "function": {"name": "api_2", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"r": 1}'},
            {"role": "tool", "tool_call_id": "call_2", "content": '{"r": 2}'},
        ]

        outputs = extract_tool_outputs(messages)

        assert len(outputs) == 2
        assert outputs[0][0] == "api_1"
        assert outputs[1][0] == "api_2"

    def test_extract_tool_outputs_returns_empty_if_no_tools(self):
        """
        extract_tool_outputs() should return empty list if no tool
        interactions in messages.
        """
        from memorch.strategies.memory_bank.ingestion import extract_tool_outputs

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        outputs = extract_tool_outputs(messages)

        assert outputs == []
