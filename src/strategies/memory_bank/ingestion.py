"""
Ingestion Pipeline - Processes tool outputs into the dual-store system.

Captures tool outputs, stores raw data in FactStore, generates summaries
via Observer LLM, and stores embeddings in InsightStore.
"""

import json
from typing import Any, Dict, List, Tuple

from memorch.strategies.memory_bank.models import InteractionRecord
from memorch.strategies.memory_bank.fact_store import FactStore
from memorch.strategies.memory_bank.insight_store import InsightStore
from memorch.strategies.memory_bank.observer import observe_tool_output
from memorch.utils.logger import get_logger

logger = get_logger("Ingestion")


def extract_tool_outputs(messages: List[Dict]) -> List[Tuple[str, Dict, Dict]]:
    """
    Extract tool call information from the latest tool interaction in messages.

    Parses assistant messages with tool_calls and subsequent tool responses
    to extract (tool_name, raw_input, raw_output) tuples.

    Args:
        messages: Conversation messages list

    Returns:
        List of (tool_name, raw_input, raw_output) tuples from the last
        tool interaction. Empty list if no tool calls found.
    """
    if not messages:
        return []

    # Find the last assistant message with tool_calls
    assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            assistant_idx = i
            break

    if assistant_idx is None:
        return []

    assistant_msg = messages[assistant_idx]
    tool_calls = assistant_msg.get("tool_calls", [])

    # Build mapping of tool_call_id -> (tool_name, raw_input)
    call_map: Dict[str, Tuple[str, Dict]] = {}
    for tc in tool_calls:
        # Handle both dict and object formats
        if isinstance(tc, dict):
            call_id = tc.get("id", "")
            func = tc.get("function", {})
            tool_name = func.get("name", "unknown")
            args_str = func.get("arguments", "{}")
        else:
            call_id = getattr(tc, "id", "")
            func = getattr(tc, "function", None)
            tool_name = getattr(func, "name", "unknown") if func else "unknown"
            args_str = getattr(func, "arguments", "{}") if func else "{}"

        # Parse arguments JSON
        try:
            raw_input = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            raw_input = {"_raw": args_str}

        call_map[call_id] = (tool_name, raw_input)

    # Collect tool responses that follow the assistant message
    outputs = []
    for msg in messages[assistant_idx + 1 :]:
        if msg.get("role") != "tool":
            break

        tool_call_id = msg.get("tool_call_id", "")
        content_str = msg.get("content", "{}")

        # Parse response JSON
        try:
            raw_output = json.loads(content_str) if content_str else {}
        except json.JSONDecodeError:
            raw_output = {"_raw": content_str}

        if tool_call_id in call_map:
            tool_name, raw_input = call_map[tool_call_id]
            outputs.append((tool_name, raw_input, raw_output))

    return outputs


def ingest_tool_outputs(
    tool_outputs: List[Tuple[str, Dict, Dict]],
    user_query: str,
    fact_store: FactStore,
    insight_store: InsightStore,
    llm_client: Any,
    observer_model: str,
    step_id: int,
) -> List[str]:
    """
    Ingest tool outputs into the dual-store system.

    For each tool output:
    1. Create InteractionRecord and store in FactStore
    2. Call Observer LLM to generate summary
    3. Embed summary and store in InsightStore

    Args:
        tool_outputs: List of (tool_name, raw_input, raw_output) tuples
        user_query: User's original task (for Observer context)
        fact_store: FactStore instance
        insight_store: InsightStore instance
        llm_client: LLM client for Observer
        observer_model: Model to use for Observer LLM
        step_id: Current step number in the task

    Returns:
        List of generated trace_ids
    """
    trace_ids = []

    for tool_name, raw_input, raw_output in tool_outputs:
        # Create and store interaction record
        record = InteractionRecord.create(
            step_id=step_id,
            tool_name=tool_name,
            raw_input=raw_input,
            raw_output=raw_output,
        )
        fact_store.store(record)

        # Generate summary via Observer LLM
        summary = observe_tool_output(
            user_query=user_query,
            tool_name=tool_name,
            raw_output=raw_output,
            llm_client=llm_client,
            model=observer_model,
        )

        # Store summary with embedding
        insight_store.add(record.trace_id, summary)

        trace_ids.append(record.trace_id)
        logger.debug(
            f"Ingested tool '{tool_name}' -> trace_id={record.trace_id[:8]}..."
        )

    return trace_ids
