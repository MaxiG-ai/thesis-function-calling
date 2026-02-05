"""
Retrieval - Fetches and formats relevant interaction history.

Queries the InsightStore for semantically similar summaries, retrieves
raw data from FactStore, and formats for context window injection.
"""

import json
from typing import Any, Dict, List

from memorch.strategies.memory_bank.fact_store import FactStore
from memorch.strategies.memory_bank.insight_store import InsightStore
from memorch.utils.logger import get_logger

logger = get_logger("Retrieval")


def retrieve_and_format(
    query: str,
    fact_store: FactStore,
    insight_store: InsightStore,
    top_k: int = 3,
    max_chars: int = 2000,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-K relevant interactions and format for context injection.

    Performs semantic search over summaries in InsightStore, then fetches
    corresponding raw data from FactStore. Results are formatted with
    truncation to prevent context window overflow.

    Args:
        query: Search query (e.g., "how to proceed")
        fact_store: FactStore with raw interaction data
        insight_store: InsightStore with embedded summaries
        top_k: Maximum number of results
        max_chars: Maximum characters per raw_data field

    Returns:
        List of dicts with {trace_id, summary, raw_data} for each result
    """
    # Search for similar summaries
    trace_ids = insight_store.search(query, top_k=top_k)

    if not trace_ids:
        return []

    # Fetch raw records from FactStore
    records = fact_store.get_many(trace_ids)

    results = []
    for record in records:
        # Get summary from insight store
        summary = (
            insight_store.get_summary(record.trace_id)
            or f"Tool '{record.tool_name}' was called."
        )

        # Serialize and truncate raw output
        raw_data_str = json.dumps(record.raw_output, ensure_ascii=False)
        if len(raw_data_str) > max_chars:
            raw_data_str = raw_data_str[:max_chars] + "..."

        results.append(
            {
                "trace_id": record.trace_id,
                "summary": summary,
                "raw_data": raw_data_str,
            }
        )

    return results


def format_retrieved_memory(records: List[Dict[str, Any]]) -> str:
    """
    Format retrieved records into the spec-defined format for context injection.

    Format:
        [RETRIEVED RECORD 1]
        Summary: <observer_summary>
        Raw Data: <truncated_json>
        -------------------
        [RETRIEVED RECORD 2]
        ...

    Args:
        records: List of {trace_id, summary, raw_data} dicts

    Returns:
        Formatted string for context window, or empty string if no records
    """
    if not records:
        return ""

    sections = []
    for i, record in enumerate(records, 1):
        section = f"""[RETRIEVED RECORD {i}]
Summary: {record["summary"]}
Raw Data: {record["raw_data"]}
-------------------"""
        sections.append(section)

    return "\n".join(sections)
