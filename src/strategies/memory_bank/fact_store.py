"""
Fact Store - In-memory key-value store for raw interaction data.

Stores ground truth tool inputs/outputs keyed by trace_id (UUID).
This is the "precision" component of the dual-store architecture.
"""

from typing import Dict, List, Optional

from src.strategies.memory_bank.models import InteractionRecord


class FactStore:
    """
    In-memory key-value store for InteractionRecords.

    Provides O(1) lookup by trace_id for retrieving raw tool data
    after vector search returns relevant trace_ids.
    """

    def __init__(self):
        self._store: Dict[str, InteractionRecord] = {}

    def store(self, record: InteractionRecord) -> None:
        """Store an interaction record, keyed by its trace_id."""
        self._store[record.trace_id] = record

    def get(self, trace_id: str) -> Optional[InteractionRecord]:
        """Retrieve a record by trace_id, or None if not found."""
        return self._store.get(trace_id)

    def get_many(self, trace_ids: List[str]) -> List[InteractionRecord]:
        """
        Retrieve multiple records by trace_ids.

        Preserves order of input trace_ids and skips missing entries.
        Used when fetching top-K results from vector search.
        """
        results = []
        for tid in trace_ids:
            record = self._store.get(tid)
            if record is not None:
                results.append(record)
        return results

    def clear(self) -> None:
        """Remove all records. Called on session reset."""
        self._store.clear()

    def size(self) -> int:
        """Return number of stored records."""
        return len(self._store)
