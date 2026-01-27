"""
Memory Bank Strategy - Vector-based retrieval for agent interaction history.

This strategy decouples reasoning (summaries in vector store) from precision
(raw data in key-value store) and serves them jointly during retrieval.
"""

from src.strategies.memory_bank.memory_bank_strategy import (
    MemoryBankState,
    apply_memory_bank_strategy,
)

__all__ = ["MemoryBankState", "apply_memory_bank_strategy"]
