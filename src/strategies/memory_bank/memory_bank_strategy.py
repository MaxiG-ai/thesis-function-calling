"""
Memory Bank Strategy - Main orchestration for vector-based retrieval.

Implements automatic retrieval-augmented context compression:
1. Ingest new tool outputs after each LLM response
2. Retrieve top-K relevant past interactions
3. Construct context: user_query + retrieved_memory + last_tool_interaction
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from FlagEmbedding import FlagModel

from src.strategies.memory_bank.fact_store import FactStore
from src.strategies.memory_bank.insight_store import InsightStore
from src.strategies.memory_bank.ingestion import (
    extract_tool_outputs,
    ingest_tool_outputs,
)
from src.strategies.memory_bank.retrieval import (
    retrieve_and_format,
    format_retrieved_memory,
)
from src.utils.logger import get_logger
from src.utils.token_count import get_token_count
from src.utils.split_trace import get_user_message, get_last_tool_interaction

logger = get_logger("MemoryBankStrategy")

# Fixed retrieval query per spec
RETRIEVAL_QUERY = "how to proceed with the task"


@dataclass
class MemoryBankState:
    """
    Mutable state for Memory Bank strategy within a single task.

    Contains both stores and the embedding model. State is reset
    between tasks via reset() method.

    Args:
        embedding_model: Optional pre-initialized FlagModel. If None,
                        must be set before first use via initialize_model().
    """

    fact_store: FactStore = field(default_factory=FactStore)
    insight_store: Optional[InsightStore] = None
    step_count: int = 0
    _embedding_model: Optional[Any] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize insight store if embedding model provided."""
        if self._embedding_model is not None:
            self.insight_store = InsightStore(embedding_model=self._embedding_model)

    @classmethod
    def create_with_model(
        cls, embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    ) -> "MemoryBankState":
        """
        Factory method that initializes with a real FlagModel.

        Args:
            embedding_model_name: HuggingFace model identifier

        Returns:
            Initialized MemoryBankState with loaded embedding model
        """
        logger.info(f"Loading embedding model: {embedding_model_name}")
        model = FlagModel(
            embedding_model_name,
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
            use_fp16=True,
        )
        state = cls(_embedding_model=model)
        state.insight_store = InsightStore(embedding_model=model)
        return state

    def initialize_model(
        self, embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    ) -> None:
        """
        Initialize embedding model if not already done.
        Called lazily on first strategy application.
        """
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {embedding_model_name}")
            self._embedding_model = FlagModel(
                embedding_model_name,
                query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
                use_fp16=True,
            )
            self.insight_store = InsightStore(embedding_model=self._embedding_model)

    def reset(self) -> None:
        """Reset state between tasks. Clears stores but keeps model loaded."""
        self.fact_store.clear()
        if self.insight_store:
            self.insight_store.clear()
        self.step_count = 0
        logger.debug("MemoryBankState reset")


def _get_user_query_text(messages: List[Dict]) -> str:
    """Extract user query text from messages."""
    user_msgs, _ = get_user_message(messages)
    if user_msgs:
        return user_msgs[0].get("content", "")
    return ""


def apply_memory_bank_strategy(
    messages: List[Dict],
    llm_client: Any,
    settings: Any,
    state: MemoryBankState,
) -> Tuple[List[Dict], int]:
    """
    Apply memory bank strategy with automatic retrieval.

    Flow:
    1. Ensure embedding model is initialized
    2. Ingest new tool outputs from messages (if any)
    3. If no history yet → pass through unchanged
    4. Otherwise → construct: user_query + retrieved_memory + last_tool_interaction

    Args:
        messages: Current conversation messages
        llm_client: LLM client for Observer calls
        settings: Memory strategy settings (MemoryDef)
        state: Current MemoryBankState

    Returns:
        (processed_messages, token_count)
    """
    state.step_count += 1

    # Ensure model is initialized (eager init per user preference)
    embedding_model_name = getattr(
        settings, "embedding_model", "BAAI/bge-small-en-v1.5"
    )
    if state.insight_store is None:
        state.initialize_model(embedding_model_name)

    logger.debug(f"Memory Bank Strategy - Step {state.step_count}")

    # Extract user query for Observer context
    user_query_text = _get_user_query_text(messages)

    # Extract and ingest new tool outputs
    tool_outputs = extract_tool_outputs(messages)
    if tool_outputs:
        observer_model = getattr(settings, "observer_model", "gpt-4-1-mini")
        trace_ids = ingest_tool_outputs(
            tool_outputs=tool_outputs,
            user_query=user_query_text,
            fact_store=state.fact_store,
            insight_store=state.insight_store,
            llm_client=llm_client,
            observer_model=observer_model,
            step_id=state.step_count,
        )
        logger.debug(f"Ingested {len(trace_ids)} tool outputs")

    # First step or no history → pass through unchanged
    if state.insight_store.is_empty():
        logger.debug("No history yet, passing through unchanged")
        return messages, get_token_count(messages)

    # Retrieve relevant context
    top_k = getattr(settings, "top_k", 3)
    max_chars = getattr(settings, "max_chars_per_record", 2000)

    retrieved = retrieve_and_format(
        query=RETRIEVAL_QUERY,
        fact_store=state.fact_store,
        insight_store=state.insight_store,
        top_k=top_k,
        max_chars=max_chars,
    )

    # Construct compressed context
    user_query_msgs, _ = get_user_message(messages)
    last_tool_msgs, _ = get_last_tool_interaction(messages)

    processed = []

    # Part 1: Anchor (user's original task)
    if user_query_msgs:
        processed.extend(user_query_msgs)

    # Part 2: Retrieved Memory
    if retrieved:
        memory_content = format_retrieved_memory(retrieved)
        processed.append(
            {
                "role": "system",
                "content": f"## Retrieved Context from Previous Steps\n\n{memory_content}",
            }
        )
        logger.debug(f"Added {len(retrieved)} retrieved records to context")

    # Part 3: Working Memory (last tool interaction only)
    if last_tool_msgs:
        processed.extend(last_tool_msgs)

    token_count = get_token_count(processed)
    logger.debug(
        f"Context constructed: {len(processed)} messages, {token_count} tokens"
    )

    return processed, token_count
