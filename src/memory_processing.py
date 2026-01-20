import weave
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger
from src.utils.token_count import get_token_count
from src.utils.trace_processing import detect_tail_loop
from src.utils.config import ExperimentConfig

from src.strategies.progressive_summarization.prog_sum import summarize_conv_history
from src.strategies.truncation.truncation import truncate_messages
from src.strategies.ace.ace_strategy import ACEState, apply_ace_strategy
from src.strategies.memory_bank.memory_bank import (
    MemoryBank,
    extract_retrieval_query,
    extract_tool_executions,
)

logger = get_logger("MemoryProcessor")

class MemoryProcessor:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.current_summary: str = ""
        self._ace_state = ACEState()
        self._memory_bank = MemoryBank()

    def reset_state(self):
        """Called by Orchestrator to reset memory between runs."""
        self.current_summary = ""
        self._ace_state.reset()
        self._memory_bank.reset()
        logger.info("🧠 Memory State Reset")

    @weave.op(
            enable_code_capture=False
    )
    def apply_strategy(
        self,
        messages: List[Dict],
        strategy_key: str,
        input_token_count: int,
        llm_client: Optional[Any] = None,
    ) -> Tuple[List[Dict], Optional[int]]:
        """
        Apply the configured memory strategy to the incoming messages.
        """
        settings = self.config.memory_strategies[strategy_key]
        logger.debug(f"🧠 Applying Memory Strategy: {settings.type}")

        # Loop detection to prevent infinite context growth
        if len(messages) > 20 and detect_tail_loop(messages, threshold=4, max_pattern_len=5):
            logger.error(
                f"🚨 Infinite loop detected in last {len(messages)} messages. Aborting."
            )
            return [{"role": "system", "content": "Infinite loop detected; aborting."}], None

        if settings.type == "ace":
            processed_messages, output_token_count = self._apply_ace(
                messages=messages,
                token_count=input_token_count,
                settings=settings,
                llm_client=llm_client
            )
            return processed_messages, output_token_count
        if settings.type == "memory_bank":
            processed_messages, output_token_count = self._apply_memory_bank(
                messages=messages,
                token_count=input_token_count,
                settings=settings,
                llm_client=llm_client,
            )
            return processed_messages, output_token_count

        # Other strategies only apply when token count exceeds threshold
        if input_token_count < self.config.compact_threshold:
            # TODO: Context reading for memory bank and ACON should happen here nonetheless.
            return messages, input_token_count
        else:
            logger.debug(
                f"🧠 Pre-Processing Token Count: {input_token_count}, exceeds compact_threshold={self.config.compact_threshold}"
            )
            if settings.type == "truncation":
                processed_messages, output_token_count = self._apply_truncation(messages, input_token_count)
            elif settings.type == "progressive_summarization":
                processed_messages, output_token_count = self._apply_progressive_summarization(
                    messages=messages, 
                    token_count=input_token_count, 
                    settings=settings, 
                    llm_client=llm_client
                )
            else:
                logger.warning(
                    f"🧠 Unknown memory strategy type: {settings.type}. No memory strategy applied; returning original messages."
                )
                return messages, None
            
        return processed_messages, output_token_count
    
    @weave.op(
            enable_code_capture=False
    )
    def _apply_truncation(self, messages: List[Dict], token_count: int) -> Tuple[List[Dict], int]:
        """Truncates archived context when token threshold is exceeded.
        """
        logger.debug(f"🧠 Applying Truncation Strategy. Current query with {token_count} tokens")
        truncated_conv = truncate_messages(messages)
        return truncated_conv, get_token_count(truncated_conv)

    @weave.op(
            enable_code_capture=False
    )
    def _apply_progressive_summarization(
        self,
        messages: List[Dict],
        token_count: int,
        settings,
        llm_client: Optional[Any],
    ) -> Tuple[List[Dict], int]:
        """Summarizes archived context when token threshold is exceeded.
        
        Summarizes all messages before user query.
        """
        logger.debug(f"🧠 Applying Progressive Summarization. Current query with {token_count} tokens")
        summarized_conv = summarize_conv_history(
            messages=messages, 
            llm_client=llm_client, 
            summarizer_model=settings.summarizer_model,
            summary_prompt_path=settings.summary_prompt,
            )
        return summarized_conv, get_token_count(summarized_conv)
    
    @weave.op(
            enable_code_capture=False
    )
    def _apply_ace(
        self,
        messages: List[Dict],
        token_count: int,
        settings,
        llm_client: Optional[Any]
    ) -> Tuple[List[Dict], int]:
        """Applies ACE strategy by delegating to ace_strategy module."""
        
        logger.debug(f"🧠 Applying ACE Strategy. Current query with {token_count} tokens")
        processed, new_count = apply_ace_strategy(
            messages, llm_client, settings, self._ace_state
        )
        return processed, new_count

    @weave.op(
            enable_code_capture=False
    )
    def _apply_memory_bank(
        self,
        messages: List[Dict],
        token_count: int,
        settings,
        llm_client: Optional[Any],
    ) -> Tuple[List[Dict], int]:
        if llm_client is None:
            raise ValueError("llm_client is required for memory bank strategy")

        user_query = extract_retrieval_query(messages)
        if not user_query:
            for message in messages:
                if message.get("role") == "user":
                    user_query = message.get("content", "") or ""

        tool_executions = extract_tool_executions(messages)
        self._memory_bank.top_k = settings.top_k or self._memory_bank.top_k
        if settings.raw_data_char_limit:
            self._memory_bank.record_char_limit = settings.raw_data_char_limit
        self._memory_bank.ingest_tool_executions(
            user_query=user_query,
            tool_executions=tool_executions,
            llm_client=llm_client,
        )

        retrieved = ""
        if user_query:
            retrieved = self._memory_bank.retrieve_relevant_history(user_query)

        if not retrieved:
            return messages, token_count

        memory_message = {
            "role": "system",
            "content": f"Here is the information you requested from memory:\n{retrieved}",
        }
        updated_messages = list(messages)
        insert_idx = next(
            (idx + 1 for idx, msg in enumerate(updated_messages) if msg.get("role") == "system"),
            0,
        )
        updated_messages.insert(insert_idx, memory_message)
        return updated_messages, get_token_count(updated_messages)
