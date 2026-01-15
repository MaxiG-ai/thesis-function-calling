from langfuse import observe
from typing import Any, Dict, List, Optional, Tuple

from src.utils.logger import get_logger
from src.utils.token_count import get_token_count
from src.utils.trace_processing import detect_tail_loop
from src.utils.config import ExperimentConfig

from src.strategies.progressive_summarization.prog_sum import summarize_conv_history
from src.strategies.truncation.truncation import truncate_messages
from src.strategies.ace.ace_strategy import ACEState, apply_ace_strategy

logger = get_logger("MemoryProcessor")

class MemoryProcessor:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.current_summary: str = ""
        self._ace_state = ACEState()

    def reset_state(self):
        """Called by Orchestrator to reset memory between runs."""
        self.current_summary = ""
        self._ace_state.reset()
        logger.info("🧠 Memory State Reset")

    @observe()
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

        # ACE strategy should be applied at all times as it's a playbook-based learning system
        # that builds and refines knowledge regardless of token count
        if settings.type == "ace":
            processed_messages, output_token_count = self._apply_ace(
                messages=messages,
                token_count=input_token_count,
                settings=settings,
                llm_client=llm_client
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
            elif settings.type == "memory_bank":
                raise NotImplementedError("Memory Bank strategy not yet implemented")
            else:
                logger.warning(
                    f"🧠 Unknown memory strategy type: {settings.type}. No memory strategy applied; returning original messages."
                )
                return messages, None
            
        return processed_messages, output_token_count
    
    @observe()
    def _apply_truncation(self, messages: List[Dict], token_count: int) -> Tuple[List[Dict], int]:
        """Truncates archived context when token threshold is exceeded.
        """
        logger.debug(f"🧠 Applying Truncation Strategy. Current query with {token_count} tokens")
        truncated_conv = truncate_messages(messages)
        return truncated_conv, get_token_count(truncated_conv)

    @observe()
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
    
    @observe()
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