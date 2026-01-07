import weave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.config import ExperimentConfig
from src.strategies.memory_bank.memory_bank import MemoryBank
from src.utils.split_trace import process_and_split_trace_user
from src.utils.logger import get_logger
from src.utils.token_count import get_token_count
from src.utils.trace_processing import detect_tail_loop

logger = get_logger("MemoryProcessor")

class MemoryProcessor:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.active_bank: Optional[MemoryBank] = None
        self.processed_message_ids: set = set()
        self.current_summary: str = ""
        self._load_summary_prompt()


    def _load_summary_prompt(self):
        """Utility to load and cache the summary prompt."""
        # Find the progressive_summarization strategy if configured
        config_prompt_path = "prompts/progressive_summary.md"
        for strategy in self.config.memory_strategies.values():
            if strategy.type == "progressive_summarization" and strategy.summary_prompt:
                config_prompt_path = strategy.summary_prompt
                break
        
        # parse string to Path
        prompt_path = Path(__file__).resolve().parents[0] / Path(config_prompt_path)
        try:
            self.summary_prompt = prompt_path.read_text(encoding="utf-8")   
        except FileNotFoundError:
            logger.error("Missing progressive summary prompt file at %s", prompt_path)

    def reset_state(self):
        """Called by Orchestrator to reset memory between runs."""
        if self.active_bank:
            self.active_bank.reset()
        self.processed_message_ids.clear()
        self.current_summary = ""
        logger.info("🧠 Memory State Reset")

    @weave.op()
    def apply_strategy(
        self,
        messages: List[Dict],
        strategy_key: str,
        input_token_info: Dict[str, Any],
        llm_client: Optional[Any] = None,
    ) -> Tuple[List[Dict], Dict[str, Any]]:
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
            return [{"role": "system", "content": "Infinite loop detected; aborting."}], {}

        # 1. Measure state before context processing
        # Use trace_raw_token_count if available for accurate baseline
        pre_count = input_token_info.get("raw_token_count") or get_token_count(messages)

        if pre_count < self.config.compact_threshold:
            # TODO: Context reading for memory bank and ACON should happen here nonetheless.
            return messages, input_token_info
        else:
            logger.debug(
                f"🧠 Pre-Processing Token Count: {pre_count}, exceeds compact_threshold={self.config.compact_threshold}"
            )
            # apply trace split here.

            # 2. Apply selected memory strategy
            if settings.type == "truncation":
                processed_messages, _ = self._apply_truncation(messages, pre_count, self.config.compact_threshold)
            elif settings.type == "memory_bank":
                processed_messages = self._apply_memory_bank(messages, pre_count, settings)
            elif settings.type == "progressive_summarization":
                processed_messages = self._apply_progressive_summarization(
                    messages, pre_count, settings, llm_client
                )
            else:
                logger.warning(
                    f"🧠 Unknown memory strategy type: {settings.type}. No memory strategy applied; returning original messages."
                )
                return messages, input_token_info

        # get token count from the list of processed messages
        post_count = get_token_count(processed_messages)

        output_token_info = {
            "post_token_count": post_count,
        }

        return processed_messages, output_token_info
    
    @weave.op()
    def _apply_truncation(self, messages: List[Dict], token_count: int, max_tokens: int) -> Tuple[List[Dict], int]:
        """
        Naive Baseline: Keeps only the last user query + preceding context that fits.
        Ensures chronological order is maintained.
        """
        if len(messages) <= 3:
            return messages, token_count

        user_query, conversation_history = process_and_split_trace_user(messages)
        
        # Build result: start with user query to reserve space for it
        user_token_count = get_token_count(user_query) if user_query else 0
        current_token_count = user_token_count
        
        # iterate from newest message to oldest in conversation_history to select messages
        selected_messages: List[Dict] = []
        for msg in reversed(conversation_history):
            msg_token_count = get_token_count([msg])
            if current_token_count + msg_token_count > max_tokens:
                break
            selected_messages.append(msg)
            current_token_count += msg_token_count

        # selected_messages currently has newest-to-oldest; reverse to restore chronological order
        selected_messages.reverse()
        
        # Build final result in chronological order: [older_context, user_query]
        result = selected_messages
        if user_query:
            result.extend(user_query)
        
        logger.debug(
            f"✂️  Truncated context from {token_count} to {current_token_count} tokens using max_tokens={max_tokens}"
        )
        return result, current_token_count
    
    @weave.op()
    def _apply_progressive_summarization(
        self,
        messages: List[Dict],
        token_count: int,
        settings,
        llm_client: Optional[Any],
    ) -> List[Dict]:
        """Summarizes archived context when token threshold is exceeded.
        
        Summarizes all messages before the last user query, keeping the 
        user query intact at the end.
        """
        if llm_client is None:
            raise ValueError("llm_client is required for progressive summarization")
        
        user_query, conversation_history = process_and_split_trace_user(messages)

        token_count = get_token_count(messages)

        # Don't summarize if below threshold or no archived content
        if token_count <= self.config.compact_threshold or not conversation_history:
            # Return: conversation_history + user query
            result = conversation_history.copy()
            if user_query:
                result.extend(user_query)
            return result
        
        # Build prompt for summarization
        prompt_messages = [ 
            {"role": "system", "content": self.summary_prompt},
            {"role": "user", "content": f"Conversation history to compress:\n{conversation_history}"},
        ]

        # Call LLM to generate summary (let exceptions propagate)
        summarizer_model = settings.summarizer_model or "gpt-4-1-mini"
        response = llm_client.generate_plain(
            input_messages=prompt_messages, model=summarizer_model
        )

        # Extract summary text from response
        message = response.choices[0].message
        if isinstance(message, dict):
            summary_text = (message.get("content") or "").strip()
        else:
            summary_text = (getattr(message, "content", "") or "").strip()

        if not summary_text:
            raise ValueError("Summarization returned empty content")

        # Build final message list: [summary, user query]
        summary_message = {"role": "system", "content": summary_text}
        
        result = [summary_message]
        if user_query:
            result.extend(user_query)
        
        return result

    @weave.op()
    def _apply_memory_bank(self, messages: List[Dict], token_count: int, settings) -> List[Dict]:
        """
        Memory bank strategy: Store archived context in vector DB and retrieve relevant memories.
        Keeps last user query and injects retrieved memories as context.
        """
        if self.active_bank is None:
            self.active_bank = MemoryBank(settings.embedding_model)

        # Split messages into components
        user_query, conversation_history = process_and_split_trace_user(messages)

        # Store archived messages in memory bank
        for i, msg in enumerate(conversation_history):
            try:
                msg_id = f"{i}_{len(str(msg.get('content', '')))}"
            except Exception as e:
                logger.error(f"Failed to generate message ID for memory storage: {e}")
                msg_id = f"{i}_{id(msg)}_{msg.get('role', 'unknown')}"
            
            if msg_id not in self.processed_message_ids:
                if msg["role"] in ["user", "assistant"]:
                    content = msg.get("content", "")
                    if content:
                        self.active_bank.add_memory(f"{msg['role']}: {content}")
                self.processed_message_ids.add(msg_id)

        self.active_bank.update_time()

        # Retrieve relevant memories based on last user query
        retrieved_context = []
        if user_query and user_query[0].get("content"):
            retrieved_context = self.active_bank.retrieve(
                query=user_query[0]["content"], top_k=settings.top_k or 3
            )

        # Build memory message if we have retrieved context
        memory_msg = []
        if retrieved_context:
            context_str = "\n".join(retrieved_context)
            memory_block = (
                f"Relevant Past Info: \n{context_str}\n"
                "End of Past Info"
            )
            memory_msg = [{"role": "system", "content": memory_block}]
            logger.debug(f"🧠 Injected {len(retrieved_context)} memories into context.")

        # Build result: [memory, user query]
        result = memory_msg.copy()
        if user_query:
            result.extend(user_query)
        
        logger.debug(
            f"""
            🧠 Memory Bank Context:
            MemoryMessages:{len(memory_msg)} 
            UserQuery:{1 if user_query else 0}
            FinalMessages:{len(result)}
            """
        )
        return result

    