import weave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.config import ExperimentConfig
from src.strategies.memory_bank.memory_bank import MemoryBank
from src.utils.history import split_llm_trace
from src.utils.logger import get_logger
from src.utils.token_count import get_token_count

logger = get_logger("MemoryProcessor")


def detect_tail_loop(messages: list[dict], threshold: int = 4, max_pattern_len: int = 10) -> bool:
    """Detects repeating patterns at the tail of a conversation."""
    n = len(messages)
    # Optimization: Don't check if history is too short to contain a loop
    if n < threshold:
        return False

    # 1. Normalize messages for comparison
    # We must exclude fields that change every turn even in a loop (like tool_call_id)
    normalized = []
    for m in messages[-(max_pattern_len * threshold):]: # Only look at the relevant tail
        # Create a signature tuple: (Role, Content, Sorted Tool Calls)
        tool_sig = None
        if "tool_calls" in m:
            tool_sig = sorted(
                [(tc.type, tc.function.name, tc.function.arguments) for tc in m["tool_calls"]]
            )
            tool_sig = tuple(tool_sig)

        normalized.append((m.get("role"), m.get("content"), tool_sig))

    # Re-calculate length based on the slice we actually took
    n_slice = len(normalized)

    # 2. Check for patterns of length L
    # We iterate L from 1 (repeating single message) up to max_pattern_len
    for L in range(1, max_pattern_len + 1):
        # We need at least L * threshold messages to verify this pattern
        if n_slice < L * threshold:
            break
            
        # The "candidate" pattern is the very last L messages
        pattern = normalized[-L:]
        
        # Check if this pattern appears 'threshold' times backwards
        is_loop = True

        for k in range(1, threshold):
            # Compare the block before the current one
            # e.g., if L=2, threshold=3:
            # Check [-2:] vs [-4:-2]
            # Check [-2:] vs [-6:-4]
            prev_block = normalized[-(k + 1) * L : -k * L]
            if prev_block != pattern:
                is_loop = False
                break

        if is_loop:
            return True

    return False


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
            # 2. Apply selected memory strategy
            if settings.type == "truncation":
                processed_messages = self._apply_truncation(messages, pre_count, self.config.compact_threshold)
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

        post_count = get_token_count(processed_messages)

        output_token_info = {
            "post_token_count": post_count,
        }

        return processed_messages, output_token_info
    
    @weave.op()
    def _apply_truncation(self, messages: List[Dict], token_count: int, max_tokens: int) -> List[Dict]:
        """
        Naive Baseline: Keeps only the last user query + tool episode.
        Ensures assistant+tool message pairs are kept together.
        """
        if len(messages) <= 3:
            return messages

        last_user_query, _, last_tool_episode = split_llm_trace(messages)
        
        # Build result: user query + tool episode (if present)
        result = []
        if last_user_query:
            result.append(last_user_query)
        result.extend(last_tool_episode)

        logger.debug(
            f"✂️  Truncated context from {len(messages)} to {len(result)} msgs using max_tokens={max_tokens}"
        )
        return result
    
    @weave.op()
    def _apply_memory_bank(self, messages: List[Dict], token_count: int, settings) -> List[Dict]:
        """
        Memory bank strategy: Store archived context in vector DB and retrieve relevant memories.
        Keeps last user query + tool episode, and injects retrieved memories as context.
        """
        if self.active_bank is None:
            self.active_bank = MemoryBank(settings.embedding_model)

        # Split messages into components
        last_user_query, archived_context, last_tool_episode = split_llm_trace(messages)

        # Store archived messages in memory bank
        for i, msg in enumerate(archived_context):
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
        if last_user_query and last_user_query.get("content"):
            retrieved_context = self.active_bank.retrieve(
                query=last_user_query["content"], top_k=settings.top_k or 3
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

        # Build result: memory + user query + tool episode
        result = memory_msg.copy()
        if last_user_query:
            result.append(last_user_query)
        result.extend(last_tool_episode)
        
        logger.debug(
            f"""🧠 Memory Bank Context:
MemoryMessages:{len(memory_msg)} 
UserQuery:{1 if last_user_query else 0}
ToolEpisode:{len(last_tool_episode)}
FinalMessages:{len(result)}
"""
        )
        return result

    @weave.op()
    def _apply_progressive_summarization(
        self,
        messages: List[Dict],
        token_count: int,
        settings,
        llm_client: Optional[Any],
    ) -> List[Dict]:
        """Summarizes archived context when token threshold is exceeded.
        
        Re-summarizes all archived context (middle of conversation) each time,
        keeping the last user query and tool episode intact.
        """
        if llm_client is None:
            raise ValueError("llm_client is required for progressive summarization")
        
        last_user_query, archived_context, last_tool_episode = split_llm_trace(messages)

        token_count = get_token_count(messages)

        # Don't summarize if below threshold or no archived content
        if token_count <= self.config.compact_threshold or not archived_context:
            # Return: user query + tool episode (if present)
            result = []
            if last_user_query:
                result.append(last_user_query)
            result.extend(last_tool_episode)
            return result
        
        # Build prompt for summarization
        prompt_messages = [
            {"role": "system", "content": self.summary_prompt},
            {"role": "user", "content": f"Conversation history to compress:\n{archived_context}"},
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

        # Build final message list: user query + summary + tool episode
        summary_message = {"role": "system", "content": summary_text}
        logger.debug(
            f"📝 Summarized {len(archived_context)} messages into {len(summary_text)} chars"
        )
        
        result = []
        if last_user_query:
            result.append(last_user_query)
        result.append(summary_message)
        result.extend(last_tool_episode)
        
        return result
