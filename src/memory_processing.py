import weave
import tiktoken
from typing import List, Dict, Optional
from src.utils.config import ExperimentConfig
from src.strategies.memory_bank.memory_bank import MemoryBank
from src.utils.logger import get_logger

logger = get_logger("MemoryProcessor")

def get_token_count(messages: list[dict], model: str) -> int:
    """Utility to count tokens."""
    enc = tiktoken.encoding_for_model(model)
    text = "".join(str(m) for m in messages) # Simplified for demo
    return len(enc.encode(text))

class MemoryProcessor:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.active_bank: Optional[MemoryBank] = None
        self.processed_message_ids: set = set()

    def reset_state(self):
        "Called by Orchestrator to reset memory between runs."
        if self.active_bank:
            self.active_bank.reset()
        self.processed_message_ids.clear()
        logger.info("🧠 Memory State Reset")

    @weave.op()
    def apply_strategy(self, messages: List[Dict], strategy_key: str) -> List[Dict]:
        """
        The core thesis function.
        Transforms Input Messages -> Optimized Messages based on active strategy.
        """

        settings = self.config.memory_strategies[strategy_key]
        logger.info(f"🧠 Applying Memory Strategy: {settings.type}")

        model="gpt-4-1-mini"
        limit = 128000
        # 1. Measure Pre-State
        pre_count = get_token_count(messages, model=model)
        pre_fill_pct = (pre_count / limit) * 100

        # Route to specific implementation       
        if settings.type == "truncation":
            processed_messages = self._apply_truncation(messages, settings.max_tokens or 2000)
        elif settings.type == "memory_bank":
            processed_messages = self._apply_memory_bank(messages, settings)
        else:
            logger.error(f"Unknown memory strategy type: {settings.type}. Returning original messages.")
            processed_messages = self._apply_no_strategy(messages)

        post_count = get_token_count(processed_messages, model=model)
        post_fill_pct = (post_count / limit) * 100
        reduction_pct = 100 - ((post_count / pre_count) * 100)

        # 3. Log the "Delta" metrics clearly
        cc = weave.require_current_call()
        
        if cc.summary is not None:
            cc.summary.update({
                "context_limit": limit,
                "pre_compression_tokens": pre_count,
                "post_compression_tokens": post_count,
                "context_filled_pre_pct": round(pre_fill_pct, 2),
                "context_filled_post_pct": round(post_fill_pct, 2),
                "compression_reduction_pct": round(reduction_pct, 2)
            })

        return processed_messages
    
    def _apply_no_strategy(self, messages: List[Dict]) -> List[Dict]:
        """
        No memory processing; return messages as-is.
        """
        logger.info("🧠 No memory strategy applied; returning original messages.")
        return messages

    # TODO: This function is currently not used, but kept for future reference if needed. 
    # It would be better to just design memory techniques so that they follow the tool loop correctly.
    def _validate_and_repair_tool_pairs(self, processed_messages: List[Dict], original_messages: List[Dict]) -> List[Dict]:
        """
        Ensures that tool messages always have their corresponding assistant message with tool_calls.
        If a tool message is orphaned, this function finds and injects the missing assistant message.
        
        Args:
            processed_messages: Messages after memory strategy has been applied
            original_messages: Original unprocessed messages for lookup
            
        Returns:
            Validated and repaired message list
        """
        validated = []
        i = 0
        
        while i < len(processed_messages):
            msg = processed_messages[i]
            
            # If we encounter a tool message, ensure its assistant message with tool_calls is present
            if msg.get('role') == 'tool':
                # Check if the previous message is an assistant with tool_calls
                has_valid_preceding = (
                    len(validated) > 0 and 
                    validated[-1].get('role') == 'assistant' and 
                    'tool_calls' in validated[-1]
                )
                
                if not has_valid_preceding:
                    # Find the corresponding assistant message in original_messages
                    tool_call_id = msg.get('tool_call_id')
                    assistant_msg = None
                    
                    # Search backwards in original messages for the assistant message with this tool_call_id
                    for orig_idx in range(len(original_messages) - 1, -1, -1):
                        orig_msg = original_messages[orig_idx]
                        if (orig_msg.get('role') == 'assistant' and 
                            'tool_calls' in orig_msg):
                            # Check if this assistant message has the matching tool_call_id
                            for tc in orig_msg.get('tool_calls', []):
                                if tc.id == tool_call_id: #change from tc.get('id') to tc.id
                                    assistant_msg = orig_msg
                                    break
                            if assistant_msg:
                                break
                    
                    if assistant_msg:
                        logger.debug("🔧 Injecting missing assistant message before tool response")
                        validated.append(assistant_msg)
                    else:
                        logger.warning(f"⚠️ Could not find assistant message for tool call {tool_call_id}, skipping tool message")
                        i += 1
                        continue
                
                validated.append(msg)
            else:
                validated.append(msg)
            
            i += 1
        
        return validated

    @weave.op()
    def _apply_truncation(self, messages: List[Dict], max_tokens: int) -> List[Dict]:
        """
        Naive Baseline: Keeps only the system prompt + last N messages.
        Ensures assistant+tool message pairs are kept together.
        """
        if len(messages) <= 3:
            return messages

        # get the user messages and system messages
        system_msgs = [m for m in messages if m["role"] == "system"]
        user_msgs = [m for m in messages if m["role"] == "user"]
        
        # Start with last 4 messages, but expand to include complete tool call sequences
        recent_count = 4
        recent_start_idx = max(0, len(messages) - recent_count)
        
        # Check if we're cutting off in the middle of a tool call sequence
        # If the first message in our selection is a tool message, we need its assistant message
        while recent_start_idx > 0 and messages[recent_start_idx].get('role') == 'tool':
            recent_start_idx -= 1
            # Also need to include the assistant message with tool_calls
            if recent_start_idx > 0 and messages[recent_start_idx].get('role') == 'assistant':
                if 'tool_calls' not in messages[recent_start_idx]:
                    # Keep going back to find the assistant with tool_calls
                    recent_start_idx -= 1
        
        recent_history = system_msgs + user_msgs + messages[recent_start_idx:]
        
        # Validate and repair any remaining issues
        result = recent_history

        logger.info(
            f"✂️  Truncated context from {len(messages)} to {len(result)} msgs"
        )
        return result

    @weave.op()
    def _apply_memory_bank(self, messages: List[Dict], settings) -> List[Dict]:
        """
        Implements Memory Bank Retrieval logic.
        Ensures assistant+tool message pairs are kept together in working memory.
        """        
        # Initialize Bank if needed
        if self.active_bank is None:
            self.active_bank = MemoryBank(settings.embedding_model)

        # 1. Update Bank with new messages (Store Phase)

        for i, msg in enumerate(messages[:-1]):
            try:
                msg_id = f"{i}_{len(msg['content'])}"
            except Exception as e:
                logger.error(f"Failed to generate message ID for memory storage: {e}")
                msg_id = f"{i}_{id(msg)}_{msg.get('role', 'unknown')}"
            if msg_id not in self.processed_message_ids:
                if msg["role"] in ["user", "assistant", "tools"]:
                    self.active_bank.add_memory(f"{msg['role']}: {msg['content']}")
                self.processed_message_ids.add(msg_id)

        # 2. Advance Time (Forgetting Phase)
        self.active_bank.update_time()

        # 3. Retrieve Context for Current Query (Retrieval Phase)
        last_msg = messages[-1]
        retrieved_context = []
        if last_msg['role'] == "user":
            retrieved_context = self.active_bank.retrieve(
                query = last_msg['content'], 
                top_k = settings.top_k or 3,
            )
        
        # 4. Construct Final Context (System + Active Memory Context + Recent History/Working Memory)

        system_msgs = [m for m in messages if m['role'] == "system"]

        # Keep last N messages in working memory, but expand to include complete tool call sequences
        working_memory_limit = 3
        recent_start_idx = max(0, len(messages) - working_memory_limit)
        
        # Check if we're cutting off in the middle of a tool call sequence
        # If the first message in our selection is a tool message, we need its assistant message
        while recent_start_idx > 0 and messages[recent_start_idx].get('role') == 'tool':
            recent_start_idx -= 1
            # Also need to include the assistant message with tool_calls
            if recent_start_idx > 0 and messages[recent_start_idx].get('role') == 'assistant':
                if 'tool_calls' not in messages[recent_start_idx]:
                    # Keep going back to find the assistant with tool_calls
                    recent_start_idx -= 1
        
        recent_msgs = messages[recent_start_idx:]

        # create memory context message
        context_str = "\n".join(retrieved_context)
        memory_msg = []
        if context_str:
            memory_block = (
                f"Relevant Past Info: \n{context_str}\n"
                "End of Past Info"
            )
            memory_msg = [{"role": "system", "content": memory_block}]
            len_retrieved_context = len(retrieved_context) if retrieved_context else 0
            logger.info(f"🧠 Injected {len_retrieved_context} memories into context.")

        # Validate and repair any remaining issues
        result = system_msgs + memory_msg + recent_msgs
        # result = self._validate_and_repair_tool_pairs(result, messages)

        logger.debug(
            f"""🧠 Memory Bank Context:
SystemMessages:{len(system_msgs)} 
MemoryMessages:{len(memory_msg)} 
RecentMessages:{len(recent_msgs)}
FinalMessages:{len(result)}
"""
        )
        return result
