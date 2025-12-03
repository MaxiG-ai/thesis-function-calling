from typing import List, Dict, Optional
from .config import ExperimentConfig
from .strategies.memory_bank.memory_bank import MemoryBank
from .utils.memory_decorators import log_token_reduction
from .utils.logger import get_logger

logger = get_logger("MemoryProcessor")


class MemoryProcessor:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.active_bank: Optional[MemoryBank] = None
        self.processed_message_ids: set = set()
        self.current_model_key: Optional[str] = None

    def set_current_model(self, model_key: str):
        """
        Set the active model for context window calculations.
        Called by orchestrator before applying strategies.
        """
        self.current_model_key = model_key
        logger.debug(f"Set current model to: {model_key}")

    def reset_state(self):
        "Called by Orchestrator to reset memory between runs."
        if self.active_bank:
            self.active_bank.reset()
        self.processed_message_ids.clear()
        logger.info("🧠 Memory State Reset")

    def apply_strategy(self, messages: List[Dict], strategy_key: str) -> List[Dict]:
        """
        The core thesis function.
        Transforms Input Messages -> Optimized Messages based on active strategy.
        """
        # 1. Get the active strategy settings
        if strategy_key not in self.config.memory_strategies:
            logger.warning(f"Strategy {strategy_key} not found. returning raw context.")
            return messages

        settings = self.config.memory_strategies[strategy_key]
        logger.info(f"🧠 Applying Memory Strategy: {settings.type}")

        # 2. Route to specific implementation
        if settings.type == "truncation":
            return self._apply_truncation(messages, settings.max_tokens or 2000)
        elif settings.type == "memory_bank":
            return self._apply_memory_bank(messages, settings)

        return messages

    @log_token_reduction
    def _apply_truncation(self, messages: List[Dict], max_tokens: int) -> List[Dict]:
        """
        Naive Baseline: Keeps only the system prompt + last N messages.
        """
        if len(messages) <= 2:
            return messages

        system_msg = [m for m in messages if m["role"] == "system"]
        # Simple heuristic: keep last 3 messages if we are 'truncating'
        recent_history = messages[-3:]

        logger.info(
            f"✂️  Truncated context from {len(messages)} to {len(system_msg) + len(recent_history)} msgs"
        )
        return system_msg + recent_history

    @log_token_reduction
    def _apply_memory_bank(self, messages: List[Dict], settings) -> List[Dict]:
        """
        Implements Memory Bank Retrieval logic.
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
                msg_id = f"{i}_no_content"
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

        # Keep last N turn to maintain conversation flow
        working_memory_limit = 2
        recent_msgs = messages[-working_memory_limit:]

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

        return system_msgs + memory_msg + recent_msgs
