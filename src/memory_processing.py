import logging
from typing import List, Dict
from .config import ExperimentConfig

logger = logging.getLogger("MemoryProcessor")


class MemoryProcessor:
    def __init__(self, config: ExperimentConfig):
        self.config = config

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
        elif settings.type == "summarization":
            return self._apply_summarization_stub(messages, settings)

        return messages

    def _apply_truncation(self, messages: List[Dict], max_tokens: int) -> List[Dict]:
        """
        Naive Baseline: Keeps only the system prompt + last N messages.
        (Real implementation should count tokens, this is just message count for demo)
        """
        if len(messages) <= 2:
            return messages

        system_msg = [m for m in messages if m["role"] == "system"]
        # Simple heuristic: keep last 5 messages if we are 'truncating'
        # In your real thesis, you'll use tiktoken here to measure exactly.
        recent_history = messages[-5:]

        logger.info(
            f"✂️  Truncated context from {len(messages)} to {len(system_msg) + len(recent_history)} msgs"
        )
        return system_msg + recent_history

    def _apply_summarization_stub(self, messages: List[Dict], settings) -> List[Dict]:
        """
        Placeholder for your advanced logic.
        """
        logger.info("🤔 (Stub) Summarizing tool outputs...")
        return messages
