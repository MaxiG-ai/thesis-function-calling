"""
Decorators for memory processing methods to add visibility and metrics.
"""

import time
import logging
from functools import wraps
from typing import List, Dict, Callable, Optional
from .token_counter import count_tokens, count_messages

logger = logging.getLogger("MemoryDecorator")


def log_token_reduction(func: Callable) -> Callable:
    """
    Decorator that logs comprehensive metrics for memory strategy application.
    
    Logs at INFO level:
    - Token count before and after (absolute and percentage reduction)
    - Message count before and after
    - Context window utilization percentage (before and after)
    - Execution time for the strategy
    
    The decorator expects the method to be part of a MemoryProcessor instance
    that has:
    - self.config: ExperimentConfig with model_registry
    - self.current_model_key: str indicating active model
    """
    
    @wraps(func)
    def wrapper(self, messages: List[Dict], *args, **kwargs) -> List[Dict]:
        # Capture metrics BEFORE strategy application
        start_time = time.perf_counter()
        tokens_before = count_tokens(messages)
        messages_before = count_messages(messages)
        
        # Get context window size for utilization calculation
        context_window: Optional[int] = None
        if hasattr(self, 'current_model_key') and self.current_model_key:
            try:
                if hasattr(self, 'config') and self.current_model_key in self.config.model_registry:
                    context_window = self.config.model_registry[self.current_model_key].context_window
            except Exception as e:
                logger.debug(f"Could not retrieve context window: {e}")
        
        # Execute the actual memory strategy
        result = func(self, messages, *args, **kwargs)
        
        # Capture metrics AFTER strategy application
        end_time = time.perf_counter()
        duration = end_time - start_time
        tokens_after = count_tokens(result)
        messages_after = count_messages(result)
        
        # Calculate reductions
        token_reduction = tokens_before - tokens_after
        token_reduction_pct = (token_reduction / tokens_before * 100) if tokens_before > 0 else 0.0
        message_reduction = messages_before - messages_after
        
        # Calculate context window utilization
        context_usage_before = None
        context_usage_after = None
        if context_window and context_window > 0:
            context_usage_before = (tokens_before / context_window) * 100
            context_usage_after = (tokens_after / context_window) * 100
        
        # Format the log message
        strategy_name = func.__name__.replace('_apply_', '')
        
        log_parts = [
            f"🔢 Memory Strategy '{strategy_name}' metrics:",
            f"   Tokens: {tokens_before} → {tokens_after} ({-token_reduction:+d}, {-token_reduction_pct:+.1f}%)"
            f" | Messages: {messages_before} → {messages_after} ({-message_reduction:+d})"
        ]
        
        if context_usage_before is not None and context_usage_after is not None:
            log_parts.append(
                f"   Context usage: {context_usage_before:.1f}% → {context_usage_after:.1f}%"
                f" (of {context_window:,} tokens) | Duration: {duration:.3f}s"
            )
        else:
            log_parts.append(f"   Duration: {duration:.3f}s")
        
        logger.info("\n".join(log_parts))
        
        return result
    
    return wrapper
