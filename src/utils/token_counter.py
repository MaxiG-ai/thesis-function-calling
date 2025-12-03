"""
Token counting utility for measuring context size.
Uses tiktoken for accurate token counting.
"""

import logging
from typing import List, Dict

logger = logging.getLogger("TokenCounter")

# Try to import tiktoken, fall back to simple estimation if unavailable
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
    # Use cl100k_base encoding (GPT-4, GPT-3.5-turbo, etc.)
    _encoding = tiktoken.get_encoding("cl100k_base")
except ImportError:
    TIKTOKEN_AVAILABLE = False
    _encoding = None
    logger.warning("tiktoken not available, using fallback token estimation")


def count_tokens(messages: List[Dict]) -> int:
    """
    Count total tokens across all messages in a conversation context.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        
    Returns:
        Approximate token count for the entire message list
    """
    if not messages:
        return 0
    
    total_tokens = 0
    
    for message in messages:
        # Count tokens in the message structure
        # Format is typically: role + content + message separators
        role = message.get("role", "")
        content = message.get("content", "")
        
        if TIKTOKEN_AVAILABLE and _encoding:
            # Accurate counting with tiktoken
            # Add tokens for role and content
            total_tokens += len(_encoding.encode(role))
            total_tokens += len(_encoding.encode(content))
            # Add ~4 tokens per message for formatting (<|im_start|>, role, etc.)
            total_tokens += 4
        else:
            # Fallback: rough estimation (1 token ≈ 4 characters)
            total_tokens += len(role) // 4
            total_tokens += len(content) // 4
            total_tokens += 4
    
    # Add ~2 tokens for the overall message list structure
    total_tokens += 2
    
    return total_tokens


def count_messages(messages: List[Dict]) -> int:
    """
    Count the number of messages in the context.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Number of messages
    """
    return len(messages) if messages else 0
