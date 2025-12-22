import logging
from typing import List, Dict, Any, Tuple, Optional

from src.utils.logger import get_logger

logger = get_logger("HistoryUtils")

def split_llm_trace(messages: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Splits trace into three lists of message dicts:
    1. System Prompt (List containing 0 or 1 message)
    2. Conversation History (List of messages)
    3. Last Tool Episode (List: 1 assistant message + N tool result messages)
    
    A tool episode is atomic: the assistant's tool_calls and all corresponding
    tool results are kept together as the smallest meaningful unit. This handles
    multiple concurrent tool calls from a single assistant message.
    
    Args:
        messages: List of message dictionaries from an LLM trace
    
    Returns:
        Tuple of (system_msgs, conversation_history, last_tool_episode)
    """
    if not messages:
        return [], [], []

    # --- 1. Extract System Prompt ---
    system_msgs: List[Dict] = []
    start_idx = 0
    
    if messages[0].get("role") == "system":
        system_msgs = [messages[0]]
        start_idx = 1

    # --- 2. Identify Last Tool Episode ---
    # Walk backwards to find all consecutive tool messages at the end.
    # A tool episode consists of: 1 assistant message with tool_calls + N tool result messages.
    history_end_idx = len(messages)
    last_tool_msgs: List[Dict] = []
    
    tool_end_idx = len(messages)
    tool_start_idx = len(messages)
    
    # Find the start of consecutive tool messages
    while tool_start_idx > start_idx and messages[tool_start_idx - 1].get("role") == "tool":
        tool_start_idx -= 1
    
    # No tool messages at end - nothing to extract
    if tool_start_idx == tool_end_idx:
        conversation_history = messages[start_idx:history_end_idx]
        return system_msgs, conversation_history, last_tool_msgs
    
    # Check for preceding assistant message with tool_calls
    if tool_start_idx <= start_idx:
        logger.error("Tool messages found but no preceding assistant message.")
        conversation_history = messages[start_idx:history_end_idx]
        return system_msgs, conversation_history, last_tool_msgs
    
    assistant_idx = tool_start_idx - 1
    potential_assistant = messages[assistant_idx]
    
    if potential_assistant.get("role") != "assistant":
        logger.error(f"Expected assistant before tool messages, found: {potential_assistant.get('role')}")
        conversation_history = messages[start_idx:history_end_idx]
        return system_msgs, conversation_history, last_tool_msgs
    
    tool_calls = potential_assistant.get("tool_calls", [])
    
    # Check for corrupted serialization (raw class dump)
    if tool_calls and isinstance(tool_calls[0], dict) and "_type" in tool_calls[0]:
        logger.error("Last tool episode invalid: 'tool_calls' contains raw class dump/corrupted data.")
        conversation_history = messages[start_idx:history_end_idx]
        return system_msgs, conversation_history, last_tool_msgs
    
    # Validate tool result IDs match tool_call IDs from assistant
    tool_call_ids = {tc.get("id") for tc in tool_calls if isinstance(tc, dict)}
    tool_result_ids = {
        messages[i].get("tool_call_id") 
        for i in range(tool_start_idx, tool_end_idx)
    }
    
    if not tool_result_ids.issubset(tool_call_ids):
        missing = tool_result_ids - tool_call_ids
        logger.error(f"Last tool episode invalid: Tool IDs {missing} not found in assistant tool_calls.")
        conversation_history = messages[start_idx:history_end_idx]
        return system_msgs, conversation_history, last_tool_msgs
    
    # Success: extract the complete tool episode as an atomic unit
    last_tool_msgs = messages[assistant_idx:tool_end_idx]
    history_end_idx = assistant_idx

    # --- 3. Extract History ---
    conversation_history = messages[start_idx:history_end_idx]

    return system_msgs, conversation_history, last_tool_msgs



# def segment_message_history(messages: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
#     """Split conversation into pinned nodes, archived context, and working memory."""
#     prefix, start_idx = _extract_pinned_prefix(messages)
#     conversation = messages[start_idx:]
#     tail_start = _find_tail_start(conversation)

#     return prefix, conversation[:tail_start], conversation[tail_start:]


# def _extract_pinned_prefix(messages: List[Dict]) -> tuple[List[Dict], int]:
#     prefix: List[Dict] = []
#     idx = 0

#     while idx < len(messages):
#         msg = messages[idx]
#         if msg.get("role") == "system": 
#             prefix.append(msg)
#             idx += 1
#             continue
#         break

#     return prefix, idx


# def _find_tail_start(conversation: List[Dict]) -> int:
#     if not conversation:
#         return 0

#     last_user_idx = _find_last_role_index(conversation, "user")
#     if last_user_idx is not None:
#         return last_user_idx

#     fallback_idx = max(0, len(conversation) - 1)
#     return _collapse_tool_sequence(conversation, fallback_idx)


# def _find_last_role_index(conversation: List[Dict], role: str) -> Optional[int]:
#     for idx in range(len(conversation) - 1, -1, -1):
#         if conversation[idx].get("role") == role:
#             return idx
#     return None


# def _collapse_tool_sequence(conversation: List[Dict], idx: int) -> int:
#     while idx > 0 and conversation[idx].get("role") == "tool":
#         idx -= 1
#     return idx
