import logging
from typing import List, Dict, Any, Tuple, Optional

from src.utils.logger import get_logger

logger = get_logger("HistoryUtils")

def split_llm_trace(messages: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Splits trace into three lists of message dicts:
    1. System Prompt (List containing 0 or 1 message)
    2. Conversation History (List of messages)
    3. Last Tool Exchange (List containing 0 or 2 messages: the call and the result)
    
    Args:
        messages: List of message dictionaries from an LLM trace
    
    Returns:
        Tuple of (system_msgs, conversation_history, last_tool_msgs)
    """
    if not messages:
        return [], [], []

    # --- 1. Extract System Prompt ---
    system_msgs: List[Dict] = []
    start_idx = 0
    
    if messages[0].get("role") == "system":
        system_msgs = [messages[0]]
        start_idx = 1

    # --- 2. Identify and Validate Last Tool Interaction ---
    # Default: assume no valid tool tail, so history goes to the end
    history_end_idx = len(messages)
    last_tool_msgs: List[Dict] = []
    
    # We need at least 2 messages to have a pair (Assistant Query -> Tool Result)
    if len(messages) >= 2:
        last_msg = messages[-1]
        second_last_msg = messages[-2]

        if last_msg.get("role") == "tool" and second_last_msg.get("role") == "assistant":
            
            # validation: Check for data corruption in the assistant message
            tool_calls = second_last_msg.get("tool_calls", [])
            
            # Your sample data has a corrupted object here instead of a list of dicts with 'function' keys
            is_corrupted = False
            valid_call_found = False
            
            # Check if tool_calls contains the corrupted class dump
            if tool_calls and isinstance(tool_calls[0], dict) and "_type" in tool_calls[0]:
                is_corrupted = True
            
            if not is_corrupted:
                target_id = last_msg.get("tool_call_id")
                # Ensure the assistant actually requested this specific ID
                valid_call_found = any(tc.get("id") == target_id for tc in tool_calls)

            if valid_call_found:
                # Success: Split this pair off from history
                last_tool_msgs = [second_last_msg, last_msg]
                history_end_idx = len(messages) - 2
            else:
                # Failure: Log specific error
                if is_corrupted:
                    logger.error("Last tool bit invalid: 'tool_calls' contains raw class dump/corrupted data.")
                else:
                    logger.error(f"Last tool bit invalid: Tool ID {last_msg.get('tool_call_id')} not found in preceding assistant message.")

    # --- 3. Extract History ---
    conversation_history = messages[start_idx:history_end_idx]

    return system_msgs, conversation_history, last_tool_msgs



def segment_message_history(messages: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split conversation into pinned nodes, archived context, and working memory."""
    prefix, start_idx = _extract_pinned_prefix(messages)
    conversation = messages[start_idx:]
    tail_start = _find_tail_start(conversation)

    return prefix, conversation[:tail_start], conversation[tail_start:]


def _extract_pinned_prefix(messages: List[Dict]) -> tuple[List[Dict], int]:
    prefix: List[Dict] = []
    idx = 0

    while idx < len(messages):
        msg = messages[idx]
        if msg.get("role") == "system": 
            prefix.append(msg)
            idx += 1
            continue
        break

    return prefix, idx


def _find_tail_start(conversation: List[Dict]) -> int:
    if not conversation:
        return 0

    last_user_idx = _find_last_role_index(conversation, "user")
    if last_user_idx is not None:
        return last_user_idx

    fallback_idx = max(0, len(conversation) - 1)
    return _collapse_tool_sequence(conversation, fallback_idx)


def _find_last_role_index(conversation: List[Dict], role: str) -> Optional[int]:
    for idx in range(len(conversation) - 1, -1, -1):
        if conversation[idx].get("role") == role:
            return idx
    return None


def _collapse_tool_sequence(conversation: List[Dict], idx: int) -> int:
    while idx > 0 and conversation[idx].get("role") == "tool":
        idx -= 1
    return idx
