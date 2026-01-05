from typing import List, Dict, Tuple

from src.utils.logger import get_logger

logger = get_logger("HistoryUtils")

def split_llm_trace(messages: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """
    Splits LLM trace into two components:
    1. Last User Query (List with single message)
    2. Conversation History (List of messages before last user query)
    """
    if not messages:
        return [], []
    
    # Identify user message index and store to variable
    user_msg_idx = 0
    for entry in messages:
        if entry.get("role") == "user":
            user_msg_idx = messages.index(entry)
    
    # ensure last_user_idx exists
    if user_msg_idx is None:
        logger.warning("No user message found in messages.")
        # return the first message separated from all other messages
        return [messages[-1]], messages[:-1]
    
    # store last user query to conversation history with one entry
    last_user_query = messages[user_msg_idx]

    # TODO: Later this needs to handle the tool call split better
    # Currently this behaviour leads to a task failure at all times, because tool calls history is deleted after summarization

    # return user query separated from rest of conversation history
    return [last_user_query], messages[user_msg_idx + 1 :]


def split_llm_trace_with_tools(messages: List[Dict]) -> Tuple[Dict, List[Dict], List[Dict]]:
    """
    Splits trace into three components:
    1. Last User Query (Dict - single message or empty dict)
    2. Conversation History (List of messages before last user query)
    3. Last Tool Episode (List: 1 assistant message + N tool result messages)
    
    A tool episode is atomic: the assistant's tool_calls and all corresponding
    tool results are kept together as the smallest meaningful unit. This handles
    multiple concurrent tool calls from a single assistant message.
    
    Args:
        messages: List of message dictionaries from an LLM trace
    
    Returns:
        Tuple of (last_user_query, conversation_history, last_tool_episode)
        - last_user_query: Dict containing the most recent user message (or empty dict)
        - conversation_history: List of all messages before the last user query
        - last_tool_episode: List containing assistant message with tool_calls + tool responses
    """
    if not messages:
        return {}, [], []

    # --- 1. Identify Last Tool Episode ---
    # Walk backwards to find all consecutive tool messages at the end.
    # A tool episode consists of: 1 assistant message with tool_calls + N tool result messages.
    
    tool_end_idx = len(messages)
    tool_start_idx = len(messages)
    
    # Find the start of consecutive tool messages at the end
    while tool_start_idx > 0 and messages[tool_start_idx - 1].get("role") == "tool":
        tool_start_idx -= 1
    
    # No tool messages at end - check if there's still a user query
    if tool_start_idx == tool_end_idx:
        # No tool episode, find last user message
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break
        
        if last_user_idx is None:
            # No user message found
            return {}, messages, []
        
        last_user_query = messages[last_user_idx]
        conversation_history = messages[:last_user_idx]
        return last_user_query, conversation_history, []
    
    # Tool messages found at end - validate the tool episode
    if tool_start_idx == 0:
        logger.error("Tool messages found at start of conversation (no preceding messages).")
        return {}, messages, []
    
    assistant_idx = tool_start_idx
    potential_assistant = messages[assistant_idx]
    
    if potential_assistant.get("role") != "assistant":
        logger.error(f"Expected assistant before tool messages, found: {potential_assistant.get('role')}")
        # Try to find last user message before this point
        last_user_idx = None
        for i in range(assistant_idx, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is None:
            return {}, messages, []
        return messages[last_user_idx], messages[:last_user_idx], []
    
    tool_calls = potential_assistant.get("tool_calls", [])
    
    # Check for corrupted serialization (raw class dump)
    if tool_calls and isinstance(tool_calls[0], dict) and "_type" in tool_calls[0]:
        logger.error("Last tool episode invalid: 'tool_calls' contains raw class dump/corrupted data.")
        # Find last user message
        last_user_idx = None
        for i in range(assistant_idx, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is None:
            return {}, messages, []
        return messages[last_user_idx], messages[:last_user_idx], []
    
    # Validate tool result IDs match tool_call IDs from assistant
    tool_call_ids = {tc.get("id") for tc in tool_calls if isinstance(tc, dict)}
    tool_result_ids = {
        messages[i].get("tool_call_id") 
        for i in range(tool_start_idx, tool_end_idx)
    }
    
    if not tool_result_ids.issubset(tool_call_ids):
        missing = tool_result_ids - tool_call_ids
        logger.error(f"Last tool episode invalid: Tool IDs {missing} not found in assistant tool_calls.")
        # Find last user message
        last_user_idx = None
        for i in range(assistant_idx, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break
        if last_user_idx is None:
            return {}, messages, []
        return messages[last_user_idx], messages[:last_user_idx], []
    
    # Success: extract the complete tool episode as an atomic unit
    last_tool_episode = messages[assistant_idx:tool_end_idx]
    
    # --- 2. Find Last User Query ---
    # The user query is the last user message before the assistant's tool call
    last_user_idx = None
    for i in range(assistant_idx - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break
    
    if last_user_idx is None:
        logger.warning("No user message found before tool episode.")
        return {}, messages[:assistant_idx], last_tool_episode
    
    last_user_query = messages[last_user_idx]
    
    # --- 3. Extract Conversation History ---
    # Everything before the last user query
    conversation_history = messages[:last_user_idx]

    return last_user_query, conversation_history, last_tool_episode