# contains utilties to split llm traces
from typing import List, Dict, Tuple


def get_user_message(messages: List[Dict]) -> List[Dict]:
    """Get user message(s) from a list of messages.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        List of user messages (can be empty if no user messages found)
    """
    if not messages:
        return []
    
    user_messages = []
    for msg in messages:
        if msg.get("role") == "user":
            user_messages.append(msg)
    
    return user_messages


def get_last_tool_interaction(messages: List[Dict]) -> List[Dict]:
    """Get the last tool interaction from a list of messages.
    
    Extracts the last tool episode (final assistant message with tool_calls 
    + all corresponding tool response messages). Returns an atomic unit where
    tool call IDs match and no corrupted data exists.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        List containing the last tool episode [assistant_msg, tool_msg1, ...],
        or empty list if no valid tool episode found at the end
    """
    if not messages:
        return []
    
    # Walk backwards to find consecutive tool messages at the end
    tool_end_idx = len(messages)
    tool_start_idx = len(messages)
    
    # Find the start of consecutive tool messages at the end
    while tool_start_idx > 0 and messages[tool_start_idx - 1].get("role") == "tool":
        tool_start_idx -= 1
    
    # No tool messages at end
    if tool_start_idx == tool_end_idx:
        return []
    
    # Need at least one message before the tool messages (the assistant)
    if tool_start_idx == 0:
        return []
    
    # Check the message before tool messages - should be assistant with tool_calls
    assistant_idx = tool_start_idx - 1
    potential_assistant = messages[assistant_idx]
    
    if potential_assistant.get("role") != "assistant":
        return []
    
    tool_calls = potential_assistant.get("tool_calls", [])
    
    # Check for no tool_calls
    if not tool_calls:
        return []
    
    # Check for corrupted serialization (raw class dump with _type field)
    if isinstance(tool_calls[0], dict) and "_type" in tool_calls[0]:
        return []
    
    # Validate tool result IDs match tool_call IDs from assistant
    tool_call_ids = {tc.get("id") for tc in tool_calls if isinstance(tc, dict)}
    tool_result_ids = {
        messages[i].get("tool_call_id") 
        for i in range(tool_start_idx, tool_end_idx)
    }
    
    # Check if all tool result IDs are present in tool_call IDs
    if not tool_result_ids.issubset(tool_call_ids):
        return []
    
    # Success: extract the complete tool episode as an atomic unit
    return messages[assistant_idx:tool_end_idx]


def process_and_split_trace_user(messages: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Process and split the trace into user messages and rest of conversation.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Tuple of (user_messages, rest_of_conversation)
    """
    if not messages:
        return [], []
    
    user_messages = get_user_message(messages)
    
    if not user_messages:
        return [], messages
    
    # Collect all non-user messages
    rest_of_conversation = []
    for msg in messages:
        if msg.get("role") != "user":
            rest_of_conversation.append(msg)
    
    return user_messages, rest_of_conversation


def process_and_split_trace_user_tool(messages: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Process and split the trace into user messages, intermediate messages, and last tool episode.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Tuple of (user_messages, intermediate_messages, last_tool_episode)
    """
    if not messages:
        return [], [], []
    
    user_messages = get_user_message(messages)
    last_tool_episode = get_last_tool_interaction(messages)
    
    # Calculate where the last tool episode starts in the original messages
    if last_tool_episode:
        # Find the index where the tool episode starts
        tool_episode_start_idx = None
        for i in range(len(messages)):
            if messages[i] is last_tool_episode[0]:
                tool_episode_start_idx = i
                break
        
        if tool_episode_start_idx is not None:
            # Intermediate messages are everything before the tool episode, excluding user messages
            intermediate_messages = []
            for i in range(tool_episode_start_idx):
                msg = messages[i]
                if msg.get("role") != "user":
                    intermediate_messages.append(msg)
        else:
            # Shouldn't happen, but handle defensively
            intermediate_messages = []
            for msg in messages:
                if msg.get("role") != "user" and msg not in last_tool_episode:
                    intermediate_messages.append(msg)
    else:
        # No tool episode, so intermediate is all non-user messages
        intermediate_messages = []
        for msg in messages:
            if msg.get("role") != "user":
                intermediate_messages.append(msg)
    
    return user_messages, intermediate_messages, last_tool_episode