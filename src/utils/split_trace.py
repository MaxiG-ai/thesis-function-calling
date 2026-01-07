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
    """Process and split the trace into last user message and messages before it.
    
    This function finds the last user message and splits the conversation into:
    - Last user message (as a list with one element)
    - All messages before the last user message
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Tuple of ([last_user_message], messages_before_last_user)
        If no user message found, returns ([], all_messages)
    """
    if not messages:
        return [], []
    
    # Find the last user message
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break
    
    if last_user_idx is None:
        return [], messages
    
    last_user_message = [messages[last_user_idx]]
    messages_before = messages[:last_user_idx]
    
    return last_user_message, messages_before


def process_and_split_trace_user_tool(messages: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Process and split the trace into last user message, intermediate messages, and last tool episode.
    
    This function performs a 3-way split of the conversation:
    - Last user message (before the tool episode)
    - Intermediate messages (before the last user, excluding tool episode)
    - Last tool episode (assistant with tool_calls + tool responses at the end)
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Tuple of ([last_user_message], intermediate_messages, last_tool_episode)
        where:
        - last_user_message: List with the last user message (empty if not found)
        - intermediate_messages: All messages before the last user message
        - last_tool_episode: The last tool episode at the end (empty if not found)
    """
    if not messages:
        return [], [], []
    
    # First, extract the last tool episode
    last_tool_episode = get_last_tool_interaction(messages)
    
    # Determine where to split based on whether we have a tool episode
    if last_tool_episode:
        # Find where the tool episode starts in the original messages
        tool_episode_start_idx = None
        for i in range(len(messages)):
            if messages[i] is last_tool_episode[0]:
                tool_episode_start_idx = i
                break
        
        if tool_episode_start_idx is None:
            # Shouldn't happen, but handle defensively
            tool_episode_start_idx = len(messages)
        
        # Look for the last user message before the tool episode
        last_user_idx = None
        for i in range(tool_episode_start_idx - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break
        
        if last_user_idx is None:
            # No user found before tool episode
            return [], messages[:tool_episode_start_idx], last_tool_episode
        
        # Split: [intermediate] [last_user] [messages between user and tool] [tool_episode]
        # We want: [intermediate], [last_user], [tool_episode]
        # So intermediate = everything before last_user
        last_user_message = [messages[last_user_idx]]
        intermediate_messages = messages[:last_user_idx]
        
        return last_user_message, intermediate_messages, last_tool_episode
    else:
        # No tool episode, just find the last user and split around it
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                last_user_idx = i
                break
        
        if last_user_idx is None:
            # No user message at all
            return [], messages, []
        
        last_user_message = [messages[last_user_idx]]
        intermediate_messages = messages[:last_user_idx]
        
        return last_user_message, intermediate_messages, []