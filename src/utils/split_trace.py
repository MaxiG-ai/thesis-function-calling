# contains utilities to split llm traces
from typing import List, Dict, Tuple


def get_user_message(messages: List[Dict]) -> Tuple[List[Dict], List[int]]:
    """Get user message(s) from a list of messages.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        List of user messages (can be empty if no user messages found)
    """
    if not messages:
        return [], []
    
    user_messages = []
    user_messages_idx = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            user_messages.append(msg)
            user_messages_idx.append(i)
    
    return user_messages, user_messages_idx 


def get_last_tool_interaction(messages: List[Dict]) -> Tuple[List[Dict], int]:
    """Get the last valid tool interaction from a list of messages.
    
    Searches backwards through messages to find the last valid tool episode
    (assistant message with tool_calls + all corresponding tool response messages).
    Skips corrupted tool_calls (those with _type field from bad serialization).
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Tuple of (tool_episode, start_index) where:
        - tool_episode: List containing [assistant_msg, tool_msg1, ...], or empty list
        - start_index: Index where the tool episode starts, or len(messages) if not found
    """
    if not messages:
        return [], len(messages)
    
    # Find all indices where tool messages end (followed by non-tool or end of list)
    i = len(messages) - 1
    while i >= 0:
        # Skip until we find a tool message
        while i >= 0 and messages[i].get("role") != "tool":
            i -= 1
        
        if i < 0:
            break
        
        # Found end of a potential tool sequence, find its start
        tool_end_idx = i + 1
        while i >= 0 and messages[i].get("role") == "tool":
            i -= 1
        tool_start_idx = i + 1
        
        # Need at least one message before the tool messages (the assistant)
        if tool_start_idx == 0:
            continue
        
        # Check the message before tool messages - should be assistant with tool_calls
        assistant_idx = tool_start_idx - 1
        potential_assistant = messages[assistant_idx]
        
        if potential_assistant.get("role") != "assistant":
            continue
        
        tool_calls = potential_assistant.get("tool_calls", [])
        
        # Check for no tool_calls
        if not tool_calls:
            continue
        
        # Check for corrupted serialization (raw class dump with _type field)
        if isinstance(tool_calls[0], dict) and "_type" in tool_calls[0]:
            continue
        
        # Validate tool result IDs match tool_call IDs from assistant
        tool_call_ids = {tc.get("id") for tc in tool_calls if isinstance(tc, dict)}
        tool_result_ids = {
            messages[j].get("tool_call_id") 
            for j in range(tool_start_idx, tool_end_idx)
        }
        
        # Check if all tool result IDs are present in tool_call IDs
        if not tool_result_ids.issubset(tool_call_ids):
            continue
        
        # Success: found a valid tool episode
        return messages[assistant_idx:tool_end_idx], assistant_idx
    
    return [], len(messages)


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
    """Process and split the trace into first user message, intermediate messages, and last tool episode.
    
    This function performs a 3-way split of the conversation:
    - First user message (the initial query)
    - Intermediate messages (between first user and the tool episode)
    - Last tool episode (assistant with tool_calls + tool responses)
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Tuple of ([first_user_message], intermediate_messages, last_tool_episode)
        where:
        - first_user_message: List with the first user message (empty if not found)
        - intermediate_messages: All messages between first user and tool episode
        - last_tool_episode: The last valid tool episode (empty if not found)
    """
    if not messages:
        return [], [], []
    
    # Get all user messages and use the first one
    user_messages, user_messages_idx = get_user_message(messages)
    # if not user_messages:
    #     last_tool_episode, tool_episode_start_idx = get_last_tool_interaction(messages)
    #     return [], messages[:tool_episode_start_idx], last_tool_episode
    
    # Extract the last valid tool episode and its start index
    last_tool_episode, tool_episode_start_idx = get_last_tool_interaction(messages)
    
    # Intermediate messages are between first user and the tool episode start
    intermediate_start = user_messages_idx[0] + 1
    intermediate_end = tool_episode_start_idx if last_tool_episode else len(messages)
    intermediate_messages = messages[intermediate_start:intermediate_end]
    
    return user_messages, intermediate_messages, last_tool_episode