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
    
    Iterates through messages to find tool interactions following the pattern:
    Assistant (with tool_calls) -> One or more Tool messages.
    
    Args:
        messages: List of message dictionaries
        
    Returns:
        Tuple of (tool_episode, start_index) where:
        - tool_episode: List containing [assistant_msg, tool_msg1, ...], or empty list
        - start_index: Index where the tool episode starts, or len(messages) if not found
    """
    if not messages:
        return [], len(messages)

    last_interaction = ([], len(messages))
    i = 0
    
    while i < len(messages):
        msg = messages[i]
        
        # Check for assistant message with tool_calls
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            current_interaction = [msg]
            start_idx = i
            
            # Look ahead for tool messages
            j = i + 1
            while j < len(messages) and messages[j].get("role") == "tool":
                current_interaction.append(messages[j])
                j += 1
            
            # If we found at least one tool response, this is a valid interaction
            if len(current_interaction) > 1:
                last_interaction = (current_interaction, start_idx)
            
            # Advance main pointer
            i = j
        else:
            i += 1
            
    return last_interaction


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
    
    user_messages, user_message_idx = get_user_message(messages)
    
    if not user_messages:
        return [], messages

    conv_after_user_message = messages[user_message_idx[-1] + 1 :]
    
    return [user_messages[-1]], conv_after_user_message


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
    
    # Extract the last valid tool episode and its start index
    last_tool_episode, tool_episode_start_idx = get_last_tool_interaction(messages)
    
    if not user_messages:
        return [], messages[:tool_episode_start_idx], last_tool_episode

    # Intermediate messages are between first user and the tool episode start
    intermediate_start = user_messages_idx[0] + 1
    intermediate_end = tool_episode_start_idx if last_tool_episode else len(messages)
    intermediate_messages = messages[intermediate_start:intermediate_end]
    
    return [user_messages[0]], intermediate_messages, last_tool_episode