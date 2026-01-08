from typing import List, Dict, Tuple
from src.utils.logger import get_logger
from src.utils.split_trace import process_and_split_trace_user_tool
from src.utils.token_count import get_token_count

logger = get_logger("TruncationStrategy")

def truncate_messages(
    messages: List[Dict],
) -> List[Dict]:
    """ Truncation strategy that splits messages into user query, conversation history, and tool interaction.
        Keeps the last user query and the last tool interaction intact,
        truncating the conversation history in between as needed.
    """

    user_query, conversation_history, tool_interaction = process_and_split_trace_user_tool(messages)
    logger.debug(f"""🧠 Truncation Strategy: 
                User Query Tokens: {get_token_count(user_query)},
                Conversation History Tokens: {get_token_count(conversation_history)}, 
                Tool Interaction Tokens: {get_token_count(tool_interaction)}"""
                 )
    return user_query + tool_interaction