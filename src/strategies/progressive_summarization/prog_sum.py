from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import weave

from src.utils.logger import get_logger
from src.utils.token_count import get_token_count

logger = get_logger("ProgressiveSummarization")

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


class ProgressiveSummarizer:
    """Progressive Summarization strategy for managing conversation context.
    
    Summarizes archived context when token threshold is exceeded,
    keeping the last user query and tool episode intact.
    """
    
    def __init__(self, summary_prompt_path: Optional[str] = None):
        """Initialize the ProgressiveSummarizer with a summary prompt.
        
        Args:
            summary_prompt_path: Path to the summary prompt file. 
                               Defaults to "prompts/progressive_summary.md"
        """
        self.summary_prompt = self._load_summary_prompt(summary_prompt_path)
    
    def _load_summary_prompt(self, config_prompt_path: Optional[str] = None) -> str:
        """Load and cache the summary prompt.
        
        Args:
            config_prompt_path: Optional path to the summary prompt file
            
        Returns:
            The loaded summary prompt text
            
        Raises:
            FileNotFoundError: If the prompt file cannot be found
        """
        if config_prompt_path is None:
            config_prompt_path = "prompts/progressive_summary.md"
        
        # Resolve path relative to the src directory
        prompt_path = Path(__file__).resolve().parents[2] / Path(config_prompt_path)
        
        try:
            return prompt_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.error("Missing progressive summary prompt file at %s", prompt_path)
            raise
    
    @weave.op()
    def apply_summarization(
        self,
        messages: List[Dict],
        token_count: int,
        compact_threshold: int,
        llm_client: Any,
        summarizer_model: Optional[str] = None,
    ) -> List[Dict]:
        """Summarizes archived context when token threshold is exceeded.
        
        Re-summarizes all archived context (middle of conversation) each time,
        keeping the last user query and tool episode intact.
        
        Args:
            messages: List of message dictionaries from the conversation
            token_count: Current token count of the messages
            compact_threshold: Token threshold for triggering summarization
            llm_client: LLM client for generating summaries
            summarizer_model: Optional model name for summarization
            
        Returns:
            List of processed messages with summary if needed
            
        Raises:
            ValueError: If llm_client is None or summarization returns empty content
        """
        if llm_client is None:
            raise ValueError("llm_client is required for progressive summarization")
        
        user_query, conversation_history = split_llm_trace(messages)

        token_count = get_token_count(messages)

        # Don't summarize if below threshold or no archived content
        if token_count <= compact_threshold or not conversation_history:
            # Return: user query + tool episode (if present)
            result = []
            if user_query:
                result.append(user_query[0])
            return result + conversation_history
        
        # Build prompt for summarization
        prompt_messages = [ 
            {"role": "system", "content": self.summary_prompt},
            {"role": "user", "content": f"Conversation history to compress:\n{conversation_history}"},
        ]

        # Call LLM to generate summary (let exceptions propagate)
        model = summarizer_model or "gpt-4-1-mini"
        response = llm_client.generate_plain(
            input_messages=prompt_messages, model=model
        )

        # Extract summary text from response
        message = response.choices[0].message
        if isinstance(message, dict):
            summary_text = (message.get("content") or "").strip()
        else:
            summary_text = (getattr(message, "content", "") or "").strip()

        if not summary_text:
            raise ValueError("Summarization returned empty content")

        # Build final message list: user query + summary + tool episode
        summary_message = {"role": "system", "content": summary_text}
        
        result = []
        if user_query:
            result.append(user_query[0])
        result.append(summary_message)
        
        return result