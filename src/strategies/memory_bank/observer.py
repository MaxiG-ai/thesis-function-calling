"""
Observer LLM - Generates natural language summaries of tool outputs.

The Observer creates semantic summaries that enable vector-based retrieval
while the raw data is preserved in the FactStore for precision.
"""

import json
from pathlib import Path
from typing import Any, Dict

from src.utils.llm_helpers import extract_content
from src.utils.logger import get_logger

logger = get_logger("Observer")

# Maximum characters for raw output before truncation
MAX_OUTPUT_CHARS = 10000


def _load_observer_prompt() -> str:
    """Load the observer prompt template from file."""
    prompt_path = Path(__file__).parent / "observer.prompt.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    # Fallback inline prompt if file doesn't exist
    return """You are an Observer that summarizes tool execution results.

Given the user's task and a tool's output, summarize what was achieved.
Highlight key entities (IDs, coordinates, names, status codes) that may 
be needed later, but do not output the full JSON.

Keep the summary under 200 words."""


def observe_tool_output(
    user_query: str,
    tool_name: str,
    raw_output: Dict[str, Any],
    llm_client: Any,
    model: str,
) -> str:
    """
    Generate a natural language summary of a tool's output.

    Args:
        user_query: The user's original task description
        tool_name: Name of the tool that was executed
        raw_output: The raw JSON output from the tool
        llm_client: LLM client with generate_plain() method
        model: Model identifier for the observer LLM

    Returns:
        Natural language summary highlighting key entities
    """
    # Serialize and truncate raw output
    output_str = json.dumps(raw_output, ensure_ascii=False, indent=2)
    if len(output_str) > MAX_OUTPUT_CHARS:
        output_str = output_str[:MAX_OUTPUT_CHARS] + "\n... [truncated]"
        logger.debug(
            f"Truncated tool output from {len(output_str)} to {MAX_OUTPUT_CHARS} chars"
        )

    system_prompt = _load_observer_prompt()

    user_content = f"""User's Task: {user_query}

Tool Called: {tool_name}

Tool Output:
{output_str}

Summarize what this tool execution achieved."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    response = llm_client.generate_plain(input_messages=messages, model=model)
    summary = extract_content(response)

    if not summary:
        logger.warning(f"Observer returned empty summary for tool {tool_name}")
        summary = f"Tool '{tool_name}' was executed."

    return summary
