"""
Helper utilities for extracting content from LLM responses.

Centralizes the common pattern of extracting text content from ChatCompletion
responses, handling both dict-style and object-style message formats.
"""

from typing import Any


def extract_content(response: Any) -> str:
    """
    Extract text content from a ChatCompletion response.

    Handles both dict-style and object-style message formats that can be
    returned by different LLM providers via LiteLLM.

    Args:
        response: ChatCompletion response object with choices[0].message

    Returns:
        Extracted text content, stripped of whitespace. Returns empty string
        if content is None or missing.

    Example:
        response = llm_client.generate_plain(input_messages=messages)
        text = extract_content(response)
    """
    message = response.choices[0].message
    if isinstance(message, dict):
        return (message.get("content") or "").strip()
    return (getattr(message, "content", "") or "").strip()
