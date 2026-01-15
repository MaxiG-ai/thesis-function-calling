import tiktoken
from src.utils.logger import get_logger

logger = get_logger("TokenCounter")


def _iter_message_text_parts(message: dict) -> list[str]:
    parts: list[str] = []

    content = message.get("content")
    if isinstance(content, str) and content:
        parts.append(content)

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function")
            if isinstance(func, dict):
                name = func.get("name")
                if isinstance(name, str) and name:
                    parts.append(name)
                arguments = func.get("arguments")
                if isinstance(arguments, str) and arguments:
                    parts.append(arguments)

    function_call = message.get("function_call")
    if function_call is not None:
        # if found necessary
        # TODO: Extract actual content from function_call instead of using str() to get accurate token counts
        parts.append(str(function_call))

    return parts

def get_token_count(count_obj, model_name: str | None = None) -> int:
    """Approximate token count for OpenAI-style chat payloads.

    Notes:
        - Uses a best-effort encoder selection; falls back to `cl100k_base`.
        - Counts `content` plus common tool-calling fields (`tool_calls[].function.arguments`)
          and benchmark-style `function_call` payloads.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    count = 0
    # count tokens in a list of messages
    if isinstance(count_obj, list):
        for m in count_obj:
            if not isinstance(m, dict):
                continue
            for text in _iter_message_text_parts(m):
                count += len(enc.encode(text))
    # count tokens in a single message (dict)
    elif isinstance(count_obj, dict):
        for text in _iter_message_text_parts(count_obj):
            count += len(enc.encode(text))

    return count