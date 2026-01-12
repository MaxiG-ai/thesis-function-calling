import tiktoken


def _normalize_model_name(model_name: str | None) -> str:
    if not model_name:
        return "gpt-4.1"
    # litellm-style names often look like "openai/gpt-4.1-mini"
    if "/" in model_name:
        model_name = model_name.split("/", 1)[1]
    return model_name


def _get_encoder(model_name: str | None):
    normalized = _normalize_model_name(model_name)
    try:
        return tiktoken.encoding_for_model(normalized)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


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
        parts.append(str(function_call))

    return parts

def get_token_count(count_obj, model_name: str | None = None) -> int:
    """Approximate token count for OpenAI-style chat payloads.

    Notes:
        - Uses a best-effort encoder selection; falls back to `cl100k_base`.
        - Counts `content` plus common tool-calling fields (`tool_calls[].function.arguments`)
          and benchmark-style `function_call` payloads.
    """
    enc = _get_encoder(model_name)
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