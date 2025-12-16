import tiktoken

def get_token_count(messages: list[dict], model: str) -> int:
    """Utility to count tokens."""
    enc = tiktoken.encoding_for_model(model)
    count = 0
    for m in messages:
        content = m.get("content") or ""
        if isinstance(content, str):
            count += len(enc.encode(content))
    return count