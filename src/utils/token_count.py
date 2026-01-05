import tiktoken

def get_token_count(count_obj) -> int:
    """Utility to count tokens."""
    enc = tiktoken.encoding_for_model("gpt-4.1")
    count = 0
    # count tokens in a list of messages
    if isinstance(count_obj, list):
        for m in count_obj:
            content = m.get("content") or ""
            if isinstance(content, str):
                count += len(enc.encode(content))
    # count tokens in a single message (dict)
    elif isinstance(count_obj, dict):
        content = count_obj.get("content") or ""
        if isinstance(content, str):
            count += len(enc.encode(content))

    return count