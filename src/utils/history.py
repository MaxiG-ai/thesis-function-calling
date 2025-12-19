from typing import Dict, List, Optional, Tuple


def segment_message_history(messages: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split conversation into pinned nodes, archived context, and working memory."""
    prefix, start_idx = _extract_pinned_prefix(messages)
    conversation = messages[start_idx:]
    tail_start = _find_tail_start(conversation)

    return prefix, conversation[:tail_start], conversation[tail_start:]


def _extract_pinned_prefix(messages: List[Dict]) -> tuple[List[Dict], int]:
    prefix: List[Dict] = []
    idx = 0

    while idx < len(messages):
        msg = messages[idx]
        if msg.get("role") == "system" or msg.get("summary_marker"):
            prefix.append(msg)
            idx += 1
            continue
        break

    return prefix, idx


def _find_tail_start(conversation: List[Dict]) -> int:
    if not conversation:
        return 0

    last_user_idx = _find_last_role_index(conversation, "user")
    if last_user_idx is not None:
        return last_user_idx

    fallback_idx = max(0, len(conversation) - 1)
    return _collapse_tool_sequence(conversation, fallback_idx)


def _find_last_role_index(conversation: List[Dict], role: str) -> Optional[int]:
    for idx in range(len(conversation) - 1, -1, -1):
        if conversation[idx].get("role") == role:
            return idx
    return None


def _collapse_tool_sequence(conversation: List[Dict], idx: int) -> int:
    while idx > 0 and conversation[idx].get("role") == "tool":
        idx -= 1
    return idx
