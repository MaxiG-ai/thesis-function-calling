from typing import List, Dict

def detect_tail_loop(messages: List[Dict], threshold: int = 4, max_pattern_len: int = 10) -> bool:
    """Detects repeating patterns at the tail of a conversation."""
    n = len(messages)
    # Optimization: Don't check if history is too short to contain a loop
    if n < threshold:
        return False

    # 1. Normalize messages for comparison
    # We must exclude fields that change every turn even in a loop (like tool_call_id)
    normalized = []
    for m in messages[-(max_pattern_len * threshold):]: # Only look at the relevant tail
        # Create a signature tuple: (Role, Content, Sorted Tool Calls)
        tool_sig = None
        if "tool_calls" in m:
            tool_sig = sorted(
                [(tc["type"], tc["function"]["name"], tc["function"]["arguments"]) for tc in m["tool_calls"]]
            )
            tool_sig = tuple(tool_sig)

        normalized.append((m.get("role"), m.get("content"), tool_sig))

    # Re-calculate length based on the slice we actually took
    n_slice = len(normalized)

    # 2. Check for patterns of length L
    # We iterate L from 1 (repeating single message) up to max_pattern_len
    for L in range(1, max_pattern_len + 1):
        # We need at least L * threshold messages to verify this pattern
        if n_slice < L * threshold:
            break
            
        # The "candidate" pattern is the very last L messages
        pattern = normalized[-L:]
        
        # Check if this pattern appears 'threshold' times backwards
        is_loop = True

        for k in range(1, threshold):
            # Compare the block before the current one
            # e.g., if L=2, threshold=3:
            # Check [-2:] vs [-4:-2]
            # Check [-2:] vs [-6:-4]
            prev_block = normalized[-(k + 1) * L : -k * L]
            if prev_block != pattern:
                is_loop = False
                break

        if is_loop:
            return True

    return False