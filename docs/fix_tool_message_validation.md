# Fix: Tool Message Validation for OpenAI API Compliance

## Problem
The benchmark was failing with:
```
BadRequestError: Error code: 400 - {'error': {'message': "Invalid parameter: messages with role 'tool' must be a response to a preceeding message with 'tool_calls'.", 'type': 'invalid_request_error', 'code': 'bad_request'}}
```

## Root Cause
The memory processing strategies (`truncation` and `memory_bank`) were breaking the required pairing between:
1. Assistant messages containing `tool_calls`
2. Their corresponding tool response messages with `role: "tool"`

When these strategies selected "recent messages" or reconstructed the message list, they could inadvertently:
- Keep a tool response message but discard its preceding assistant message with tool_calls
- Break the sequence required by the OpenAI API

## Solution Implemented

### 1. Message Pair Validation Function (`_validate_and_repair_tool_pairs`)
Added a validation function that:
- Scans through processed messages
- Detects orphaned tool messages (without preceding assistant+tool_calls)
- Searches the original message list to find the missing assistant message
- Injects the assistant message before the tool response
- Logs warnings if repair is needed

### 2. Enhanced Truncation Strategy
Modified `_apply_truncation` to:
- Start with the intended last N messages
- Check if the selection starts with a tool message
- Backtrack to include the complete assistant+tool_calls message
- Apply validation/repair as a safety net

### 3. Enhanced Memory Bank Strategy
Modified `_apply_memory_bank` to:
- Use the same backtracking logic when selecting working memory messages
- Ensure complete tool call sequences are included in recent_msgs
- Apply validation/repair as a safety net

### 4. Added Debug Logging
- Logs when assistant messages are injected to repair broken pairs
- Warns when tool messages must be skipped due to missing assistants
- Provides message count summaries for debugging

## Files Modified
- `src/memory_processing.py`: All changes implemented here

## Testing Recommendations
Run the benchmark with both memory strategies:
```bash
uv run python cfb_run_eval.py
```

Monitor logs for:
- `🔧 Injecting missing assistant message` - indicates repair was needed
- `⚠️ Could not find assistant message` - indicates a tool message was skipped
- No BadRequestError exceptions

## Future Considerations
If this error persists or becomes common:
1. Consider tracking tool call sequences as separate data structures
2. Implement a more sophisticated message selection algorithm
3. Add metrics to track how often repairs are needed
