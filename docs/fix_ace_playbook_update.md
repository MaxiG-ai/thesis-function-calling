# Fix: ACE Playbook Not Updating Beyond Template

## Issue Summary

The ACE (Adaptive Contextual Experience) strategy's playbook is not updating beyond its initial template. The playbook remains static throughout all steps of task execution.

## Root Cause Analysis

The debug logging revealed **two cascading bugs** that prevent both the Reflector and Curator from running:

### Bug 1: Generator Returns Empty `bullet_ids_used`

**Location**: [generator.py](../src/strategies/ace/generator.py) - `_extract_bullet_ids()` method

**Problem**: The LLM returns bullet IDs as **strings** (e.g., `["TSD", "CTX", "RSN", "COM"]`) referencing section codes, but the extraction method only accepts **integer** IDs.

**Evidence from debug logs**:
```
Generator LLM response: "bullet_ids_used": ["TSD", "CTX", "RSN", "COM"]
Generator extracted bullet_ids: []  # Empty because "TSD" is not an integer
```

The extraction code at lines 93-97:
```python
if json_data and "bullet_ids_used" in json_data:
    ids = json_data["bullet_ids_used"]
    if isinstance(ids, list):
        return [int(i) for i in ids if isinstance(i, (int, str)) and str(i).isdigit()]
```

This only accepts numeric strings like `"1"`, `"2"`, etc. - not section codes like `"TSD"`, `"CTX-1"`.

**Impact**: `state.last_bullet_ids` is always `[]`, so Reflector condition `has_bullets=False` is always true.

### Bug 2: Reflector Gate Requires Non-Empty `bullet_ids`

**Location**: [ace_strategy.py](../src/strategies/ace/ace_strategy.py) - lines ~77-80

**Problem**: The Reflector only runs when **both** `last_reasoning_trace` AND `last_bullet_ids` are non-empty:

```python
if state.last_reasoning_trace and state.last_bullet_ids:  # Reflector runs
```

**Evidence from debug logs**:
```
Step 2:
State: last_reasoning_trace=<set>
State: last_bullet_ids=[]
Reflector conditions: has_reasoning=True, has_bullets=False
✗ Reflector SKIPPED (has_reasoning=True, has_bullets=False)
```

Even though `last_reasoning_trace` is set, `last_bullet_ids` is empty → Reflector never runs.

### Bug 3: Curator Blocked by Missing Reflection

**Location**: [ace_strategy.py](../src/strategies/ace/ace_strategy.py) - lines ~102-105

**Problem**: Since Reflector never runs, `state.last_reflection` is never set. The Curator requires `has_reflection=True`:

```python
if state.step_count % curator_frequency == 0 and state.last_reflection:  # Curator runs
```

**Evidence from debug logs**:
```
Curator conditions: step=2, frequency=1, frequency_match=True, has_reflection=False
✗ Curator SKIPPED (frequency_match=True, has_reflection=False)
```

## Cascade Effect

```
Empty Playbook Template → LLM uses section codes (TSD, CTX) as "bullet IDs"
                        ↓
Generator extracts [] (empty) because codes aren't numeric
                        ↓
state.last_bullet_ids = []
                        ↓
Reflector gate fails: has_bullets=False
                        ↓
Reflector NEVER runs → state.last_reflection = ""
                        ↓
Curator gate fails: has_reflection=False
                        ↓
Curator NEVER runs → Playbook NEVER updates
```

## Required Fixes

### Fix 1: Update Generator Prompt to Request Numeric IDs

**File**: [prompts/generator.prompt.md](../src/strategies/ace/prompts/generator.prompt.md)

Modify the prompt to clarify that bullets have numeric IDs and that section codes should NOT be used. If the playbook is empty, the Generator should return an empty list `[]` rather than section codes.

Example prompt update:
```markdown
## Response Format
- `bullet_ids_used` must be a list of **integer IDs** (e.g., [1, 3, 5])
- Each ID must correspond to an actual numbered bullet in the playbook
- If the playbook has no numbered bullets yet, return an empty list []
- Do NOT use section codes like "TSD", "CTX", etc. as bullet IDs
```

### Fix 2: Make Reflector Run on First Step Without Bullets

**File**: [ace_strategy.py](../src/strategies/ace/ace_strategy.py)

Option A - Allow Reflector to run without bullet IDs:
```python
# Change line ~77 from:
if state.last_reasoning_trace and state.last_bullet_ids:

# To:
if state.last_reasoning_trace:  # Run Reflector if we have reasoning trace
```

Option B - Bootstrap with initial reflection (preferred for ACE semantics):
```python
# On step 1, create an initial reflection to bootstrap the Curator
if state.step_count == 1:
    state.last_reflection = "Initial step - no performance data yet. Playbook should be initialized with basic task guidance."
```

### Fix 3: Handle Empty Playbook Gracefully

**File**: [ace_strategy.py](../src/strategies/ace/ace_strategy.py)

When the playbook is empty/template-only, the Curator should still be allowed to ADD initial bullets based on task observation:

```python
# Change Curator gate from:
if frequency_match and has_reflection:

# To:
playbook_is_empty = "<!-- " in state.playbook and state.playbook.count("- [") == 0
if frequency_match and (has_reflection or playbook_is_empty):
```

### Fix 4: Update Bullet ID Extraction for Flexible Formats

**File**: [generator.py](../src/strategies/ace/generator.py)

Update `_extract_bullet_ids()` to handle section-prefixed IDs like `"TSD-1"`, `"CTX-2"`:

```python
def _extract_bullet_ids(self, text: str) -> List[int]:
    # Try JSON extraction
    json_data = extract_json_from_text(text)
    if json_data and "bullet_ids_used" in json_data:
        ids = json_data["bullet_ids_used"]
        if isinstance(ids, list):
            result = []
            for i in ids:
                # Handle pure integers
                if isinstance(i, int):
                    result.append(i)
                elif isinstance(i, str):
                    # Handle "123" format
                    if i.isdigit():
                        result.append(int(i))
                    # Handle "TSD-1", "CTX-2" format - extract trailing number
                    elif "-" in i:
                        parts = i.split("-")
                        if parts[-1].isdigit():
                            result.append(int(parts[-1]))
            return result
    # ... rest of fallback logic
```

## Recommended Implementation Order

1. **Fix 1** (Generator prompt) - Prevents bad data from entering the system
2. **Fix 3** (Empty playbook handling) - Allows bootstrapping
3. **Fix 2** (Reflector gate) - Enables learning loop
4. **Fix 4** (Flexible ID extraction) - Handles edge cases

## Testing

After implementing fixes, run:
```bash
uv run python cfb_run_eval.py
```

Verify in DEBUG logs:
- [ ] Generator returns numeric bullet IDs (or empty list)
- [ ] Reflector runs from step 2 onwards
- [ ] Curator runs and logs "Applied N curator operations"
- [ ] Final playbook shows actual bullet points, not just section headers

## Related Files

- [ace_strategy.py](../src/strategies/ace/ace_strategy.py) - Main orchestration
- [generator.py](../src/strategies/ace/generator.py) - Generates actions, extracts bullet IDs
- [reflector.py](../src/strategies/ace/reflector.py) - Analyzes performance, tags bullets
- [curator.py](../src/strategies/ace/curator.py) - Updates playbook
- [prompts/generator.prompt.md](../src/strategies/ace/prompts/generator.prompt.md) - Generator prompt
- [playbook_utils.py](../src/strategies/ace/playbook_utils.py) - `EMPTY_PLAYBOOK_TEMPLATE`
