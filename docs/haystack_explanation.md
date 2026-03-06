# Haystack Data Generation Process

*2026-03-03T13:30:28Z by Showboat 0.6.1*
<!-- showboat-id: 9f62c357-75e1-469a-8a2d-6ba4e1b35b1b -->

The haystack benchmark evaluates an LLM's ability to locate relevant function calls within a large context window. The data generation process creates synthetic test scenarios by embedding target test cases (needles) within distractor tool interaction messages (haystack) sampled from other domains in ComplexFuncBench.

## Data Generation Pipeline

The generation process operates on ComplexFuncBench, which contains five domains: Flights, Hotels, Car-Rental, Attraction, and Cross. For each test case:

**Step 1: Domain isolation**
- Extract the domain of the target test case
- Build a donor pool from all other domains (cross-validation style)

**Step 2: Haystack construction**
- Sample tool interactions from the donor pool
- Convert to OpenAI-format message pairs (assistant with tool_calls + tool responses)
- Add interactions until target token count is reached (±10% tolerance)
- Use deterministic seeding (case_id + threshold) for reproducibility

**Step 3: Output generation**
- Preserve original conversations and functions
- Add haystack_messages and haystack_token_count fields
- Export to JSONL format per target threshold

## Technical Implementation

**Data source:**
- ComplexFuncBench.jsonl (5 domains: Flights, Hotels, Car-Rental, Attraction, Cross)

**Cross-validation approach:**
- Distractor interactions sampled only from domains different than the test case
- Ensures haystack context is semantically unrelated to the target task

**Token counting:**
- Uses tiktoken cl100k_base encoder (GPT-4/4o/3.5-turbo compatible)
- Counts tokens in message content, tool call names, and arguments

**Output schema:**
- Original fields: id, conversations, functions
- New fields: haystack_messages (list of OpenAI-format messages), haystack_token_count (int)

## Context Length Statistics

```python

import json
from pathlib import Path

# Find haystack dataset files
data_dir = Path('benchmarks/complex_func_bench/data')
haystack_files = sorted(data_dir.glob('haystack_*.jsonl'))

stats = []

for jsonl_file in haystack_files:
    # Extract target threshold from filename
    threshold = jsonl_file.stem.split('_')[1]
    context_lengths = []
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            context_lengths.append(data['haystack_token_count'])
    
    if context_lengths:
        stats.append({
            'dataset': f'haystack_{threshold}',
            'min': min(context_lengths),
            'max': max(context_lengths),
            'avg': sum(context_lengths) / len(context_lengths),
            'count': len(context_lengths)
        })

# Print table
print('| Dataset | Min Tokens | Max Tokens | Avg Tokens | Test Cases |')
print('|---------|-----------|-----------|-----------|------------|')
for s in stats:
    print(f"| {s['dataset']} | {s['min']:,} | {s['max']:,} | {s['avg']:,.0f} | {s['count']} |")

```

```output
| Dataset | Min Tokens | Max Tokens | Avg Tokens | Test Cases |
|---------|-----------|-----------|-----------|------------|
| haystack_100000 | 90,000 | 109,966 | 95,017 | 1000 |
| haystack_20000 | 18,002 | 21,998 | 19,545 | 1000 |
| haystack_40000 | 36,002 | 43,990 | 38,698 | 1000 |
| haystack_60000 | 54,000 | 65,932 | 57,812 | 1000 |
| haystack_80000 | 72,015 | 87,762 | 76,458 | 1000 |
```

The table shows updated context length statistics measured in tokens for each target threshold. Dataset names indicate the target context length with 10% tolerance (e.g., haystack_20000 targets 18k-22k tokens). The tighter clustering of min/max values compared to previous versions reflects improvements in the haystack generation algorithm's consistency.



---

This approach systematically evaluates model performance degradation as context length increases, providing insights into long-context function calling capabilities while maintaining semantic separation between target tasks and distractor context.
