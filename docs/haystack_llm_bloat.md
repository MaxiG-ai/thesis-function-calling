# Haystack Research Setup & LLM Call Bloat Analysis

*2026-03-02T15:11:40Z by Showboat 0.6.1*
<!-- showboat-id: 2c7644fc-306b-4424-ad93-3be40df92072 -->

## Overview

This document explains how the Needle-in-a-Haystack (NIAH) research setup works in this repository, and traces the LLM call bloat that occurs when the haystack context is not removed after step 1.

The project benchmarks five memory strategies (no_strategy, truncation, progressive_summarization, memory_bank, ace) against ComplexFuncBench under increasing amounts of injected distractor context (the 'haystack'). The hypothesis being tested: how well does each strategy cope as irrelevant context grows from 0 → 20k → 60k → 100k tokens?

### Architecture in one diagram

    cfb_run_eval.main()
     └─ run_model_configs()           [parallel per haystack threshold]
          └─ evaluate_single_case()
               └─ SAPGPTRunner.run()  [multi-turn tool-calling loop]
                    └─ FunctionCallSAPGPT.generate_response()
                         │  Step 1 only: prepend haystack_messages → self.messages
                         └─ LLMOrchestrator.generate_with_memory_applied(self.messages)
                              ├─ MemoryProcessor.apply_strategy()  → compressed_view
                              └─ litellm.completion(messages=compressed_view)

## Step 1: Haystack injection (occurs exactly once)

The haystack is a list of tool-call/response pairs sampled from *other domains* — purely distractor context. Each pre-generated .jsonl file embeds these into every benchmark case under the key `haystack_messages`.

The runner assigns them to the model before the first turn (`sap_gpt_runner.py:87`). Inside `generate_response` they are appended to `self.messages` on the very first call — and immediately set to None so they are never re-injected (`sap_gpt.py:71-75`).

The key point: `self.messages` (the LLM-facing buffer) now permanently contains the haystack for the rest of the task. The runner's local `messages` list (used for scoring) never contains the haystack — only `self.model.messages` does.

## The haystack data: structure and scale

Let's inspect what the haystack actually contains and how large it is across the configured thresholds.

```python3

import json, pathlib

data_dir = pathlib.Path('benchmarks/complex_func_bench/data')
thresholds = [20_000, 60_000, 100_000]

for t in thresholds:
    path = data_dir / f'haystack_{t}.jsonl'
    cases = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    token_counts = [c['haystack_token_count'] for c in cases]
    msg_counts   = [len(c['haystack_messages']) for c in cases]
    print(f'haystack_{t}.jsonl  →  {len(cases)} cases')
    print(f'  token_count : min={min(token_counts):,}  max={max(token_counts):,}  mean={sum(token_counts)//len(token_counts):,}')
    print(f'  msg_count   : min={min(msg_counts)}  max={max(msg_counts)}  mean={sum(msg_counts)//len(msg_counts)}')
    sample = cases[0]
    roles = [m['role'] for m in sample['haystack_messages']]
    print(f'  message roles (first case, first 8): {roles[:8]}')
    print()

```

```output
haystack_20000.jsonl  →  1000 cases
  token_count : min=88  max=61,811  mean=16,426
  msg_count   : min=2  max=51  mean=15
  message roles (first case, first 8): ['assistant', 'tool', 'assistant', 'tool', 'assistant', 'tool', 'tool', 'tool']

haystack_60000.jsonl  →  1000 cases
  token_count : min=4,842  max=65,993  mean=57,864
  msg_count   : min=3  max=123  mean=43
  message roles (first case, first 8): ['assistant', 'tool', 'tool', 'assistant', 'tool', 'assistant', 'tool', 'assistant']

haystack_100000.jsonl  →  1000 cases
  token_count : min=51,581  max=109,993  mean=100,410
  msg_count   : min=16  max=148  mean=70
  message roles (first case, first 8): ['assistant', 'tool', 'tool', 'assistant', 'tool', 'assistant', 'tool', 'tool']

```

The haystack messages are pure tool-call/response pairs (assistant + tool roles) from other domains. Notice that actual token counts vary significantly around the target — the 20k file has cases ranging from 88 to 61k tokens because some source domains produce short tool outputs.

## The LLM call bloat: what each strategy sees on turn 2+

After the first LLM call, `self.model.messages` grows as the benchmark loop appends the assistant response + tool observations (`sap_gpt_runner.py:114-173`). On every subsequent call the *entire* buffer — user query + haystack + all new turns — is passed into `LLMOrchestrator.generate_with_memory_applied()`.

Each strategy then receives this bloated input. The table below maps what each strategy does with it:

```python3

# Simulate what each strategy receives and sends to the LLM on turn N>1
# Input: user_query(1 msg) + haystack(~H msgs) + turn_1_assistant + tool_obs + turn_2_assistant + ...

strategies = {
    'no_strategy': {
        'what_it_does': 'Returns input unchanged',
        'haystack_in_output': True,
        'extra_llm_calls': 0,
        'output_formula': 'user + haystack + all_turns',
    },
    'truncation': {
        'what_it_does': 'Keeps user_query + last tool interaction ONLY (drops everything else incl. haystack)',
        'haystack_in_output': False,
        'extra_llm_calls': 0,
        'output_formula': 'user + last_tool_interaction',
    },
    'progressive_summarization': {
        'what_it_does': 'Splits into user_query vs history, summarizes history (incl. haystack) via 1 LLM call, returns [user, summary]',
        'haystack_in_output': 'As part of the summarization INPUT — the haystack is fed to the summarizer LLM in full',
        'extra_llm_calls': 1,
        'output_formula': 'user + summary_of(haystack + all_prior_turns)',
    },
    'memory_bank': {
        'what_it_does': 'Ingests new tool outputs, retrieves top-K. Output = user + retrieved_memory + last_tool_interaction',
        'haystack_in_output': 'Haystack tool outputs are ingested into vector store on step 1 — they may appear in retrieved_memory',
        'extra_llm_calls': '1 Observer call per new tool output for ingestion',
        'output_formula': 'user + retrieved_memory(top_k) + last_tool_interaction',
    },
    'ace': {
        'what_it_does': 'Runs Reflector + Curator + Generator (3 LLM calls), injects playbook + reasoning. Returns [playbook] + FULL messages + [reasoning]',
        'haystack_in_output': True,
        'extra_llm_calls': 3,
        'output_formula': 'playbook + user + haystack + all_turns + reasoning_trace',
    },
}

for name, info in strategies.items():
    print(f'=== {name} ===')
    print(f'  What it does      : {info["what_it_does"]}')
    print(f'  Haystack in output: {info["haystack_in_output"]}')
    print(f'  Extra LLM calls   : {info["extra_llm_calls"]}')
    print(f'  LLM input formula : {info["output_formula"]}')
    print()

```

```output
=== no_strategy ===
  What it does      : Returns input unchanged
  Haystack in output: True
  Extra LLM calls   : 0
  LLM input formula : user + haystack + all_turns

=== truncation ===
  What it does      : Keeps user_query + last tool interaction ONLY (drops everything else incl. haystack)
  Haystack in output: False
  Extra LLM calls   : 0
  LLM input formula : user + last_tool_interaction

=== progressive_summarization ===
  What it does      : Splits into user_query vs history, summarizes history (incl. haystack) via 1 LLM call, returns [user, summary]
  Haystack in output: As part of the summarization INPUT — the haystack is fed to the summarizer LLM in full
  Extra LLM calls   : 1
  LLM input formula : user + summary_of(haystack + all_prior_turns)

=== memory_bank ===
  What it does      : Ingests new tool outputs, retrieves top-K. Output = user + retrieved_memory + last_tool_interaction
  Haystack in output: Haystack tool outputs are ingested into vector store on step 1 — they may appear in retrieved_memory
  Extra LLM calls   : 1 Observer call per new tool output for ingestion
  LLM input formula : user + retrieved_memory(top_k) + last_tool_interaction

=== ace ===
  What it does      : Runs Reflector + Curator + Generator (3 LLM calls), injects playbook + reasoning. Returns [playbook] + FULL messages + [reasoning]
  Haystack in output: True
  Extra LLM calls   : 3
  LLM input formula : playbook + user + haystack + all_turns + reasoning_trace

```

**Key observation:** Two strategies — `no_strategy` and `ace` — pass the full haystack to the primary LLM on *every* turn after step 1. ACE is doubly expensive: it also runs 3 additional LLM calls (Reflector, Curator, Generator) each of which receives a subset of the bloated context. `progressive_summarization` feeds the haystack to the summarizer once per turn, then the main LLM only sees the (shorter) summary — but the summarizer call itself scales with haystack size.

The only strategies that genuinely escape haystack cost after step 1 are `truncation` (hard-drops all history) and `memory_bank` (replaces full history with top-K vector retrievals).

## Token cost per turn: concrete numbers

Let's estimate the token growth at each turn for a representative case across haystack sizes. A typical ComplexFuncBench case has ~5 turns with tool calls, each tool response adding ~200-500 tokens.

```python3

import json, pathlib

# Load a sample case from each haystack file and the base dataset
data_dir = pathlib.Path('benchmarks/complex_func_bench/data')
base_cases = {
    json.loads(l)['id']: json.loads(l)
    for l in (data_dir / 'ComplexFuncBench.jsonl').read_text().splitlines() if l.strip()
}

# Pick a medium-length case (5+ conversation turns)
sample_id = None
for cid, c in base_cases.items():
    if len(c['conversations']) >= 8:
        sample_id = cid
        break

base = base_cases[sample_id]
print(f'Sample case: {sample_id}')
print(f'  Conversation turns (ground truth): {len(base["conversations"])}')
print(f'  Available functions: {len(base["functions"])}')
print()

# Estimate base query tokens (first user message)
user_query = base['conversations'][0]['content']
print(f'  User query length (chars): {len(user_query)}')
print(f'  Estimated user query tokens (~4 chars/tok): {len(user_query)//4}')
print()

# Show haystack token counts for this case across thresholds
print('  Haystack overhead per threshold:')
for t in [20_000, 60_000, 100_000]:
    path = data_dir / f'haystack_{t}.jsonl'
    hs_cases = {json.loads(l)['id']: json.loads(l) for l in path.read_text().splitlines() if l.strip()}
    if sample_id in hs_cases:
        hs = hs_cases[sample_id]
        print(f'    haystack_{t}: {hs["haystack_token_count"]:,} tokens  ({len(hs["haystack_messages"])} messages)')
    else:
        print(f'    haystack_{t}: case not found')

```

```output
Sample case: Car-Rental-40
  Conversation turns (ground truth): 8
  Available functions: 5

  User query length (chars): 252
  Estimated user query tokens (~4 chars/tok): 63

  Haystack overhead per threshold:
    haystack_20000: 19,289 tokens  (14 messages)
    haystack_60000: 51,535 tokens  (27 messages)
    haystack_100000: 105,941 tokens  (97 messages)
```

```python3

# Model the LLM input token count per turn for each strategy
# Using Car-Rental-40 as the representative case

import json, pathlib

data_dir = pathlib.Path('benchmarks/complex_func_bench/data')
sample_id = 'Car-Rental-40'
base = json.loads(next(l for l in (data_dir / 'ComplexFuncBench.jsonl').read_text().splitlines() if sample_id in l))

# Estimate real tool turn sizes from conversations
TOOL_CALL_TOKENS   = 80   # assistant message with tool_call JSON
TOOL_RESULT_TOKENS = 400  # typical tool observation
QUERY_TOKENS = len(base['conversations'][0]['content']) // 4

def tokens_in_buffer_at_turn(query_t, haystack_t, turn_n):
    '''Total tokens in self.model.messages at the start of turn N (1-indexed).
    Turn 1: query + haystack  (first call, nothing appended yet)
    Turn 2+: query + haystack + (turn_n-1) * (call_tokens + result_tokens)
    '''
    base_tokens = query_t + haystack_t
    history_tokens = (turn_n - 1) * (TOOL_CALL_TOKENS + TOOL_RESULT_TOKENS)
    return base_tokens + history_tokens

haystack_sizes = {
    'baseline (0)': 0,
    'haystack_20k': 19_289,
    'haystack_60k': 51_535,
    'haystack_100k': 105_941,
}

strategies = {
    'no_strategy'     : lambda total: total,
    'truncation'      : lambda total: QUERY_TOKENS + TOOL_CALL_TOKENS + TOOL_RESULT_TOKENS,  # drops to query + last interaction
    'prog_summ (main)': lambda total: QUERY_TOKENS + min(total // 4, 2048),  # summary ~1/4 of input, capped
    'prog_summ (summ_LLM)': lambda total: total,  # summarizer sees the full buffer
    'memory_bank'     : lambda total: QUERY_TOKENS + 3 * 600 + TOOL_CALL_TOKENS + TOOL_RESULT_TOKENS,  # top_k=3 retrievals
    'ace (main LLM)'  : lambda total: total + 4096 + 800,  # +playbook + reasoning injected
    'ace (each sub)': lambda total: 300 + 800,  # Reflector/Curator/Generator use short context slices
}

print(f'LLM input tokens per strategy at each turn (sample case: {sample_id}, query={QUERY_TOKENS} tok)')
print(f'  Assumed: each tool turn adds {TOOL_CALL_TOKENS} (call) + {TOOL_RESULT_TOKENS} (result) tokens')
print()

for hs_label, hs_tok in haystack_sizes.items():
    print(f'--- {hs_label} ---')
    print(f'  {"Strategy":<28}  Turn1    Turn3    Turn5    Turn8')
    for strat, fn in strategies.items():
        row = []
        for turn in [1, 3, 5, 8]:
            total = tokens_in_buffer_at_turn(QUERY_TOKENS, hs_tok, turn)
            row.append(fn(total))
        print(f'  {strat:<28}  {row[0]:>6,}   {row[1]:>6,}   {row[2]:>6,}   {row[3]:>6,}')
    print()

```

```output
LLM input tokens per strategy at each turn (sample case: Car-Rental-40, query=63 tok)
  Assumed: each tool turn adds 80 (call) + 400 (result) tokens

--- baseline (0) ---
  Strategy                      Turn1    Turn3    Turn5    Turn8
  no_strategy                       63    1,023    1,983    3,423
  truncation                       543      543      543      543
  prog_summ (main)                  78      318      558      918
  prog_summ (summ_LLM)              63    1,023    1,983    3,423
  memory_bank                    2,343    2,343    2,343    2,343
  ace (main LLM)                 4,959    5,919    6,879    8,319
  ace (each sub)                 1,100    1,100    1,100    1,100

--- haystack_20k ---
  Strategy                      Turn1    Turn3    Turn5    Turn8
  no_strategy                   19,352   20,312   21,272   22,712
  truncation                       543      543      543      543
  prog_summ (main)               2,111    2,111    2,111    2,111
  prog_summ (summ_LLM)          19,352   20,312   21,272   22,712
  memory_bank                    2,343    2,343    2,343    2,343
  ace (main LLM)                24,248   25,208   26,168   27,608
  ace (each sub)                 1,100    1,100    1,100    1,100

--- haystack_60k ---
  Strategy                      Turn1    Turn3    Turn5    Turn8
  no_strategy                   51,598   52,558   53,518   54,958
  truncation                       543      543      543      543
  prog_summ (main)               2,111    2,111    2,111    2,111
  prog_summ (summ_LLM)          51,598   52,558   53,518   54,958
  memory_bank                    2,343    2,343    2,343    2,343
  ace (main LLM)                56,494   57,454   58,414   59,854
  ace (each sub)                 1,100    1,100    1,100    1,100

--- haystack_100k ---
  Strategy                      Turn1    Turn3    Turn5    Turn8
  no_strategy                   106,004   106,964   107,924   109,364
  truncation                       543      543      543      543
  prog_summ (main)               2,111    2,111    2,111    2,111
  prog_summ (summ_LLM)          106,004   106,964   107,924   109,364
  memory_bank                    2,343    2,343    2,343    2,343
  ace (main LLM)                110,900   111,860   112,820   114,260
  ace (each sub)                 1,100    1,100    1,100    1,100

```

These numbers expose the bloat precisely:

- **no_strategy**: every LLM call scales linearly with haystack size. At 100k haystack, turn 8 = ~109k tokens per call — entirely dominated by irrelevant distractor content.
- **ace (main LLM)**: even worse — adds ~4-5k tokens *on top* of the full buffer for playbook + reasoning injection. At 100k haystack, every main call is ~114k tokens.
- **prog_summ (summ_LLM)**: the summarizer call receives the entire buffer including haystack, scaling to 106k tokens at the 100k threshold. The main LLM is protected (only ~2k), but you pay for one expensive summarizer call per turn.
- **truncation**: fully immune — drops to ~543 tokens regardless of haystack size.
- **memory_bank**: fully immune — only the top-k retrieved records + last interaction reach the LLM (~2k tokens).

The cost amplifier for `ace` and `no_strategy` is not just per-token billing — at 100k+ tokens, many models hit their context window limit, potentially causing hard failures or degraded quality.

## Total LLM calls per benchmark case

The number of LLM calls is not just about context size — some strategies multiply the call count.

```python3

def total_calls(strategy, turns):
    if strategy in ('no_strategy', 'truncation'):
        return turns
    elif strategy == 'progressive_summarization':
        return turns * 2  # 1 main + 1 summarizer per turn
    elif strategy == 'memory_bank':
        return turns * 2  # 1 main + 1 observer per turn
    elif strategy == 'ace':
        # step1: Curator(1) + Generator(1) + main(1) = 3
        # step2+: Reflector(1) + Curator(1) + Generator(1) + main(1) = 4
        return 3 + (turns - 1) * 4

header = 'Strategy'
print('Total LLM calls per case (across all turns):')
print(f'  {header:<32}  N=3    N=5    N=8    N=12')
for strat in ['no_strategy', 'truncation', 'progressive_summarization', 'memory_bank', 'ace']:
    row = [total_calls(strat, n) for n in [3, 5, 8, 12]]
    print(f'  {strat:<32}  {row[0]:>4}   {row[1]:>4}   {row[2]:>4}   {row[3]:>4}')

print()
print('Relative call multiplier vs no_strategy at N=8:')
base = total_calls('no_strategy', 8)
for strat in ['truncation', 'progressive_summarization', 'memory_bank', 'ace']:
    mult = total_calls(strat, 8) / base
    print(f'  {strat:<32}  {mult:.1f}x')

```

```output
Total LLM calls per case (across all turns):
  Strategy                          N=3    N=5    N=8    N=12
  no_strategy                          3      5      8     12
  truncation                           3      5      8     12
  progressive_summarization            6     10     16     24
  memory_bank                          6     10     16     24
  ace                                 11     19     31     47

Relative call multiplier vs no_strategy at N=8:
  truncation                        1.0x
  progressive_summarization         2.0x
  memory_bank                       2.0x
  ace                               3.9x
```

ACE makes nearly 4x more LLM calls than no_strategy — and each of those main calls carries the full haystack on top. This compounds multiplicatively: at 100k haystack with N=8 turns, ACE produces 31 calls of which 8 carry ~114k tokens each.

## Where the bloat lives in the code

Pinpointing the exact lines responsible for the observed behaviour:

| Concern | File | Lines | What happens |
|---------|------|-------|--------------|
| Haystack injected once into self.messages | sap_gpt.py | 71-75 | `self.messages = messages + haystack_messages`; haystack_messages set to None |
| self.messages grows each turn | sap_gpt_runner.py | 114-173 | Assistant + tool results appended to self.model.messages |
| Full self.messages passed to orchestrator every turn | sap_gpt.py | 79-84 | `orchestrator.generate_with_memory_applied(input_messages=self.messages)` |
| no_strategy returns input unchanged | memory_processing.py | 92-96 | Haystack stays in output |
| ace returns full messages + playbook + reasoning | ace_strategy.py | 239 | `[playbook_message] + messages + [reasoning_message]` |
| prog_summ feeds full buffer to summarizer | prog_sum.py | 48-65 | `process_and_split_trace_user(messages)` then `llm_client.generate_plain(prompt_messages)` |
| truncation drops to query + last interaction | truncation.py (via split_trace) | - | Only user query and last tool pair survive |
| memory_bank replaces history with top-K | memory_bank_strategy.py | 195 | `user_query_msgs + memory_content + last_tool_msgs` |

## Summary: the bloat problem by strategy

The root cause is simple: `self.model.messages` (in `FunctionCallSAPGPT`) accumulates the entire conversation including the haystack, and is passed verbatim to `generate_with_memory_applied` on every turn. Each memory strategy then decides what to do with it.

**Strategies that are immune to haystack bloat after step 1:**

- `truncation` — hard-discards everything except user query + last tool interaction. The haystack is gone from turn 2 onward.
- `memory_bank` — replaces the entire buffer with user query + top-K vector retrievals + last interaction. Constant token cost regardless of haystack size. (Note: haystack tool outputs *are* ingested on step 1 and may pollute the vector store with distractor facts.)

**Strategies that pay full haystack cost on every turn:**

- `no_strategy` — passes `self.messages` unchanged; the haystack rides along for every LLM call.
- `ace` — prepends playbook + appends reasoning trace to the *full* `self.messages`; main LLM sees haystack + overhead. 3 additional sub-agent calls per turn (though those do NOT see the full buffer).

**Strategies with partial protection:**

- `progressive_summarization` — the *main* LLM call is protected (sees summary only), but the summarizer LLM call receives the entire buffer including the haystack on each turn where the threshold is exceeded. Cost = O(haystack) per turn for the summarizer.

**Design implication:** For the NIAH experiment to fairly stress-test compression, strategies like `no_strategy` and `ace` are working as intended — they serve as the 'bloated baseline'. But if the goal is to reduce LLM costs, `truncation` and `memory_bank` decouple main-LLM cost from haystack size, while `progressive_summarization` shifts but does not eliminate the cost.
