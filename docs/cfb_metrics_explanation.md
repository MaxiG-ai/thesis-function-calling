# ComplexFuncBench Dataset Metrics Detailed Explanation

*2026-03-03T12:48:14Z by Showboat 0.6.1*
<!-- showboat-id: f2c16957-b465-45a5-bf83-35eaf3f2a75c -->

## Overview

ComplexFuncBench is a benchmark designed to evaluate complex function calling capabilities of Large Language Models (LLMs). The evaluation framework, called ComplexEval, measures both function calling ability and response generation quality across 1,000 test cases spanning five domains: Flights, Hotels, Car-Rental, Attraction, and Cross-domain.

The benchmark evaluates five key aspects:

1. Multi-step function calling in single turn
2. Function calling with user-provided constraints
3. Parameter value reasoning from implicit information
4. Long parameter values (exceeding 500 tokens)
5. Long-context handling (up to 128k tokens)

## Evaluation Metrics

ComplexFuncBench computes seven primary metrics organized into three categories:

### 1. Function Calling Metrics

These metrics evaluate the accuracy of function calls made by the model compared to the ground truth.

```python
# Overall Call Accuracy (overall_call_acc) - Example Calculation

# This metric represents the percentage of correctly generated function calls
# across all test cases and all function calls within each case.

# Example data from a hypothetical run:
domain_call_count = {
    'Flights': [45, 50],    # [correct_calls, total_calls] for Flights domain
    'Hotels': [38, 45],     # 38 correct out of 45 total calls
    'Car-Rental': [42, 48],
    'Attraction': [40, 47],
    'Cross': [155, 210]     # Cross-domain cases (more calls)
}

# Calculate overall call accuracy
total_correct_calls = sum([v[0] for v in domain_call_count.values()])
total_calls = sum([v[1] for v in domain_call_count.values()])
overall_call_acc = total_correct_calls / total_calls * 100 if total_calls > 0 else 0

print(f'Total Correct Calls: {total_correct_calls}')
print(f'Total Calls: {total_calls}')
print(f'Overall Call Accuracy: {overall_call_acc:.2f}%')

# Per-domain call accuracy
domain_call_acc = {
    k: v[0] / v[1] * 100 if v[1] != 0 else 0 
    for k, v in domain_call_count.items()
}
print(f'\nPer-domain Call Accuracy:')
for domain, acc in domain_call_acc.items():
    print(f'  {domain}: {acc:.2f}%')
```

```output
Total Correct Calls: 320
Total Calls: 400
Overall Call Accuracy: 80.00%

Per-domain Call Accuracy:
  Flights: 90.00%
  Hotels: 84.44%
  Car-Rental: 87.50%
  Attraction: 85.11%
  Cross: 73.81%
```

#### How Function Call Accuracy is Determined

Each function call is evaluated using a hierarchical comparison strategy in :

1. **Rule-based matching**: Exact string comparison of function name and all parameters
2. **Similarity-based matching**: Uses BGE-large-en-v1.5 embeddings with 0.98 similarity threshold
3. **Response-based matching**: Calls the actual API and compares responses
4. **LLM-based matching**: Uses GPT-4 to judge semantic equivalence

A function call is considered correct if it passes any of these checks.

### 2. Success Rate Metrics

These metrics measure whether the model successfully completed the entire conversation.

```python
# Overall Success Rate (overall_success) - Example Calculation

# This metric represents the percentage of test cases where ALL function calls
# in ALL turns were correct. A single incorrect call causes the entire case to fail.

# Example data from a hypothetical run with 1000 test cases:
# - 150 cases each for: Flights, Hotels, Car-Rental, Attraction
# - 400 cases for Cross-domain

domain_success = {
    'Flights': 95,      # 95 out of 150 cases fully successful
    'Hotels': 88,       # 88 out of 150 cases fully successful
    'Car-Rental': 92,
    'Attraction': 85,
    'Cross': 250        # 250 out of 400 cases fully successful
}

total_cases = 150 * 4 + 400  # 1000 total cases
total_successes = sum(domain_success.values())
overall_success = total_successes / total_cases * 100

print(f'Total Successful Cases: {total_successes}')
print(f'Total Cases: {total_cases}')
print(f'Overall Success Rate: {overall_success:.2f}%')

# Per-domain success rates
domain_success_rate = {
    k: v / 150 * 100 if k != 'Cross' else v / 400 * 100
    for k, v in domain_success.items()
}
print(f'\nPer-domain Success Rate:')
for domain, rate in domain_success_rate.items():
    cases = 150 if domain != 'Cross' else 400
    print(f'  {domain}: {rate:.2f}% ({domain_success[domain]}/{cases} cases)')
```

```output
Total Successful Cases: 610
Total Cases: 1000
Overall Success Rate: 61.00%

Per-domain Success Rate:
  Flights: 63.33% (95/150 cases)
  Hotels: 58.67% (88/150 cases)
  Car-Rental: 61.33% (92/150 cases)
  Attraction: 56.67% (85/150 cases)
  Cross: 62.50% (250/400 cases)
```

#### Key Difference: Success Rate vs Call Accuracy

- **Success Rate**: Binary per-case metric (all-or-nothing). Even one wrong function call in a multi-turn conversation marks the entire case as failed.
- **Call Accuracy**: Granular per-call metric. Measures what percentage of individual function calls were correct, regardless of case-level success.

Example: A case with 10 function calls where 9 are correct:

- Contributes 0% to success rate (case failed)
- Contributes 90% to call accuracy (9/10 calls correct)

### 3. Response Quality Metrics

These metrics evaluate the quality of the natural language response generated after executing function calls.

```python
# Completeness Score (complete_score_avg) - Example Calculation

# Evaluates whether the response addresses all parts of the user's query.
# Scoring: 0 (incomplete), 1 (partial), 2 (complete)

# Example data from evaluation:
complete_score_count = {
    'Flights': [250, 140],      # [sum_of_scores, num_evaluated] = avg 1.79
    'Hotels': [235, 135],        # avg 1.74
    'Car-Rental': [248, 138],    # avg 1.80
    'Attraction': [230, 137],    # avg 1.68
    'Cross': [680, 360]          # avg 1.89
}

complete_score_sum = sum([v[0] for v in complete_score_count.values()])
complete_score_total = sum([v[1] for v in complete_score_count.values()])
complete_score_avg = complete_score_sum / complete_score_total if complete_score_total > 0 else 0

print(f'Total Completeness Score: {complete_score_sum}')
print(f'Total Cases Evaluated: {complete_score_total}')
print(f'Average Completeness Score: {complete_score_avg:.2f} / 2.0')

# Per-domain averages
print(f'\nPer-domain Completeness:')
for domain, (score_sum, count) in complete_score_count.items():
    avg = score_sum / count if count > 0 else 0
    print(f'  {domain}: {avg:.2f} / 2.0')
```

```output
Total Completeness Score: 1643
Total Cases Evaluated: 910
Average Completeness Score: 1.81 / 2.0

Per-domain Completeness:
  Flights: 1.79 / 2.0
  Hotels: 1.74 / 2.0
  Car-Rental: 1.80 / 2.0
  Attraction: 1.68 / 2.0
  Cross: 1.89 / 2.0
```
