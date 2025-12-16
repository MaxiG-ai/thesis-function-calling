# Feature: Selective Test Case Execution

## Overview

This feature allows users to run the ComplexFuncBench benchmark on specific test cases by selecting them by their test case IDs. This provides a targeted way to test individual cases without processing the entire dataset.

## Implementation

### Configuration Schema

Added `selected_test_cases` field to `ExperimentConfig` in `src/utils/config.py`:
- Type: `Optional[List[str]]`
- Default: `None` (disabled)
- When set, contains a list of test case IDs to run

### Dataset Filtering Logic

Modified `cfb_run_eval.py` to filter the dataset before processing:
1. If `selected_test_cases` is configured, filter the dataset to only include matching test cases
2. If no matches are found, log an error and exit
3. If `selected_test_cases` is not configured, fall back to existing behavior (random sampling or full dataset)

### Priority

When `selected_test_cases` is set, it takes priority over `benchmark_sample_size`. This ensures minimal overhead by only processing the explicitly selected test cases.

## Usage

### Basic Usage

Edit `config.toml` and add the `selected_test_cases` option:

```toml
# Run specific test cases by ID
selected_test_cases = ["Car-Rental-0", "Car-Rental-1", "Travel-5"]
```

### Example Configuration

See `config_example_selected_tests.toml` for a complete example configuration.

### Running the Benchmark

```bash
python cfb_run_eval.py
```

The benchmark will automatically filter to only the selected test cases.

## Benefits

1. **Targeted Testing**: Run specific test cases for debugging or validation
2. **Minimal Overhead**: Only selected test cases are loaded and processed
3. **Fast Iteration**: Quickly test individual cases without waiting for full benchmark
4. **Backward Compatible**: Feature is optional and defaults to existing behavior

## Testing

### Unit Tests

- `tests/test_selected_test_cases.py`: Tests configuration schema
- `tests/test_dataset_filtering.py`: Tests filtering logic

### Demonstration

Run the demonstration script to see the feature in action:

```bash
python demo_selected_test_cases.py
```

This script shows:
- How the configuration is loaded
- How the dataset is filtered
- The minimal overhead achieved
- Comparison between selected cases and default behavior

## Test Case ID Format

Test case IDs follow the format: `{Domain}-{Number}`

Examples:
- `Car-Rental-0`
- `Car-Rental-1`
- `Travel-5`
- `Cross-10`

You can find all available test case IDs by examining:
```bash
jq -r '.id' benchmarks/complex_func_bench/data/ComplexFuncBench.jsonl
```

## Edge Cases

1. **No matches found**: If none of the selected IDs match any test cases, an error is logged and execution stops
2. **Empty list**: If `selected_test_cases = []`, it is treated as disabled
3. **None/Not set**: If not configured, falls back to existing behavior

## Security

✅ No security vulnerabilities detected by CodeQL scan

## Files Changed

- `src/utils/config.py`: Added `selected_test_cases` field to configuration schema
- `cfb_run_eval.py`: Added filtering logic
- `config.toml`: Added documentation comment
- `config_example_selected_tests.toml`: Example configuration (new file)
- `README.md`: Added documentation section
- `tests/test_selected_test_cases.py`: Configuration tests (new file)
- `tests/test_dataset_filtering.py`: Filtering logic tests (new file)
- `demo_selected_test_cases.py`: Demonstration script (new file)
- `FEATURE_SUMMARY.md`: This summary document (new file)
