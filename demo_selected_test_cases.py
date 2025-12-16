#!/usr/bin/env python3
"""
Demonstration script showing how the selected_test_cases feature works.

This script simulates the filtering logic without running the full benchmark.
It shows:
1. How to configure selected_test_cases in config.toml
2. How the filtering is applied to the dataset
3. The minimal overhead approach (only selected cases are loaded)
"""

import json
import tomllib
from pathlib import Path


def load_dataset(path: str):
    """Load the ComplexFuncBench dataset"""
    dataset = []
    with open(path, 'r') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


def simulate_filtering(config_path: str, dataset_path: str):
    """Simulate the dataset filtering logic from cfb_run_eval.py"""
    
    print("=" * 80)
    print("Demonstration: selected_test_cases Feature")
    print("=" * 80)
    print()
    
    # Load configuration
    print(f"📄 Loading configuration from: {config_path}")
    with open(config_path, 'rb') as f:
        config = tomllib.load(f)
    
    experiment_name = config.get('experiment_name', 'unknown')
    selected_test_cases = config.get('selected_test_cases')
    benchmark_sample_size = config.get('benchmark_sample_size')
    
    print(f"   Experiment: {experiment_name}")
    print(f"   selected_test_cases: {selected_test_cases}")
    print(f"   benchmark_sample_size: {benchmark_sample_size}")
    print()
    
    # Load dataset
    print(f"📦 Loading dataset from: {dataset_path}")
    dataset = load_dataset(dataset_path)
    print(f"   Total test cases in dataset: {len(dataset)}")
    print()
    
    # Apply filtering logic (same as in cfb_run_eval.py)
    if selected_test_cases is not None and len(selected_test_cases) > 0:
        print("🎯 Filtering dataset by selected test case IDs...")
        filtered_dataset = [case for case in dataset if case.get('id') in selected_test_cases]
        
        if not filtered_dataset:
            print(f"   ❌ No test cases found matching: {selected_test_cases}")
            return
        
        print(f"   ✅ Filtered to {len(filtered_dataset)} test case(s)")
        print(f"   Selected IDs: {selected_test_cases}")
        print(f"   Matched IDs: {[case['id'] for case in filtered_dataset]}")
        print()
        print(f"   💡 Overhead: Only {len(filtered_dataset)} cases loaded and processed")
        print(f"   💡 benchmark_sample_size={benchmark_sample_size} is IGNORED when selected_test_cases is set")
        
    else:
        print("📊 No specific test cases selected")
        if benchmark_sample_size and benchmark_sample_size > 0:
            print(f"   Using random sampling with size: {benchmark_sample_size}")
            filtered_dataset = dataset[:benchmark_sample_size]  # Simplified for demo
        else:
            print(f"   Using full dataset: {len(dataset)} test cases")
            filtered_dataset = dataset
    
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Configuration: {config_path}")
    print(f"Total dataset size: {len(dataset)}")
    print(f"Test cases to run: {len(filtered_dataset) if filtered_dataset else 0}")
    
    if selected_test_cases:
        print(f"Feature enabled: YES")
        print(f"Selected cases: {selected_test_cases}")
    else:
        print(f"Feature enabled: NO")
        print(f"Using: {'Random sampling' if benchmark_sample_size else 'Full dataset'}")
    
    print("=" * 80)
    print()


if __name__ == "__main__":
    # Set paths
    base_path = Path(__file__).parent
    dataset_path = base_path / "benchmarks/complex_func_bench/data/ComplexFuncBench.jsonl"
    
    print("\n" + "=" * 80)
    print("SCENARIO 1: Using selected_test_cases")
    print("=" * 80)
    config_path = base_path / "config_example_selected_tests.toml"
    if config_path.exists() and dataset_path.exists():
        simulate_filtering(str(config_path), str(dataset_path))
    else:
        print(f"⚠️  Files not found: {config_path} or {dataset_path}")
    
    print("\n" + "=" * 80)
    print("SCENARIO 2: Using default config (no selected_test_cases)")
    print("=" * 80)
    config_path = base_path / "config.toml"
    if config_path.exists() and dataset_path.exists():
        simulate_filtering(str(config_path), str(dataset_path))
    else:
        print(f"⚠️  Files not found: {config_path} or {dataset_path}")
