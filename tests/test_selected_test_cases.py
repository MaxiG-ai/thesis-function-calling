import pytest
import os
import tomllib
from src.utils.config import ExperimentConfig, load_configs


def test_config_accepts_selected_test_cases():
    """Test that ExperimentConfig accepts selected_test_cases field"""
    # Create minimal config with selected_test_cases
    config_data = {
        "experiment_name": "test_exp",
        "results_dir": "results",
        "log_dir": "logs",
        "logging_level": "INFO",
        "input_file": "test.jsonl",
        "enabled_models": ["model1"],
        "enabled_memory_methods": ["truncation"],
        "selected_test_cases": ["Car-Rental-0", "Car-Rental-1"],
        "memory_strategies": {
            "truncation": {"type": "truncation", "max_tokens": 10000}
        },
        "model_registry": {}
    }
    
    config = ExperimentConfig(**config_data)
    
    assert config.selected_test_cases == ["Car-Rental-0", "Car-Rental-1"]
    assert len(config.selected_test_cases) == 2


def test_config_optional_selected_test_cases():
    """Test that selected_test_cases is optional and defaults to None"""
    # Create minimal config without selected_test_cases
    config_data = {
        "experiment_name": "test_exp",
        "results_dir": "results",
        "log_dir": "logs",
        "logging_level": "INFO",
        "input_file": "test.jsonl",
        "enabled_models": ["model1"],
        "enabled_memory_methods": ["truncation"],
        "memory_strategies": {
            "truncation": {"type": "truncation", "max_tokens": 10000}
        },
        "model_registry": {}
    }
    
    config = ExperimentConfig(**config_data)
    
    assert config.selected_test_cases is None


def test_config_empty_selected_test_cases():
    """Test that selected_test_cases can be an empty list"""
    # Create minimal config with empty selected_test_cases
    config_data = {
        "experiment_name": "test_exp",
        "results_dir": "results",
        "log_dir": "logs",
        "logging_level": "INFO",
        "input_file": "test.jsonl",
        "enabled_models": ["model1"],
        "enabled_memory_methods": ["truncation"],
        "selected_test_cases": [],
        "memory_strategies": {
            "truncation": {"type": "truncation", "max_tokens": 10000}
        },
        "model_registry": {}
    }
    
    config = ExperimentConfig(**config_data)
    
    assert config.selected_test_cases == []
    assert len(config.selected_test_cases) == 0
