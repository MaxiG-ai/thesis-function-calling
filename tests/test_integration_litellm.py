"""
Integration tests for litellm model availability and completions.

These tests verify that each model defined in model_config.toml:
1. Is properly recognized by litellm
2. Can successfully execute litellm.completion() calls
3. Returns valid ChatCompletion responses

Each test is minimal, using a simple system+user message pair to verify
basic functionality similar to how the LLMOrchestrator uses the models.
"""

import pytest
import litellm
from src.utils.config import load_configs


@pytest.fixture(scope="module")
def config():
    """Load the experiment and model configuration for the test suite."""
    return load_configs("config.toml", "model_config.toml")


def test_gpt_5_available(config):
    """Test that gpt-5 is available in litellm and can run completion.

    Verifies that the OpenAI/AICore gpt-5 model is properly configured
    and responds to litellm.completion() calls with valid responses.
    """
    model_def = config.model_registry["gpt-5"]

    response = litellm.completion(
        model=model_def.litellm_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        api_base=model_def.api_base,
        api_key=model_def.api_key,
        drop_params=True,
    )

    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert response.choices[0].message.content is not None


def test_gpt_5_mini_available(config):
    """Test that gpt-5-mini is available in litellm and can run completion.

    Verifies that the OpenAI/AICore gpt-5-mini model is properly configured
    and responds to litellm.completion() calls with valid responses.
    """
    model_def = config.model_registry["gpt-5-mini"]

    response = litellm.completion(
        model=model_def.litellm_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        api_base=model_def.api_base,
        api_key=model_def.api_key,
        drop_params=True,
    )

    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert response.choices[0].message.content is not None


def test_gpt_4_1_available(config):
    """Test that gpt-4.1 is available in litellm and can run completion.

    Verifies that the OpenAI/AICore gpt-4.1 model is properly configured
    and responds to litellm.completion() calls with valid responses.
    """
    model_def = config.model_registry["gpt-4-1"]
    model_kwargs = {}
    if model_def.temperature is not None:
        model_kwargs["temperature"] = model_def.temperature

    response = litellm.completion(
        model=model_def.litellm_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        api_base=model_def.api_base,
        api_key=model_def.api_key,
        drop_params=True,
        **model_kwargs,
    )

    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert response.choices[0].message.content is not None


def test_gpt_4_1_mini_available(config):
    """Test that gpt-4.1-mini is available in litellm and can run completion.

    Verifies that the OpenAI/AICore gpt-4.1-mini model is properly configured
    and responds to litellm.completion() calls with valid responses.
    """
    model_def = config.model_registry["gpt-4-1-mini"]
    model_kwargs = {}
    if model_def.temperature is not None:
        model_kwargs["temperature"] = model_def.temperature

    response = litellm.completion(
        model=model_def.litellm_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        api_base=model_def.api_base,
        api_key=model_def.api_key,
        drop_params=True,
        **model_kwargs,
    )

    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert response.choices[0].message.content is not None


def test_claude_sonnet_4_5_available(config):
    """Test that claude-sonnet-4-5 is available in litellm and can run completion.

    Verifies that the Bedrock/AICore claude-sonnet-4-5 model is properly configured
    and responds to litellm.completion() calls with valid responses.
    """
    model_def = config.model_registry["claude-sonnet-4-5"]
    model_kwargs = {}
    if model_def.temperature is not None:
        model_kwargs["temperature"] = model_def.temperature

    response = litellm.completion(
        model=model_def.litellm_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello."},
        ],
        api_base=model_def.api_base,
        api_key=model_def.api_key,
        drop_params=True,
        **model_kwargs,
    )

    assert response is not None
    assert hasattr(response, "choices")
    assert len(response.choices) > 0
    assert response.choices[0].message.content is not None