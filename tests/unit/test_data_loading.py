"""Tests for robust error extraction from task `message` payloads.

These tests validate that error metadata is extracted consistently from supported
shapes and that unsupported or partial payloads fail safely.
"""

from __future__ import annotations

import pytest

from tools.helpers.data_loading import extract_error_message


def test_extract_error_message_from_single_error_dict() -> None:
    """Extract error data when `message` is a single dictionary.

    This test verifies the direct and simplest valid shape:
    `{"message": {"error_type": str, "content": str}}`.

    Expected behavior:
    - return the same `error_type` string unchanged
    - return the same `content` string unchanged
    - do not aggregate or transform values in this mode
    """
    task = {
        "message": {
            "error_type": "decode_error",
            "content": "Function(arguments='{\"date\": \"2024-12-15\"}', name='SearchFlightsMultiStopsLegs') is not Valid.",
        }
    }

    error_type, error_reasoning = extract_error_message(task)

    assert error_type == "decode_error"
    assert (
        error_reasoning
        == "Function(arguments='{\"date\": \"2024-12-15\"}', name='SearchFlightsMultiStopsLegs') is not Valid."
    )


def test_extract_error_message_from_list_aggregates_counts_and_contents() -> None:
    """Aggregate repeated list errors into count-string and joined explanations.

    This test verifies the list shape:
    `{"message": [{"error_type": str, "content": str}, ...]}`.

    Expected behavior:
    - count repeated `error_type` entries and format as `<count> <error_type>`
    - preserve first-seen error-type order when rendering summaries
    - concatenate `content` values using `"; "` in original list order
    """
    task = {
        "message": [
            {
                "error_type": "value_error",
                "content": "Parameter arrival_date value is not correct in prediction.",
            },
            {
                "error_type": "value_error",
                "content": "Parameter arrival_date value is not correct in prediction.",
            },
        ]
    }

    error_type, error_reasoning = extract_error_message(task)

    assert error_type == "2 value_error"
    assert (
        error_reasoning
        == "Parameter arrival_date value is not correct in prediction.; Parameter arrival_date value is not correct in prediction."
    )


def test_extract_error_message_from_mixed_list_ignores_invalid_items() -> None:
    """Ignore malformed list entries while still aggregating valid error records.

    This test intentionally includes non-dict entries and wrong-typed fields in
    the message list to ensure robust type checking and fail safety.

    Expected behavior:
    - skip invalid entries without raising exceptions
    - aggregate only valid string-based `error_type` and `content` values
    - keep first-seen error-type order in the summary
    """
    task = {
        "message": [
            {"error_type": "value_error", "content": "A"},
            {"error_type": "decode_error", "content": "B"},
            {"error_type": "value_error", "content": "C"},
            "not-a-dict",
            {"error_type": 123, "content": None},
        ]
    }

    error_type, error_reasoning = extract_error_message(task)

    assert error_type == "2 value_error; 1 decode_error"
    assert error_reasoning == "A; B; C"


@pytest.mark.parametrize(
    "task",
    [
        None,
        {},
        {"message": "Success"},
        {"message": []},
        {"message": {"content": "missing error_type"}},
    ],
)
def test_extract_error_message_returns_default_for_unsupported_shapes(task: object) -> None:
    """Return default sentinel text when payload shape is unsupported.

    This test covers invalid top-level types, missing keys, and message formats
    that do not match the two supported schemas.

    Expected behavior:
    - always return the exact fallback string for both outputs:
      `"no error detected"`
    - never raise due to malformed inputs
    """
    error_type, error_reasoning = extract_error_message(task)

    assert error_type == "no error detected"
    assert error_reasoning == "no error detected"
