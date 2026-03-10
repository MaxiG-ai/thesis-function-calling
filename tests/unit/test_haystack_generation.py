"""
Tests for the needle-in-a-haystack data generation module.

These tests verify the haystack generation pipeline that creates benchmark
data files with injected distractor context. The haystack data is sampled
from other domains (cross-validation style) and formatted as OpenAI-compatible
tool interactions (assistant with tool_calls + tool responses).

Test categories:
1. Domain mapping  - cross-validation exclusion logic
2. Format validity - OpenAI chat format for synthetic tool interactions
3. Token targeting - generated haystack hits the token budget within tolerance
4. Determinism     - identical seeds produce identical outputs
5. Integration     - end-to-end generation of a haystack case
"""

import json
import re
import uuid
import pytest
from unittest.mock import patch

from benchmarks.complex_func_bench.data.generate_haystack import (
    DOMAINS,
    get_donor_domains,
    extract_domain,
    build_donor_pool,
    extract_tool_interactions,
    convert_interaction_to_openai_format,
    generate_haystack_for_case,
    replace_invalid_chars,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_case_flights():
    """A minimal Flights-domain test case with one tool interaction."""
    return {
        "id": "Flights-42",
        "conversations": [
            {"role": "user", "content": "Find me a flight from NYC to LAX"},
            {
                "role": "assistant",
                "function_call": [
                    {
                        "name": "Search_Flight_Location",
                        "arguments": {"query": "New York"},
                    }
                ],
            },
            {
                "role": "observation",
                "content": [
                    {"status": True, "data": [{"id": "NYC", "name": "New York"}]}
                ],
            },
            {"role": "assistant", "content": "I found flights for you."},
        ],
        "functions": [
            {
                "name": "Search_Flight_Location",
                "description": "Search for flight locations",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ],
    }


@pytest.fixture
def sample_case_hotels():
    """A minimal Hotels-domain test case with one tool interaction."""
    return {
        "id": "Hotels-10",
        "conversations": [
            {"role": "user", "content": "Find hotels in Paris"},
            {
                "role": "assistant",
                "function_call": [
                    {
                        "name": "Search_Hotel_Destination",
                        "arguments": {"query": "Paris"},
                    }
                ],
            },
            {
                "role": "observation",
                "content": [{"status": True, "data": [{"id": "PAR", "name": "Paris"}]}],
            },
            {"role": "assistant", "content": "Here are Paris hotels."},
        ],
        "functions": [
            {
                "name": "Search_Hotel_Destination",
                "description": "Search hotel destinations",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ],
    }


@pytest.fixture
def sample_case_car_rental():
    """A minimal Car-Rental case with two parallel tool calls in one turn."""
    return {
        "id": "Car-Rental-5",
        "conversations": [
            {"role": "user", "content": "Rent a car in San Diego"},
            {
                "role": "assistant",
                "function_call": [
                    {
                        "name": "Search_Car_Location",
                        "arguments": {"query": "San Diego"},
                    },
                    {
                        "name": "Search_Car_Location",
                        "arguments": {"query": "Los Angeles"},
                    },
                ],
            },
            {
                "role": "observation",
                "content": [
                    {"status": True, "data": [{"id": "SAN"}]},
                    {"status": True, "data": [{"id": "LAX"}]},
                ],
            },
            {"role": "assistant", "content": "Found car rentals."},
        ],
        "functions": [
            {
                "name": "Search_Car_Location",
                "description": "Search car locations",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ],
    }


@pytest.fixture
def sample_dataset(sample_case_flights, sample_case_hotels, sample_case_car_rental):
    """A small dataset spanning multiple domains for testing donor pool logic."""
    attraction_case = {
        "id": "Attraction-0",
        "conversations": [
            {"role": "user", "content": "Find attractions in Rome"},
            {
                "role": "assistant",
                "function_call": [
                    {
                        "name": "Search_Attraction_Location",
                        "arguments": {"query": "Rome"},
                    }
                ],
            },
            {
                "role": "observation",
                "content": [{"status": True, "data": [{"id": "ROM"}]}],
            },
            {"role": "assistant", "content": "Attractions found."},
        ],
        "functions": [
            {
                "name": "Search_Attraction_Location",
                "description": "Search attractions",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ],
    }
    cross_case = {
        "id": "Cross-0",
        "conversations": [
            {"role": "user", "content": "Plan a trip"},
            {
                "role": "assistant",
                "function_call": [
                    {
                        "name": "Search_Hotel_Destination",
                        "arguments": {"query": "Tokyo"},
                    },
                    {
                        "name": "Search_Attraction_Location",
                        "arguments": {"query": "Tokyo"},
                    },
                ],
            },
            {
                "role": "observation",
                "content": [
                    {"status": True, "data": [{"id": "TYO"}]},
                    {"status": True, "data": [{"id": "TYO_A"}]},
                ],
            },
            {"role": "assistant", "content": "Trip planned."},
        ],
        "functions": [
            {
                "name": "Search_Hotel_Destination",
                "description": "Search hotels",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "Search_Attraction_Location",
                "description": "Search attractions",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        ],
    }
    return [
        sample_case_flights,
        sample_case_hotels,
        sample_case_car_rental,
        attraction_case,
        cross_case,
    ]


# ---------------------------------------------------------------------------
# 1. Domain mapping
# ---------------------------------------------------------------------------


def test_extract_domain_single_word():
    """
    extract_domain must parse the domain prefix from case IDs. Single-word
    domains like 'Flights-42' should return 'Flights'.
    """
    assert extract_domain("Flights-42") == "Flights"
    assert extract_domain("Hotels-0") == "Hotels"
    assert extract_domain("Attraction-149") == "Attraction"
    assert extract_domain("Cross-399") == "Cross"


def test_extract_domain_hyphenated():
    """
    'Car-Rental-131' has a hyphen inside the domain name. extract_domain
    must use rsplit to correctly return 'Car-Rental', not 'Car'.
    """
    assert extract_domain("Car-Rental-131") == "Car-Rental"
    assert extract_domain("Car-Rental-0") == "Car-Rental"


def test_get_donor_domains_excludes_own():
    """
    Cross-validation logic: get_donor_domains returns all domains EXCEPT
    the case's own domain. For a Flights case, the donor pool must include
    Hotels, Car-Rental, Attraction, and Cross but NOT Flights.
    """
    donors = get_donor_domains("Flights")
    assert "Flights" not in donors
    assert set(donors) == {"Hotels", "Car-Rental", "Attraction", "Cross"}


def test_get_donor_domains_cross():
    """
    Cross-domain cases use all four single-domain categories as donors.
    Cross should NOT appear in its own donor set.
    """
    donors = get_donor_domains("Cross")
    assert "Cross" not in donors
    assert set(donors) == {"Flights", "Hotels", "Car-Rental", "Attraction"}


def test_get_donor_domains_all_domains_covered():
    """
    Every domain in DOMAINS must produce a valid donor set containing
    exactly len(DOMAINS)-1 entries, and the case's own domain must never
    appear in its donor set.
    """
    for domain in DOMAINS:
        donors = get_donor_domains(domain)
        assert len(donors) == len(DOMAINS) - 1
        assert domain not in donors


# ---------------------------------------------------------------------------
# 2. Format validity
# ---------------------------------------------------------------------------


def test_convert_interaction_to_openai_format_single_call():
    """
    A single-call tool interaction (one function_call + one observation)
    must produce exactly 2 OpenAI messages: an assistant message with
    tool_calls and a tool response message. The tool_call_id must match
    between assistant and tool messages, and function names must be sanitized.
    """
    interaction = {
        "function_call": [
            {"name": "Search_Hotel_Destination", "arguments": {"query": "Paris"}}
        ],
        "observation": [{"status": True, "data": [{"id": "PAR"}]}],
    }
    messages = convert_interaction_to_openai_format(interaction, prefix="test")

    assert len(messages) == 2
    # Assistant message
    assert messages[0]["role"] == "assistant"
    assert messages[0]["content"] is None
    assert len(messages[0]["tool_calls"]) == 1
    tc = messages[0]["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["id"].startswith("test_")
    assert tc["function"]["name"] == replace_invalid_chars("Search_Hotel_Destination")
    assert json.loads(tc["function"]["arguments"]) == {"query": "Paris"}

    # Tool response
    assert messages[1]["role"] == "tool"
    assert messages[1]["tool_call_id"] == tc["id"]


def test_convert_interaction_to_openai_format_parallel_calls():
    """
    A parallel-call interaction (two function_calls + two observations)
    must produce 3 messages: 1 assistant with 2 tool_calls, then 2 tool
    response messages. Each tool response must reference the correct
    tool_call_id from the assistant message.
    """
    interaction = {
        "function_call": [
            {"name": "Search_Car_Location", "arguments": {"query": "San Diego"}},
            {"name": "Search_Car_Location", "arguments": {"query": "LA"}},
        ],
        "observation": [
            {"status": True, "data": [{"id": "SAN"}]},
            {"status": True, "data": [{"id": "LAX"}]},
        ],
    }
    messages = convert_interaction_to_openai_format(interaction, prefix="p")

    # 1 assistant + 2 tool = 3 messages
    assert len(messages) == 3
    assert messages[0]["role"] == "assistant"
    assert len(messages[0]["tool_calls"]) == 2
    assert messages[1]["role"] == "tool"
    assert messages[2]["role"] == "tool"

    # tool_call_id linkage
    assert messages[1]["tool_call_id"] == messages[0]["tool_calls"][0]["id"]
    assert messages[2]["tool_call_id"] == messages[0]["tool_calls"][1]["id"]


def test_replace_invalid_chars():
    """
    Function names must be sanitized to match OpenAI's [a-zA-Z0-9_-] pattern
    and truncated to 64 characters, consistent with the runner's logic.
    """
    assert replace_invalid_chars("Search_Hotels") == "Search_Hotels"
    assert replace_invalid_chars("Get_Hotel_Reviews(Tips)") == "Get_Hotel_Reviews-Tips-"
    # Length truncation
    long_name = "A" * 100
    assert len(replace_invalid_chars(long_name)) == 64


# ---------------------------------------------------------------------------
# 3. Tool interaction extraction
# ---------------------------------------------------------------------------


def test_extract_tool_interactions(sample_case_flights):
    """
    extract_tool_interactions must walk through a case's conversation
    and return a list of {function_call, observation} dicts, one per
    assistant-turn that contains function calls.
    """
    interactions = extract_tool_interactions(sample_case_flights)
    assert len(interactions) == 1
    assert interactions[0]["function_call"] == [
        {"name": "Search_Flight_Location", "arguments": {"query": "New York"}}
    ]
    assert interactions[0]["observation"] == [
        {"status": True, "data": [{"id": "NYC", "name": "New York"}]}
    ]


def test_extract_tool_interactions_parallel(sample_case_car_rental):
    """
    Cases with parallel tool calls (multiple function_calls in one assistant
    turn) must produce a single interaction with both calls and both
    observations preserved in order.
    """
    interactions = extract_tool_interactions(sample_case_car_rental)
    assert len(interactions) == 1
    assert len(interactions[0]["function_call"]) == 2
    assert len(interactions[0]["observation"]) == 2


# ---------------------------------------------------------------------------
# 4. Donor pool construction
# ---------------------------------------------------------------------------


def test_build_donor_pool(sample_dataset):
    """
    build_donor_pool must return only tool interactions from domains that
    are NOT the target domain. For a Flights case, the pool must not
    contain any Flights interactions.
    """
    pool = build_donor_pool("Flights", sample_dataset)
    # Pool should contain interactions from Hotels, Car-Rental, Attraction, Cross
    assert len(pool) > 0
    # Verify no Flights interactions leaked in (check source_domain tag)
    for entry in pool:
        assert entry["source_domain"] != "Flights"


def test_build_donor_pool_cross(sample_dataset):
    """
    For Cross cases, the donor pool should contain interactions from
    all four single-domain categories but not from Cross itself.
    """
    pool = build_donor_pool("Cross", sample_dataset)
    source_domains = {entry["source_domain"] for entry in pool}
    assert "Cross" not in source_domains
    # Should have at least some of the single-domain categories
    assert len(source_domains) >= 1


# ---------------------------------------------------------------------------
# 5. Determinism
# ---------------------------------------------------------------------------


def test_generate_haystack_deterministic(sample_dataset, sample_case_flights):
    """
    Generating haystack for the same case and target_tokens twice with the
    same seed must produce byte-identical output. This ensures reproducibility
    across runs without runtime randomness.
    """
    result1 = generate_haystack_for_case(
        case=sample_case_flights,
        dataset=sample_dataset,
        target_tokens=500,
    )
    result2 = generate_haystack_for_case(
        case=sample_case_flights,
        dataset=sample_dataset,
        target_tokens=500,
    )
    assert json.dumps(result1["haystack_messages"]) == json.dumps(
        result2["haystack_messages"]
    )
    assert result1["haystack_token_count"] == result2["haystack_token_count"]


def test_generate_haystack_different_thresholds_differ(
    sample_dataset, sample_case_flights
):
    """
    Different target_tokens values must produce different haystack data, even
    for the same case. This validates that the random seed incorporates
    the threshold so each context-fill level gets unique distractor data.
    """
    result_500 = generate_haystack_for_case(
        case=sample_case_flights,
        dataset=sample_dataset,
        target_tokens=500,
    )
    result_1000 = generate_haystack_for_case(
        case=sample_case_flights,
        dataset=sample_dataset,
        target_tokens=1000,
    )
    # They may share some interactions but should not be identical
    assert result_500["haystack_token_count"] != result_1000["haystack_token_count"]


# ---------------------------------------------------------------------------
# 6. Integration: end-to-end case generation
# ---------------------------------------------------------------------------


def test_generate_haystack_preserves_original_data(sample_dataset, sample_case_flights):
    """
    The generated output must preserve the original case data (id,
    conversations, functions) without modification. Only haystack_messages
    and haystack_token_count are added.
    """
    result = generate_haystack_for_case(
        case=sample_case_flights,
        dataset=sample_dataset,
        target_tokens=500,
    )
    assert result["id"] == sample_case_flights["id"]
    assert result["conversations"] == sample_case_flights["conversations"]
    assert result["functions"] == sample_case_flights["functions"]
    assert "haystack_messages" in result
    assert "haystack_token_count" in result


def test_generate_haystack_messages_valid_openai_format(
    sample_dataset, sample_case_flights
):
    """
    Every message in haystack_messages must follow OpenAI chat format:
    - assistant messages have role='assistant', content=None, tool_calls=[...]
    - tool messages have role='tool', tool_call_id=str, content=str
    - tool_call_ids in tool messages must reference the preceding assistant's tool_calls
    """
    result = generate_haystack_for_case(
        case=sample_case_flights,
        dataset=sample_dataset,
        target_tokens=500,
    )
    messages = result["haystack_messages"]
    assert len(messages) > 0

    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg["role"] == "assistant":
            assert msg["content"] is None
            assert "tool_calls" in msg
            assert len(msg["tool_calls"]) > 0
            # Collect tool_call_ids from this assistant message
            expected_ids = [tc["id"] for tc in msg["tool_calls"]]
            # The next N messages must be tool responses
            for j, tc_id in enumerate(expected_ids):
                tool_msg = messages[i + 1 + j]
                assert tool_msg["role"] == "tool"
                assert tool_msg["tool_call_id"] == tc_id
                assert isinstance(tool_msg["content"], str)
            i += 1 + len(expected_ids)
        else:
            pytest.fail(f"Expected assistant message at index {i}, got {msg['role']}")


def test_generate_haystack_no_self_domain(sample_dataset, sample_case_flights):
    """
    The haystack data must not contain tool interactions from the case's own
    domain. For a Flights case, no Flights-domain function names should appear
    in the haystack messages.
    """
    result = generate_haystack_for_case(
        case=sample_case_flights,
        dataset=sample_dataset,
        target_tokens=500,
    )
    # Flights-domain functions contain "Flight" in the name
    # Donor domains should not include Flights
    haystack_str = json.dumps(result["haystack_messages"])
    assert "Search_Flight_Location" not in haystack_str
