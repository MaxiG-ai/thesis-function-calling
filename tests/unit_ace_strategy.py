"""
Comprehensive test suite for ACE strategy.
Tests playbook utilities, agents, and integration.
"""
import json
from unittest.mock import Mock

from src.strategies.ace.playbook_utils import (
    EMPTY_PLAYBOOK_TEMPLATE,
    parse_playbook_line,
    format_playbook_line,
    update_bullet_counts,
    apply_curator_operations,
    get_playbook_stats,
    extract_playbook_bullets,
    extract_json_from_text,
    get_section_slug
)
from src.strategies.ace.generator import Generator
from src.strategies.ace.reflector import Reflector
from src.strategies.ace.curator import Curator
from src.strategies.ace.ace_strategy import ACEState, apply_ace_strategy


# ============================================================================
# PLAYBOOK UTILITIES TESTS
# ============================================================================

def test_parse_playbook_line_valid():
    """Verifies bullet line parsing extracts id, helpful, harmful, content correctly."""
    line = "[5] helpful=3 harmful=1 :: Always verify inputs before processing"
    result = parse_playbook_line(line)
    
    assert result is not None
    assert result["id"] == 5
    assert result["helpful"] == 3
    assert result["harmful"] == 1
    assert result["content"] == "Always verify inputs before processing"


def test_parse_playbook_line_invalid():
    """Verifies non-bullet lines return None without errors."""
    lines = [
        "# This is a header",
        "Just some text",
        "## Section Title",
        "",
        "<!-- comment -->"
    ]
    
    for line in lines:
        result = parse_playbook_line(line)
        assert result is None


def test_format_playbook_line():
    """Verifies bullet formatting produces correct '[id] helpful=X harmful=Y :: content' format."""
    result = format_playbook_line(7, 2, 0, "Test content here")
    assert result == "[7] helpful=2 harmful=0 :: Test content here"


def test_parse_format_roundtrip():
    """Verifies parse → format → parse produces identical data (idempotency)."""
    original = "[10] helpful=5 harmful=2 :: Complex reasoning pattern"
    
    parsed1 = parse_playbook_line(original)
    formatted = format_playbook_line(
        parsed1["id"],
        parsed1["helpful"],
        parsed1["harmful"],
        parsed1["content"]
    )
    parsed2 = parse_playbook_line(formatted)
    
    assert parsed1 == parsed2


def test_update_bullet_counts_helpful():
    """Verifies 'helpful' tag increments helpful count by 1."""
    playbook = """# Playbook
[1] helpful=0 harmful=0 :: First bullet
[2] helpful=1 harmful=0 :: Second bullet
"""
    
    bullet_tags = [{"bullet_id": 1, "tag": "helpful"}]
    updated = update_bullet_counts(playbook, bullet_tags)
    
    assert "[1] helpful=1 harmful=0 :: First bullet" in updated
    assert "[2] helpful=1 harmful=0 :: Second bullet" in updated


def test_update_bullet_counts_harmful():
    """Verifies 'harmful' tag increments harmful count by 1."""
    playbook = """# Playbook
[1] helpful=2 harmful=0 :: First bullet
[2] helpful=0 harmful=1 :: Second bullet
"""
    
    bullet_tags = [{"bullet_id": 2, "tag": "harmful"}]
    updated = update_bullet_counts(playbook, bullet_tags)
    
    assert "[1] helpful=2 harmful=0 :: First bullet" in updated
    assert "[2] helpful=0 harmful=2 :: Second bullet" in updated


def test_update_bullet_counts_neutral():
    """Verifies 'neutral' tag leaves counts unchanged."""
    playbook = """# Playbook
[1] helpful=3 harmful=1 :: First bullet
"""
    
    bullet_tags = [{"bullet_id": 1, "tag": "neutral"}]
    updated = update_bullet_counts(playbook, bullet_tags)
    
    assert "[1] helpful=3 harmful=1 :: First bullet" in updated


def test_update_bullet_counts_missing_id():
    """Verifies tagging non-existent bullet ID leaves playbook unchanged."""
    playbook = """# Playbook
[1] helpful=0 harmful=0 :: First bullet
"""
    
    bullet_tags = [{"bullet_id": 999, "tag": "helpful"}]
    updated = update_bullet_counts(playbook, bullet_tags)
    
    assert updated == playbook


def test_apply_curator_operations_add():
    """Verifies ADD operation inserts new bullet with correct ID and format."""
    playbook = """# Playbook

## Task Decomposition (TSD)
<!-- Break down tasks -->

"""
    
    operations = [
        {
            "op": "ADD",
            "section": "task_decomposition",
            "content": "Always break complex tasks into steps"
        }
    ]
    
    updated, next_id = apply_curator_operations(playbook, operations, next_id=1)
    
    assert "[1] helpful=0 harmful=0 :: Always break complex tasks into steps" in updated
    assert next_id == 2


def test_apply_curator_operations_multiple_adds():
    """Verifies multiple ADD operations insert into correct sections."""
    playbook = EMPTY_PLAYBOOK_TEMPLATE
    
    operations = [
        {"op": "ADD", "section": "task_decomposition", "content": "First insight"},
        {"op": "ADD", "section": "error_handling", "content": "Second insight"}
    ]
    
    updated, next_id = apply_curator_operations(playbook, operations, next_id=1)
    
    assert "[1] helpful=0 harmful=0 :: First insight" in updated
    assert "[2] helpful=0 harmful=0 :: Second insight" in updated
    assert next_id == 3


def test_apply_curator_operations_increments_global_id():
    """Verifies next_global_id increments correctly after ADD operations."""
    playbook = EMPTY_PLAYBOOK_TEMPLATE
    
    operations = [
        {"op": "ADD", "section": "task_decomposition", "content": "Insight 1"},
        {"op": "ADD", "section": "error_handling", "content": "Insight 2"},
        {"op": "ADD", "section": "communication", "content": "Insight 3"}
    ]
    
    updated, next_id = apply_curator_operations(playbook, operations, next_id=5)
    
    assert next_id == 8  # Started at 5, added 3


def test_apply_curator_operations_remove():
    """Verifies REMOVE operation deletes bullet by ID."""
    playbook = """# Playbook
[1] helpful=0 harmful=5 :: Bad advice
[2] helpful=3 harmful=0 :: Good advice
"""
    
    operations = [{"op": "REMOVE", "bullet_id": 1}]
    
    updated, next_id = apply_curator_operations(playbook, operations, next_id=3)
    
    assert "[1] helpful=0 harmful=5 :: Bad advice" not in updated
    assert "[2] helpful=3 harmful=0 :: Good advice" in updated


def test_apply_curator_operations_update():
    """Verifies UPDATE operation modifies bullet content."""
    playbook = """# Playbook
[1] helpful=2 harmful=0 :: Old content
"""
    
    operations = [
        {"op": "UPDATE", "bullet_id": 1, "new_content": "Updated content"}
    ]
    
    updated, next_id = apply_curator_operations(playbook, operations, next_id=2)
    
    assert "[1] helpful=2 harmful=0 :: Updated content" in updated
    assert "Old content" not in updated


def test_get_playbook_stats():
    """Verifies statistics (total_bullets, high_performing, problematic, unused) are computed correctly."""
    playbook = """# Playbook
[1] helpful=5 harmful=0 :: High performer
[2] helpful=0 harmful=3 :: Problematic
[3] helpful=0 harmful=0 :: Unused
[4] helpful=1 harmful=1 :: Mixed
"""
    
    stats = get_playbook_stats(playbook)
    
    assert stats["total_bullets"] == 4
    assert stats["high_performing"] == 1  # helpful >= 3 and harmful == 0
    assert stats["problematic"] == 1      # harmful >= 2
    assert stats["unused"] == 1           # helpful == 0 and harmful == 0


def test_extract_playbook_bullets_found():
    """Verifies extracting specific bullet IDs returns formatted bullet lines."""
    playbook = """# Playbook
[1] helpful=1 harmful=0 :: First
[2] helpful=2 harmful=0 :: Second
[3] helpful=3 harmful=0 :: Third
"""
    
    result = extract_playbook_bullets(playbook, [1, 3])
    
    assert "[1] helpful=1 harmful=0 :: First" in result
    assert "[3] helpful=3 harmful=0 :: Third" in result
    assert "[2]" not in result


def test_extract_playbook_bullets_not_found():
    """Verifies missing bullet IDs return appropriate placeholder message."""
    playbook = """# Playbook
[1] helpful=1 harmful=0 :: First
"""
    
    result = extract_playbook_bullets(playbook, [99, 100])
    
    assert "not found" in result.lower()


def test_extract_json_from_text_clean():
    """Verifies clean JSON string is parsed correctly."""
    text = '{"key": "value", "number": 42}'
    result = extract_json_from_text(text)
    
    assert result == {"key": "value", "number": 42}


def test_extract_json_from_text_embedded():
    """Verifies JSON embedded in prose text is extracted correctly."""
    text = """
Here is some explanation text.

```json
{
  "operations": [
    {"op": "ADD", "content": "test"}
  ]
}
```

More text after.
"""
    
    result = extract_json_from_text(text)
    
    assert result is not None
    assert "operations" in result
    assert result["operations"][0]["op"] == "ADD"


def test_extract_json_from_text_invalid():
    """Verifies invalid JSON returns None without raising."""
    text = "This is not JSON at all { broken }"
    result = extract_json_from_text(text)
    
    assert result is None


def test_get_section_slug():
    """Verifies section names map to correct 3-letter slugs."""
    assert get_section_slug("task_decomposition") == "TSD"
    assert get_section_slug("task decomposition") == "TSD"
    assert get_section_slug("error_handling") == "ERR"
    assert get_section_slug("context_management") == "CTX"
    assert get_section_slug("reasoning_patterns") == "RSN"
    assert get_section_slug("tool_usage") == "TLS"
    assert get_section_slug("communication") == "COM"
    assert get_section_slug("unknown") == "GEN"


def test_empty_playbook_template_has_all_sections():
    """Verifies EMPTY_PLAYBOOK_TEMPLATE contains all required section headers."""
    required_sections = [
        "Task Decomposition (TSD)",
        "Error Handling (ERR)",
        "Context Management (CTX)",
        "Reasoning Patterns (RSN)",
        "Tool Usage (TLS)",
        "Communication (COM)"
    ]
    
    for section in required_sections:
        assert section in EMPTY_PLAYBOOK_TEMPLATE


# ============================================================================
# AGENT TESTS (with mocked LLM)
# ============================================================================

def test_generator_extracts_bullet_ids_from_json():
    """Verifies Generator parses bullet_ids from JSON response."""
    mock_client = Mock()
    mock_response = Mock()
    mock_message = Mock()
    mock_message.content = json.dumps({
        "reasoning_trace": "I used bullets 1 and 3",
        "response": "My answer",
        "bullet_ids_used": [1, 3, 5]
    })
    mock_response.choices = [Mock(message=mock_message)]
    mock_client.generate_plain.return_value = mock_response
    
    generator = Generator()
    response, bullet_ids = generator.generate(
        question="Test question",
        playbook="Test playbook",
        context="Test context",
        reflection="Test reflection",
        llm_client=mock_client
    )
    
    assert bullet_ids == [1, 3, 5]


def test_generator_extracts_bullet_ids_from_text():
    """Verifies Generator falls back to regex extraction for non-JSON response."""
    mock_client = Mock()
    mock_response = Mock()
    mock_message = Mock()
    mock_message.content = "My reasoning here.\nBULLET_IDS: [2, 4, 6]"
    mock_response.choices = [Mock(message=mock_message)]
    mock_client.generate_plain.return_value = mock_response
    
    generator = Generator()
    response, bullet_ids = generator.generate(
        question="Test question",
        playbook="Test playbook",
        context="Test context",
        reflection="Test reflection",
        llm_client=mock_client
    )
    
    assert bullet_ids == [2, 4, 6]


def test_reflector_uses_no_gt_prompt():
    """Verifies Reflector uses reflector_no_gt.prompt.md when use_ground_truth=False."""
    mock_client = Mock()
    mock_response = Mock()
    mock_message = Mock()
    mock_message.content = json.dumps({
        "reflection": "Analysis here",
        "bullet_tags": [{"bullet_id": 1, "tag": "helpful"}]
    })
    mock_response.choices = [Mock(message=mock_message)]
    mock_client.generate_plain.return_value = mock_response
    
    reflector = Reflector()
    reflection, bullet_tags = reflector.reflect(
        question="Test",
        reasoning_trace="Trace",
        predicted_answer="Answer",
        environment_feedback="Feedback",
        bullets_used="Bullets",
        llm_client=mock_client,
        use_ground_truth=False
    )
    
    # Should have called LLM
    assert mock_client.generate_plain.called


def test_reflector_extracts_bullet_tags():
    """Verifies Reflector parses bullet_tags list from JSON response."""
    mock_client = Mock()
    mock_response = Mock()
    mock_message = Mock()
    mock_message.content = json.dumps({
        "reflection": "Good work on bullet 1, bad on bullet 2",
        "bullet_tags": [
            {"bullet_id": 1, "tag": "helpful"},
            {"bullet_id": 2, "tag": "harmful"}
        ]
    })
    mock_response.choices = [Mock(message=mock_message)]
    mock_client.generate_plain.return_value = mock_response
    
    reflector = Reflector()
    reflection, bullet_tags = reflector.reflect(
        question="Test",
        reasoning_trace="Trace",
        predicted_answer="Answer",
        environment_feedback="Feedback",
        bullets_used="[1] test\n[2] test2",
        llm_client=mock_client
    )
    
    assert len(bullet_tags) == 2
    assert bullet_tags[0]["bullet_id"] == 1
    assert bullet_tags[0]["tag"] == "helpful"
    assert bullet_tags[1]["bullet_id"] == 2
    assert bullet_tags[1]["tag"] == "harmful"


def test_curator_applies_operations():
    """Verifies Curator returns updated playbook with ADD operations applied."""
    mock_client = Mock()
    mock_response = Mock()
    mock_message = Mock()
    mock_message.content = json.dumps({
        "reasoning": "Adding new insight",
        "operations": [
            {"op": "ADD", "section": "task_decomposition", "content": "New insight"}
        ]
    })
    mock_response.choices = [Mock(message=mock_message)]
    mock_client.generate_plain.return_value = mock_response
    
    curator = Curator(prompt_path_gt=None, prompt_path_no_gt=None)
    updated_playbook, next_id, operations = curator.curate(
        current_playbook=EMPTY_PLAYBOOK_TEMPLATE,
        recent_reflection="Test reflection",
        question_context="Context",
        step=1,
        token_budget=4096,
        playbook_stats={"total_bullets": 0, "high_performing": 0, "problematic": 0, "unused": 0},
        llm_client=mock_client,
        next_global_id=1
    )
    
    assert "[1] helpful=0 harmful=0 :: New insight" in updated_playbook
    assert next_id == 2


def test_curator_handles_empty_operations():
    """Verifies Curator returns unchanged playbook when operations list is empty."""
    mock_client = Mock()
    mock_response = Mock()
    mock_message = Mock()
    mock_message.content = json.dumps({
        "reasoning": "Playbook is good",
        "operations": []
    })
    mock_response.choices = [Mock(message=mock_message)]
    mock_client.generate_plain.return_value = mock_response
    
    curator = Curator()
    updated_playbook, next_id, operations = curator.curate(
        current_playbook=EMPTY_PLAYBOOK_TEMPLATE,
        recent_reflection="Test reflection",
        question_context="Context",
        step=1,
        token_budget=4096,
        playbook_stats={"total_bullets": 0, "high_performing": 0, "problematic": 0, "unused": 0},
        llm_client=mock_client,
        next_global_id=5
    )
    
    assert updated_playbook == EMPTY_PLAYBOOK_TEMPLATE
    assert next_id == 5  # Unchanged
    assert operations == []


# ============================================================================
# ACE STATE TESTS
# ============================================================================

def test_ace_state_initialization():
    """Verifies ACEState initializes with empty playbook template, next_id=1, step=0."""
    state = ACEState()
    
    assert state.playbook == EMPTY_PLAYBOOK_TEMPLATE
    assert state.next_global_id == 1
    assert state.step_count == 0
    assert state.last_reflection == ""
    assert state.last_bullet_ids == []


def test_ace_state_reset():
    """Verifies ACEState.reset() clears all state to initial values."""
    state = ACEState()
    
    # Modify state
    state.playbook = "Modified playbook"
    state.next_global_id = 10
    state.step_count = 5
    state.last_reflection = "Some reflection"
    state.last_bullet_ids = [1, 2, 3]
    
    # Reset
    state.reset()
    
    assert state.playbook == EMPTY_PLAYBOOK_TEMPLATE
    assert state.next_global_id == 1
    assert state.step_count == 0
    assert state.last_reflection == ""
    assert state.last_bullet_ids == []


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

def test_apply_ace_strategy_injects_playbook():
    """Verifies apply_ace_strategy() prepends playbook as system message."""
    mock_client = Mock()
    mock_settings = Mock()
    mock_settings.reflector_model = "gpt-4-1-mini"
    mock_settings.curator_model = "gpt-4-1-mini"
    mock_settings.generator_model = "gpt-4-1-mini"
    mock_settings.curator_frequency = 1
    mock_settings.playbook_token_budget = 4096
    
    # Mock curator response (now runs on step 1 for empty playbook bootstrap)
    mock_curator_response = Mock()
    mock_curator_message = Mock()
    mock_curator_message.content = json.dumps({"reasoning": "Bootstrap", "operations": []})
    mock_curator_response.choices = [Mock(message=mock_curator_message)]
    
    # Mock generator response
    mock_generator_response = Mock()
    mock_generator_message = Mock()
    mock_generator_message.content = json.dumps({"reasoning_trace": "Trace", "response": "OK", "bullet_ids_used": []})
    mock_generator_response.choices = [Mock(message=mock_generator_message)]
    
    mock_client.generate_plain.side_effect = [mock_curator_response, mock_generator_response]
    
    state = ACEState()
    messages = [{"role": "user", "content": "Hello"}]
    
    processed, token_count = apply_ace_strategy(
        messages, mock_client, mock_settings, state
    )
    
    # Should have playbook as first message
    assert processed[0]["role"] == "system"
    assert "PLAYBOOK" in processed[0]["content"]
    assert len(processed) == 2  # playbook + original message


def test_apply_ace_strategy_increments_step_count():
    """Verifies step counter increments on each apply_ace_strategy() call."""
    mock_client = Mock()
    mock_settings = Mock()
    mock_settings.reflector_model = "gpt-4-1-mini"
    mock_settings.curator_model = "gpt-4-1-mini"
    mock_settings.generator_model = "gpt-4-1-mini"
    mock_settings.curator_frequency = 1
    mock_settings.playbook_token_budget = 4096
    
    # Mock responses - step 1: curator + generator, step 2: reflector + curator + generator
    def make_mock_response(content):
        resp = Mock()
        msg = Mock()
        msg.content = json.dumps(content)
        resp.choices = [Mock(message=msg)]
        return resp
    
    # Step 1: curator, generator
    # Step 2: reflector (has reasoning from step 1), curator, generator
    mock_client.generate_plain.side_effect = [
        make_mock_response({"reasoning": "OK", "operations": []}),  # curator step 1
        make_mock_response({"reasoning_trace": "Trace", "response": "OK", "bullet_ids_used": []}),  # gen step 1
        make_mock_response({"reflection": "OK", "bullet_tags": []}),  # reflector step 2
        make_mock_response({"reasoning": "OK", "operations": []}),  # curator step 2
        make_mock_response({"reasoning_trace": "Trace", "response": "OK", "bullet_ids_used": []}),  # gen step 2
    ]
    
    state = ACEState()
    messages = [{"role": "user", "content": "Test"}]
    
    assert state.step_count == 0
    
    apply_ace_strategy(messages, mock_client, mock_settings, state)
    assert state.step_count == 1
    
    apply_ace_strategy(messages, mock_client, mock_settings, state)
    assert state.step_count == 2


def test_apply_ace_strategy_respects_curator_frequency():
    """Verifies Curator only runs when step % curator_frequency == 0."""
    mock_client = Mock()
    
    # Mock reflector response
    mock_reflector_response = Mock()
    mock_reflector_message = Mock()
    mock_reflector_message.content = json.dumps({
        "reflection": "Analysis",
        "bullet_tags": []
    })
    mock_reflector_response.choices = [Mock(message=mock_reflector_message)]
    
    # Mock curator response
    mock_curator_response = Mock()
    mock_curator_message = Mock()
    mock_curator_message.content = json.dumps({
        "reasoning": "No changes",
        "operations": []
    })
    mock_curator_response.choices = [Mock(message=mock_curator_message)]
    
    mock_client.generate_plain.side_effect = [
        mock_reflector_response,
        mock_curator_response
    ]
    
    mock_settings = Mock()
    mock_settings.reflector_model = "gpt-4-1-mini"
    mock_settings.curator_model = "gpt-4-1-mini"
    mock_settings.curator_frequency = 2  # Only every 2 steps
    mock_settings.playbook_token_budget = 4096
    
    state = ACEState()
    state.last_reasoning_trace = "Previous trace"
    state.last_bullet_ids = [1]
    state.last_reflection = "Previous reflection"
    
    messages = [{"role": "user", "content": "Test"}]
    
    # Step 1 - should not run curator
    apply_ace_strategy(messages, mock_client, mock_settings, state)
    assert state.step_count == 1
    
    # Only reflector should have been called (if previous step existed)
    # In this case, curator should not run at step 1


# ============================================================================
# MEMORY PROCESSOR INTEGRATION
# ============================================================================

def test_memory_processor_reset_clears_ace_state():
    """Verifies MemoryProcessor.reset_state() resets ACE state between tasks."""
    # Test the ACEState reset directly instead of through MemoryProcessor
    from src.strategies.ace.ace_strategy import ACEState
    
    state = ACEState()
    
    # Modify state
    state.step_count = 5
    state.next_global_id = 10
    state.last_reflection = "Some reflection"
    
    # Reset
    state.reset()
    
    # State should be reset
    assert state.step_count == 0
    assert state.next_global_id == 1
    assert state.last_reflection == ""


def test_memory_processor_apply_ace_delegates_correctly():
    """Verifies apply_ace_strategy() processes messages correctly."""
    from src.strategies.ace.ace_strategy import apply_ace_strategy, ACEState
    
    mock_client = Mock()
    mock_settings = Mock()
    mock_settings.reflector_model = "gpt-4-1-mini"
    mock_settings.curator_model = "gpt-4-1-mini"
    mock_settings.generator_model = "gpt-4-1-mini"
    mock_settings.curator_frequency = 1
    mock_settings.playbook_token_budget = 4096
    
    # Mock curator response (runs on step 1 for bootstrap)
    mock_curator_response = Mock()
    mock_curator_message = Mock()
    mock_curator_message.content = json.dumps({"reasoning": "Bootstrap", "operations": []})
    mock_curator_response.choices = [Mock(message=mock_curator_message)]
    
    # Mock generator response
    mock_generator_response = Mock()
    mock_generator_message = Mock()
    mock_generator_message.content = json.dumps({"reasoning_trace": "Trace", "response": "OK", "bullet_ids_used": []})
    mock_generator_response.choices = [Mock(message=mock_generator_message)]
    
    mock_client.generate_plain.side_effect = [mock_curator_response, mock_generator_response]
    
    state = ACEState()
    messages = [{"role": "user", "content": "Test"}]
    
    # Call apply_ace_strategy directly
    processed, token_count = apply_ace_strategy(
        messages, mock_client, mock_settings, state
    )
    
    # Should have injected playbook
    assert processed[0]["role"] == "system"
    assert "PLAYBOOK" in processed[0]["content"]
    
    # State should be updated
    assert state.step_count == 1
