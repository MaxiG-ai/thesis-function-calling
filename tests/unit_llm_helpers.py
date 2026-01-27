"""
Tests for the LLM helper utilities.

These tests verify that the extract_content helper correctly handles
both dict-style and object-style message formats from LLM responses.
"""

from unittest.mock import MagicMock

from src.utils.llm_helpers import extract_content


class TestExtractContent:
    """Tests for extract_content function."""

    def test_extract_content_from_dict_message(self) -> None:
        """
        Test extraction when message is a dict (common with some providers).

        Some LLM providers return messages as plain dictionaries rather than
        objects. This test verifies the helper handles that case correctly.
        """
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = {"content": "Hello, world!", "role": "assistant"}

        result = extract_content(response)

        assert result == "Hello, world!"

    def test_extract_content_from_object_message(self) -> None:
        """
        Test extraction when message is an object with content attribute.

        OpenAI and most providers return message objects with a content attribute.
        This is the most common case.
        """
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = "Response from assistant"

        result = extract_content(response)

        assert result == "Response from assistant"

    def test_extract_content_strips_whitespace(self) -> None:
        """
        Test that extracted content has leading/trailing whitespace removed.

        LLM responses sometimes include extra whitespace that should be cleaned.
        """
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = {"content": "  trimmed content  \n"}

        result = extract_content(response)

        assert result == "trimmed content"

    def test_extract_content_handles_none_in_dict(self) -> None:
        """
        Test handling of None content in dict message format.

        Some responses may have content=None (e.g., function-call-only responses).
        The helper should return an empty string rather than raising.
        """
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = {"content": None, "role": "assistant"}

        result = extract_content(response)

        assert result == ""

    def test_extract_content_handles_none_in_object(self) -> None:
        """
        Test handling of None content in object message format.

        When the message object has content=None, return empty string.
        """
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = MagicMock()
        response.choices[0].message.content = None

        result = extract_content(response)

        assert result == ""

    def test_extract_content_handles_missing_content_key(self) -> None:
        """
        Test handling of dict message without content key.

        Edge case where the message dict doesn't have a content field at all.
        """
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = {"role": "assistant"}

        result = extract_content(response)

        assert result == ""

    def test_extract_content_handles_missing_content_attr(self) -> None:
        """
        Test handling of object message without content attribute.

        Edge case where the message object doesn't have a content attribute.
        Uses getattr with default to handle gracefully.
        """
        response = MagicMock(spec=[])  # Empty spec means no attributes
        response.choices = [MagicMock(spec=[])]
        message = MagicMock(spec=[])  # No content attribute
        response.choices[0].message = message

        result = extract_content(response)

        assert result == ""

    def test_extract_content_with_empty_string(self) -> None:
        """
        Test that empty string content is returned as empty string.

        Verifies the helper doesn't treat empty strings specially.
        """
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = {"content": ""}

        result = extract_content(response)

        assert result == ""

    def test_extract_content_with_multiline_content(self) -> None:
        """
        Test extraction of multiline content preserves internal newlines.

        Only leading/trailing whitespace should be stripped, not internal.
        """
        multiline = "Line 1\nLine 2\nLine 3"
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message = {"content": f"  {multiline}  "}

        result = extract_content(response)

        assert result == multiline
