"""
Tests for logging redirection to ensure Rich dashboard displays cleanly.

These tests verify that all logging output (from memorch, CFB benchmarks, and
the experiment runner) goes to log files and not to stdout/stderr, which would
interrupt the Rich live dashboard.
"""

import logging
import os
import tempfile
from io import StringIO
import sys

import pytest


def test_file_logger_no_console_output():
    """
    FileLogger should only write to file, not to console.

    This is critical for the Rich dashboard to display cleanly without
    log message interruptions.
    """
    from benchmarks.complex_func_bench.utils.logger import Logger as FileLogger

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
        log_file = f.name

    try:
        # Capture stdout
        captured_output = StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output

        # Create FileLogger
        logger = FileLogger("test_runner", log_file, level=logging.ERROR)

        # Log messages at various levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

        # Restore stdout
        sys.stdout = original_stdout
        console_output = captured_output.getvalue()

        # Verify NO console output
        assert console_output == "", (
            f"FileLogger wrote to console when it should only write to file: {console_output}"
        )

        # Verify file contains the logs (at ERROR level and above)
        with open(log_file, "r") as f:
            file_contents = f.read()

        assert "Error message" in file_contents
        assert "Critical message" in file_contents
        # DEBUG and INFO should not appear (level is ERROR)
        assert "Debug message" not in file_contents
        assert "Info message" not in file_contents

    finally:
        os.unlink(log_file)


def test_file_logger_prevents_propagation():
    """
    FileLogger should set propagate=False to prevent parent logger handlers.

    This ensures that even if parent loggers have StreamHandlers, our
    FileLogger's messages won't propagate up to them.
    """
    from benchmarks.complex_func_bench.utils.logger import Logger as FileLogger

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
        log_file = f.name

    try:
        logger = FileLogger("test_propagation", log_file, level=logging.INFO)

        # Verify propagation is disabled
        assert logger.logger.propagate is False, (
            "FileLogger should disable propagation to prevent parent handler interference"
        )

    finally:
        os.unlink(log_file)


def test_file_logger_removes_stream_handlers():
    """
    FileLogger should remove all StreamHandlers from the underlying logger.

    The base logger from memorch.utils.logger may have StreamHandlers attached.
    FileLogger must remove them to prevent console output.
    """
    from benchmarks.complex_func_bench.utils.logger import Logger as FileLogger

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
        log_file = f.name

    try:
        logger = FileLogger("test_no_streams", log_file, level=logging.INFO)

        # Check that no StreamHandlers remain
        stream_handlers = [
            h
            for h in logger.logger.handlers
            if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]

        assert len(stream_handlers) == 0, (
            f"FileLogger should remove all StreamHandlers, but found {len(stream_handlers)}"
        )

        # Verify we have exactly one FileHandler
        file_handlers = [
            h for h in logger.logger.handlers if isinstance(h, logging.FileHandler)
        ]

        assert len(file_handlers) == 1, (
            f"FileLogger should have exactly one FileHandler, but found {len(file_handlers)}"
        )

    finally:
        os.unlink(log_file)


def test_retry_decorator_uses_debug_level():
    """
    The retry() decorator should use logging.debug() for attempt messages.

    Since experiments run at INFO level, DEBUG messages won't appear in
    logs or console, preventing "Attempt X/Y failed" spam.
    """
    from benchmarks.complex_func_bench.utils.utils import retry
    import time

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
        log_file = f.name

    try:
        # Set up a logger at INFO level
        logger = logging.getLogger("test_retry")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

        # Test function that fails twice then succeeds
        call_count = 0

        @retry(max_attempts=3, delay=0.01)
        def sometimes_fails():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return None  # Fail
            return "success"

        result = sometimes_fails()

        assert result == "success"
        assert call_count == 3

        # Check log file - should NOT contain "Attempt" messages (they're DEBUG level)
        with open(log_file, "r") as f:
            log_contents = f.read()

        assert "Attempt" not in log_contents, (
            "Retry decorator should use DEBUG level, but 'Attempt' message appeared in INFO-level log"
        )

    finally:
        os.unlink(log_file)
