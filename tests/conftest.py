"""Pytest configuration for thesis-function-calling tests.

Handles path setup so test modules can import from tools/ directory and
automatically marks tests by their folder placement.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path for importing tools module
# This is necessary because tools/ is not an installed package
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Apply unit/integration markers from the test file path.

    Files under tests/unit receive the ``unit`` marker and files under
    tests/integration receive the ``integration`` marker.
    """
    for item in items:
        path_parts = Path(str(item.fspath)).parts
        if "tests" not in path_parts:
            continue

        if "unit" in path_parts:
            item.add_marker(pytest.mark.unit)
        elif "integration" in path_parts:
            item.add_marker(pytest.mark.integration)
