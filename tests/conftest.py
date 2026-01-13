from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure() -> None:
    """Ensure the repository root is on sys.path.

    Tests import the package as `src.*` but the project is not necessarily installed
    into the active environment during `uv run pytest`.
    """
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
