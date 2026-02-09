"""
Pytest configuration for thesis-function-calling tests.

Handles path setup so test modules can import from tools/ directory.
"""

import sys
from pathlib import Path

# Add project root to path for importing tools module
# This is necessary because tools/ is not an installed package
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
