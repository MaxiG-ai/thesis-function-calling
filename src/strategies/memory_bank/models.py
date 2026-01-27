"""
Data models for the Memory Bank strategy.

InteractionRecord stores the ground truth data for each tool interaction.
"""

from dataclasses import dataclass, field
from typing import Any, Dict
import time
import uuid


@dataclass
class InteractionRecord:
    """
    Represents a single tool interaction stored in the Fact Store.

    Attributes:
        trace_id: Unique identifier (UUID) for this interaction
        step_id: Which step in the task sequence this occurred
        tool_name: Name of the tool that was called
        raw_input: The arguments passed to the tool
        raw_output: The response from the tool (ground truth)
        timestamp: Unix timestamp when this was recorded
    """

    trace_id: str
    step_id: int
    tool_name: str
    raw_input: Dict[str, Any]
    raw_output: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def create(
        cls,
        step_id: int,
        tool_name: str,
        raw_input: Dict[str, Any],
        raw_output: Dict[str, Any],
    ) -> "InteractionRecord":
        """Factory method that auto-generates trace_id and timestamp."""
        return cls(
            trace_id=str(uuid.uuid4()),
            step_id=step_id,
            tool_name=tool_name,
            raw_input=raw_input,
            raw_output=raw_output,
            timestamp=time.time(),
        )
