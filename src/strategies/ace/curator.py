"""
Curator agent for ACE strategy.
Maintains and improves the playbook.
"""

import os
from typing import Dict, List, Tuple

from memorch.strategies.ace.playbook_utils import (
    extract_json_from_text,
    apply_curator_operations,
)
from memorch.utils.llm_helpers import extract_content
from memorch.utils.logger import get_logger

logger = get_logger("ACE.Curator")


class Curator:
    """
    Curator agent that maintains and improves the playbook.
    """

    def __init__(self, prompt_path: str | None = None):
        """
        Initialize curator with prompt template.

        Args:
            prompt_path: Path to curator prompt file
        """
        if prompt_path is None:
            prompt_path = os.path.join(
                os.path.dirname(__file__), "prompts", "curator.prompt.md"
            )

        with open(prompt_path, "r") as f:
            self.prompt_template = f.read()

    def curate(
        self,
        current_playbook: str,
        recent_reflection: str,
        question_context: str,
        step: int,
        token_budget: int,
        playbook_stats: Dict,
        llm_client,
        model: str = "gpt-4-1-mini",
        next_global_id: int = 1,
    ) -> Tuple[str, int, List[Dict]]:
        """
        Curate the playbook based on recent performance.

        Args:
            current_playbook: Current playbook content
            recent_reflection: Recent reflection text
            question_context: Context about current task
            step: Current step number
            token_budget: Maximum playbook size
            playbook_stats: Statistics dict
            llm_client: LLM client
            model: Model to use
            next_global_id: Next available bullet ID

        Returns:
            (updated_playbook, next_global_id, operations)
        """
        # Format stats for prompt
        stats_text = f"""
Total bullets: {playbook_stats.get("total_bullets", 0)}
High performing: {playbook_stats.get("high_performing", 0)}
Problematic: {playbook_stats.get("problematic", 0)}
Unused: {playbook_stats.get("unused", 0)}
"""

        prompt = self.prompt_template.format(
            current_playbook=current_playbook,
            playbook_stats=stats_text,
            recent_reflection=recent_reflection or "No recent reflection",
            question_context=question_context or "No context",
            step=step,
            token_budget=token_budget,
        )

        logger.debug(
            f"Curator input - current_playbook (first 200 chars): {current_playbook[:200]}..."
        )
        logger.debug(
            f"Curator input - recent_reflection (first 200 chars): {recent_reflection[:200] if recent_reflection else '<empty>'}..."
        )
        logger.debug(f"Curator input - stats: {stats_text}")

        # Call LLM
        messages = [{"role": "user", "content": prompt}]
        response = llm_client.generate_plain(input_messages=messages, model=model)
        response_text = extract_content(response)

        logger.debug(
            f"Curator LLM response (first 400 chars): {response_text[:400]}..."
        )

        # Extract operations
        operations = self._extract_operations(response_text)
        logger.debug(f"Curator extracted operations: {operations}")

        # Apply operations
        updated_playbook, updated_id = apply_curator_operations(
            current_playbook, operations, next_global_id
        )

        logger.debug(
            f"Curator updated_playbook (first 300 chars): {updated_playbook[:300]}..."
        )

        return updated_playbook, updated_id, operations

    def _extract_operations(self, text: str) -> List[Dict]:
        """
        Extract curator operations from response.

        Args:
            text: Response text

        Returns:
            List of operation dicts
        """
        # Try JSON extraction
        json_data = extract_json_from_text(text)
        if json_data and "operations" in json_data:
            ops = json_data["operations"]
            if isinstance(ops, list):
                return ops

        return []
