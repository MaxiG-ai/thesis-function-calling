"""
Reflector agent for ACE strategy.
Analyzes performance and tags playbook bullets.
"""

import os
from typing import Dict, List, Tuple

from memorch.strategies.ace.playbook_utils import extract_json_from_text
from memorch.utils.llm_helpers import extract_content
from memorch.utils.logger import get_logger

logger = get_logger("ACE.Reflector")


class Reflector:
    """
    Reflector agent that evaluates performance and tags bullets.
    """

    def __init__(self, prompt_path: str | None = None):
        """
        Initialize reflector with prompt template.

        Args:
            prompt_path: Path to reflector prompt file
        """
        if prompt_path is None:
            prompt_path = os.path.join(
                os.path.dirname(__file__), "prompts", "reflector.prompt.md"
            )

        with open(prompt_path, "r") as f:
            self.prompt_template = f.read()

    def reflect(
        self,
        question: str,
        reasoning_trace: str,
        predicted_answer: str,
        environment_feedback: str,
        bullets_used: str,
        llm_client,
        model: str = "gpt-4-1-mini",
    ) -> Tuple[str, List[Dict]]:
        """
        Reflect on performance and tag bullets.

        Args:
            question: The question/task
            reasoning_trace: Agent's reasoning process
            predicted_answer: Agent's answer
            environment_feedback: Feedback from environment
            bullets_used: Formatted bullets that were used
            llm_client: LLM client
            model: Model to use

        Returns:
            (reflection_text, bullet_tags)
            bullet_tags: List of dicts with bullet_id and tag
        """
        prompt = self.prompt_template.format(
            question=question,
            reasoning_trace=reasoning_trace,
            predicted_answer=predicted_answer,
            environment_feedback=environment_feedback or "No feedback",
            bullets_used=bullets_used or "No bullets used",
        )

        logger.debug(
            f"Reflector input - bullets_used: {bullets_used[:200] if bullets_used else '<empty>'}..."
        )
        logger.debug(
            f"Reflector input - reasoning_trace (first 200 chars): {reasoning_trace[:200] if reasoning_trace else '<empty>'}..."
        )

        # Call LLM
        messages = [{"role": "user", "content": prompt}]
        response = llm_client.generate_plain(input_messages=messages, model=model)
        response_text = extract_content(response)

        logger.debug(
            f"Reflector LLM response (first 300 chars): {response_text[:300]}..."
        )

        # Extract bullet tags
        bullet_tags = self._extract_bullet_tags(response_text)
        logger.debug(f"Reflector extracted bullet_tags: {bullet_tags}")

        return response_text, bullet_tags

    def _extract_bullet_tags(self, text: str) -> List[Dict]:
        """
        Extract bullet tags from reflection response.

        Args:
            text: Response text

        Returns:
            List of dicts with bullet_id and tag
        """
        # Try JSON extraction
        json_data = extract_json_from_text(text)
        if json_data and "bullet_tags" in json_data:
            tags = json_data["bullet_tags"]
            if isinstance(tags, list):
                try:
                    return [
                        {
                            "bullet_id": int(tag["bullet_id"]),
                            "tag": tag["tag"],
                        }
                        for tag in tags
                    ]
                except (KeyError, ValueError, TypeError) as e:
                    logger.warning(
                        f"⚠ Error parsing bullet_tags from JSON: {e}. Falling back to text extraction."
                    )

        return []
