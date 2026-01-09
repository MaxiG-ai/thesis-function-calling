"""
Reflector agent for ACE strategy.
Analyzes performance and tags playbook bullets.
"""
import os
from typing import Dict, List, Tuple

from src.strategies.ace.playbook_utils import extract_json_from_text


class Reflector:
    """
    Reflector agent that evaluates performance and tags bullets.
    """
    
    def __init__(self, prompt_path_gt: str = None, prompt_path_no_gt: str = None):
        """
        Initialize reflector with prompt templates.
        
        Args:
            prompt_path_gt: Path to ground truth prompt
            prompt_path_no_gt: Path to no ground truth prompt
        """
        if prompt_path_gt is None:
            prompt_path_gt = os.path.join(
                os.path.dirname(__file__),
                "prompts",
                "reflector.prompt.md"
            )
        
        if prompt_path_no_gt is None:
            prompt_path_no_gt = os.path.join(
                os.path.dirname(__file__),
                "prompts",
                "reflector_no_gt.prompt.md"
            )
        
        with open(prompt_path_gt, 'r') as f:
            self.prompt_template_gt = f.read()
        
        with open(prompt_path_no_gt, 'r') as f:
            self.prompt_template_no_gt = f.read()
    
    def reflect(
        self,
        question: str,
        reasoning_trace: str,
        predicted_answer: str,
        environment_feedback: str,
        bullets_used: str,
        llm_client,
        model: str = "gpt-4-1-mini",
        use_ground_truth: bool = False,
        ground_truth: str = ""
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
            use_ground_truth: Whether ground truth is available
            ground_truth: Ground truth answer (if available)
            
        Returns:
            (reflection_text, bullet_tags)
            bullet_tags: List of dicts with bullet_id and tag
        """
        # Choose prompt template
        if use_ground_truth:
            prompt = self.prompt_template_gt.format(
                question=question,
                reasoning_trace=reasoning_trace,
                predicted_answer=predicted_answer,
                environment_feedback=environment_feedback or "No feedback",
                ground_truth=ground_truth,
                bullets_used=bullets_used or "No bullets used"
            )
        else:
            prompt = self.prompt_template_no_gt.format(
                question=question,
                reasoning_trace=reasoning_trace,
                predicted_answer=predicted_answer,
                environment_feedback=environment_feedback or "No feedback",
                bullets_used=bullets_used or "No bullets used"
            )
        
        # Call LLM
        messages = [{"role": "user", "content": prompt}]
        response = llm_client.generate_plain(input_messages=messages, model=model)
        
        # Extract response text
        if isinstance(response.choices[0].message, dict):
            response_text = response.choices[0].message.get("content", "")
        else:
            response_text = getattr(response.choices[0].message, "content", "")
        
        # Extract bullet tags
        bullet_tags = self._extract_bullet_tags(response_text)
        
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
                return [
                    {
                        "bullet_id": int(tag.get("bullet_id")),
                        "tag": tag.get("tag", "neutral")
                    }
                    for tag in tags
                    if isinstance(tag, dict) and "bullet_id" in tag
                ]
        
        return []
