"""
Curator agent for ACE strategy.
Maintains and improves the playbook.
"""
import os
from typing import Dict, List, Tuple

from src.strategies.ace.playbook_utils import extract_json_from_text, apply_curator_operations
from src.utils.logger import get_logger

logger = get_logger("ACE.Curator")


class Curator:
    """
    Curator agent that maintains and improves the playbook.
    """
    
    def __init__(self, prompt_path_gt: str = None, prompt_path_no_gt: str = None):
        """
        Initialize curator with prompt templates.
        
        Args:
            prompt_path_gt: Path to ground truth prompt
            prompt_path_no_gt: Path to no ground truth prompt
        """
        if prompt_path_gt is None:
            prompt_path_gt = os.path.join(
                os.path.dirname(__file__),
                "prompts",
                "curator.prompt.md"
            )
        
        if prompt_path_no_gt is None:
            prompt_path_no_gt = os.path.join(
                os.path.dirname(__file__),
                "prompts",
                "curator_no_gt.prompt.md"
            )
        
        with open(prompt_path_gt, 'r') as f:
            self.prompt_template_gt = f.read()
        
        with open(prompt_path_no_gt, 'r') as f:
            self.prompt_template_no_gt = f.read()
    
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
        use_ground_truth: bool = False
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
            use_ground_truth: Whether ground truth is available
            
        Returns:
            (updated_playbook, next_global_id, operations)
        """
        # Format stats for prompt
        stats_text = f"""
Total bullets: {playbook_stats.get('total_bullets', 0)}
High performing: {playbook_stats.get('high_performing', 0)}
Problematic: {playbook_stats.get('problematic', 0)}
Unused: {playbook_stats.get('unused', 0)}
"""
        
        # Choose prompt template
        if use_ground_truth:
            prompt = self.prompt_template_gt.format(
                current_playbook=current_playbook,
                playbook_stats=stats_text,
                recent_reflection=recent_reflection or "No recent reflection",
                question_context=question_context or "No context",
                step=step,
                token_budget=token_budget
            )
        else:
            prompt = self.prompt_template_no_gt.format(
                current_playbook=current_playbook,
                playbook_stats=stats_text,
                recent_reflection=recent_reflection or "No recent reflection",
                question_context=question_context or "No context",
                step=step,
                token_budget=token_budget
            )
        
        logger.debug(f"Curator input - current_playbook (first 200 chars): {current_playbook[:200]}...")
        logger.debug(f"Curator input - recent_reflection (first 200 chars): {recent_reflection[:200] if recent_reflection else '<empty>'}...")
        logger.debug(f"Curator input - stats: {stats_text}")
        
        # Call LLM
        messages = [{"role": "user", "content": prompt}]
        response = llm_client.generate_plain(input_messages=messages, model=model)
        
        # Extract response text
        if isinstance(response.choices[0].message, dict):
            response_text = response.choices[0].message.get("content", "")
        else:
            response_text = getattr(response.choices[0].message, "content", "")
        
        logger.debug(f"Curator LLM response (first 400 chars): {response_text[:400]}...")
        
        # Extract operations
        operations = self._extract_operations(response_text)
        logger.debug(f"Curator extracted operations: {operations}")
        
        # Apply operations
        updated_playbook, updated_id = apply_curator_operations(
            current_playbook,
            operations,
            next_global_id
        )
        
        logger.debug(f"Curator updated_playbook (first 300 chars): {updated_playbook[:300]}...")
        
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
