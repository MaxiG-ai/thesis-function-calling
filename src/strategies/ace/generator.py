"""
Generator agent for ACE strategy.
Generates actions using the playbook as guidance.
"""
import os
import re
from typing import Dict, List, Tuple

from src.strategies.ace.playbook_utils import extract_json_from_text


class Generator:
    """
    Generator agent that uses playbook to guide decision-making.
    """
    
    def __init__(self, prompt_path: str = None):
        """
        Initialize generator with prompt template.
        
        Args:
            prompt_path: Path to prompt file (defaults to generator.prompt.md)
        """
        if prompt_path is None:
            prompt_path = os.path.join(
                os.path.dirname(__file__),
                "prompts",
                "generator.prompt.md"
            )
        
        with open(prompt_path, 'r') as f:
            self.prompt_template = f.read()
    
    def generate(
        self,
        question: str,
        playbook: str,
        context: str,
        reflection: str,
        llm_client,
        model: str = "gpt-4-1-mini"
    ) -> Tuple[str, List[int]]:
        """
        Generate response using playbook guidance.
        
        Args:
            question: Current task/question
            playbook: Current playbook content
            context: Additional context
            reflection: Recent reflection text
            llm_client: LLM client for inference
            model: Model to use
            
        Returns:
            (response_text, bullet_ids_used)
        """
        # Build prompt
        prompt = self.prompt_template.format(
            playbook=playbook,
            reflection=reflection or "No recent reflection",
            question=question,
            context=context or "No additional context"
        )
        
        # Call LLM
        messages = [{"role": "user", "content": prompt}]
        response = llm_client.generate_plain(input_messages=messages, model=model)
        
        # Extract response text
        if isinstance(response.choices[0].message, dict):
            response_text = response.choices[0].message.get("content", "")
        else:
            response_text = getattr(response.choices[0].message, "content", "")
        
        # Extract bullet IDs
        bullet_ids = self._extract_bullet_ids(response_text)
        
        return response_text, bullet_ids
    
    def _extract_bullet_ids(self, text: str) -> List[int]:
        """
        Extract bullet IDs from response text.
        
        Tries JSON first, then falls back to regex.
        
        Args:
            text: Response text
            
        Returns:
            List of bullet IDs
        """
        # Try JSON extraction
        json_data = extract_json_from_text(text)
        if json_data and "bullet_ids_used" in json_data:
            ids = json_data["bullet_ids_used"]
            if isinstance(ids, list):
                return [int(i) for i in ids if isinstance(i, (int, str)) and str(i).isdigit()]
        
        # Fallback: regex for BULLET_IDS: [1, 2, 3]
        pattern = r'BULLET_IDS:\s*\[([^\]]+)\]'
        match = re.search(pattern, text)
        if match:
            ids_str = match.group(1)
            return [int(x.strip()) for x in ids_str.split(',') if x.strip().isdigit()]
        
        # Fallback: any list of numbers in brackets
        pattern = r'\[(\d+(?:\s*,\s*\d+)*)\]'
        matches = re.findall(pattern, text)
        if matches:
            # Use the last match (most likely to be the bullet IDs)
            ids_str = matches[-1]
            return [int(x.strip()) for x in ids_str.split(',') if x.strip()]
        
        return []
