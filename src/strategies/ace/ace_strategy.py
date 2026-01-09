"""
Main ACE strategy orchestration.
Implements the apply_ace_strategy function and ACEState dataclass.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from src.strategies.ace.playbook_utils import (
    EMPTY_PLAYBOOK_TEMPLATE,
    update_bullet_counts,
    get_playbook_stats,
    extract_playbook_bullets
)
from src.strategies.ace.generator import Generator
from src.strategies.ace.reflector import Reflector
from src.strategies.ace.curator import Curator
from src.utils.logger import get_logger

logger = get_logger("ACEStrategy")


@dataclass
class ACEState:
    """
    Mutable state for ACE strategy within a single task.
    Reset between tasks.
    """
    playbook: str = field(default_factory=lambda: EMPTY_PLAYBOOK_TEMPLATE)
    next_global_id: int = 1
    last_reflection: str = ""
    last_bullet_ids: List[int] = field(default_factory=list)
    last_reasoning_trace: str = ""
    last_predicted_answer: str = ""
    step_count: int = 0
    
    def reset(self):
        """Reset state to initial values."""
        self.playbook = EMPTY_PLAYBOOK_TEMPLATE
        self.next_global_id = 1
        self.last_reflection = ""
        self.last_bullet_ids = []
        self.last_reasoning_trace = ""
        self.last_predicted_answer = ""
        self.step_count = 0


def apply_ace_strategy(
    messages: List[Dict],
    llm_client,
    settings,
    state: ACEState
) -> Tuple[List[Dict], int, ACEState]:
    """
    Apply ACE strategy to messages.
    
    This function:
    1. Extracts current action/observation from messages
    2. Runs Reflector on previous step (if exists)
    3. Updates playbook bullet counts
    4. Runs Curator (based on frequency)
    5. Injects playbook into messages as system prefix
    
    Args:
        messages: Current conversation messages
        llm_client: LLM client for agent calls
        settings: Memory strategy settings (MemoryDef)
        state: Current ACE state
        
    Returns:
        (processed_messages, token_count, updated_state)
    """
    state.step_count += 1
    logger.debug(f"ACE Strategy - Step {state.step_count}")
    
    # Extract current action/observation from messages
    # The last user message typically contains the observation
    last_user_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break
    
    # Run Reflector if we have previous step data
    if state.last_reasoning_trace and state.last_bullet_ids:
        logger.debug("Running Reflector...")
        reflector = Reflector()
        
        # Extract bullets used
        bullets_used = extract_playbook_bullets(state.playbook, state.last_bullet_ids)
        
        # Run reflection
        reflection_text, bullet_tags = reflector.reflect(
            question=last_user_msg,
            reasoning_trace=state.last_reasoning_trace,
            predicted_answer=state.last_predicted_answer,
            environment_feedback=last_user_msg,
            bullets_used=bullets_used,
            llm_client=llm_client,
            model=getattr(settings, 'reflector_model', 'gpt-4-1-mini'),
            use_ground_truth=False
        )
        
        # Update bullet counts
        if bullet_tags:
            state.playbook = update_bullet_counts(state.playbook, bullet_tags)
            logger.debug(f"Updated {len(bullet_tags)} bullet counts")
        
        state.last_reflection = reflection_text
    
    # Run Curator based on frequency
    curator_frequency = getattr(settings, 'curator_frequency', 1)
    if state.step_count % curator_frequency == 0 and state.last_reflection:
        logger.debug("Running Curator...")
        curator = Curator()
        
        # Get playbook stats
        stats = get_playbook_stats(state.playbook)
        
        # Run curation
        updated_playbook, updated_id, operations = curator.curate(
            current_playbook=state.playbook,
            recent_reflection=state.last_reflection,
            question_context=last_user_msg,
            step=state.step_count,
            token_budget=getattr(settings, 'playbook_token_budget', 4096),
            playbook_stats=stats,
            llm_client=llm_client,
            model=getattr(settings, 'curator_model', 'gpt-4-1-mini'),
            next_global_id=state.next_global_id,
            use_ground_truth=False
        )
        
        if operations:
            state.playbook = updated_playbook
            state.next_global_id = updated_id
            logger.debug(f"Applied {len(operations)} curator operations")
    
    # Inject playbook into messages
    # Insert as first system message
    playbook_message = {
        "role": "system",
        "content": f"## PLAYBOOK\n\n{state.playbook}"
    }
    
    processed_messages = [playbook_message] + messages
    
    # Calculate token count (simplified - actual implementation would use tokenizer)
    token_count = sum(len(msg.get("content", "").split()) for msg in processed_messages)
    
    return processed_messages, token_count, state
