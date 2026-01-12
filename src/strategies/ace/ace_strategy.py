"""
Main ACE strategy orchestration.
Implements the apply_ace_strategy function and ACEState dataclass.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

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
from src.utils.token_count import get_token_count

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
) -> Tuple[List[Dict], int]:
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
        (processed_messages, token_count)
    """
    state.step_count += 1
    curator_frequency = getattr(settings, 'curator_frequency', 1)
    
    # Debug: Log entry state and configuration
    logger.debug(f"\n{'='*60}")
    logger.debug(f"ACE Strategy - Step {state.step_count}")
    logger.debug(f"{'='*60}")
    logger.debug(f"State: last_reasoning_trace={'<set>' if state.last_reasoning_trace else '<empty>'}")
    logger.debug(f"State: last_bullet_ids={state.last_bullet_ids}")
    logger.debug(f"State: last_reflection={'<set>' if state.last_reflection else '<empty>'}")
    logger.debug(f"State: next_global_id={state.next_global_id}")
    logger.debug(f"Config: curator_frequency={curator_frequency}")
    logger.debug(f"Playbook preview (first 200 chars): {state.playbook[:200]}...")
    
    # Extract current action/observation from messages
    # The last user message typically contains the observation
    last_user_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break
    
    # Run Reflector if we have previous step data
    # Note: Only require reasoning trace - bullets may be empty on first steps (empty playbook bootstrap)
    has_reasoning = bool(state.last_reasoning_trace)
    has_bullets = bool(state.last_bullet_ids)
    logger.debug(f"Reflector conditions: has_reasoning={has_reasoning}, has_bullets={has_bullets}")
    
    if has_reasoning:
        logger.debug("✓ Reflector WILL run (conditions met)")
        reflector = Reflector()
        
        # Extract bullets used
        bullets_used = extract_playbook_bullets(state.playbook, state.last_bullet_ids)
        logger.debug(f"Bullets extracted for reflection: {bullets_used[:200] if bullets_used else '<empty>'}...")
        
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
        
        logger.debug(f"Reflector output - bullet_tags: {bullet_tags}")
        logger.debug(f"Reflector output - reflection_text (first 200 chars): {reflection_text[:200] if reflection_text else '<empty>'}...")
        
        # Update bullet counts
        if bullet_tags:
            state.playbook = update_bullet_counts(state.playbook, bullet_tags)
            logger.debug(f"Updated {len(bullet_tags)} bullet counts")
        else:
            logger.debug("⚠ No bullet_tags returned from Reflector - playbook counts NOT updated")
        
        state.last_reflection = reflection_text
        logger.debug(f"✓ last_reflection now set (len={len(reflection_text)})")
    else:
        logger.debug(f"✗ Reflector SKIPPED (has_reasoning={has_reasoning}, has_bullets={has_bullets})")
    
    # Run Curator based on frequency (runs even without reflection to bootstrap empty playbook)
    frequency_match = (state.step_count % curator_frequency == 0)
    has_reflection = bool(state.last_reflection)
    logger.debug(f"Curator conditions: step={state.step_count}, frequency={curator_frequency}, frequency_match={frequency_match}, has_reflection={has_reflection}")
    
    if frequency_match:
        logger.debug("✓ Curator WILL run (conditions met)")
        curator = Curator()
        
        # Get playbook stats
        stats = get_playbook_stats(state.playbook)
        logger.debug(f"Playbook stats: {stats}")
        
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
        
        logger.debug(f"Curator output - operations: {operations}")
        
        if operations:
            state.playbook = updated_playbook
            state.next_global_id = updated_id
            logger.debug(f"✓ Applied {len(operations)} curator operations")
            logger.debug(f"Playbook after curation (first 300 chars): {state.playbook[:300]}...")
        else:
            logger.debug("⚠ No operations returned from Curator - playbook NOT updated")
    else:
        logger.debug(f"✗ Curator SKIPPED (frequency_match={frequency_match}, has_reflection={has_reflection})")
    
    # Run Generator to prepare the next step
    # The Generator uses the playbook to guide decision-making and returns
    # a reasoning trace and bullet IDs that will be used in the next reflection
    logger.debug("Running Generator to prepare next step...")
    generator = Generator()
    
    # Extract context from messages (handle None content values)
    context = "\n".join([
        f"{msg.get('role', 'unknown')}: {(msg.get('content') or '')[:200]}..."
        for msg in messages[-3:]  # Last 3 messages for context
    ])
    
    reasoning_trace, bullet_ids_used = generator.generate(
        question=last_user_msg,
        playbook=state.playbook,
        context=context,
        reflection=state.last_reflection,
        llm_client=llm_client,
        model=getattr(settings, 'generator_model', 'gpt-4-1-mini')
    )
    
    logger.debug(f"Generator output - reasoning_trace (first 200 chars): {reasoning_trace[:200] if reasoning_trace else '<empty>'}...")
    logger.debug(f"Generator output - bullet_ids_used: {bullet_ids_used}")
    
    # Store for next reflection cycle
    state.last_reasoning_trace = reasoning_trace
    state.last_bullet_ids = bullet_ids_used
    state.last_predicted_answer = reasoning_trace  # Use reasoning trace as predicted answer
    logger.debug(f"✓ State updated for next cycle: last_bullet_ids={bullet_ids_used}")
    
    # Inject playbook into messages
    # Insert as first system message
    playbook_message = {
        "role": "system",
        "content": f"## PLAYBOOK\n\n{state.playbook}"
    }
    
    processed_messages = [playbook_message] + messages
    
    # Calculate token count using the existing token counter
    token_count = get_token_count(processed_messages)
    
    # Final summary logging
    logger.debug(f"\n--- ACE Step {state.step_count} Summary ---")
    logger.debug(f"Final playbook (first 400 chars): {state.playbook[:400]}...")
    logger.debug(f"Output token count: {token_count}")
    logger.debug(f"{'='*60}\n")
    
    return processed_messages, token_count
