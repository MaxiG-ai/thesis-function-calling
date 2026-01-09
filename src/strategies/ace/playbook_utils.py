"""
Playbook utilities for ACE strategy.
Handles parsing, formatting, and manipulation of playbook bullets.
"""
import re
import json
from typing import Dict, List, Optional, Tuple


# Empty playbook template with standard sections
EMPTY_PLAYBOOK_TEMPLATE = """# Agent Playbook

## Task Decomposition (TSD)
<!-- Break down complex tasks into manageable steps -->

## Error Handling (ERR)
<!-- Strategies for detecting and recovering from errors -->

## Context Management (CTX)
<!-- Techniques for maintaining relevant context -->

## Reasoning Patterns (RSN)
<!-- Proven reasoning approaches and heuristics -->

## Tool Usage (TLS)
<!-- Best practices for using available tools -->

## Communication (COM)
<!-- Guidelines for clear and effective responses -->
"""


def parse_playbook_line(line: str) -> Optional[Dict]:
    """
    Parse a playbook bullet line into its components.
    
    Format: [id] helpful=X harmful=Y :: content
    
    Args:
        line: A line from the playbook
        
    Returns:
        Dict with keys: id, helpful, harmful, content
        None if line is not a valid bullet
    """
    # Match pattern: [id] helpful=X harmful=Y :: content
    pattern = r'^\[(\d+)\]\s+helpful=(\d+)\s+harmful=(\d+)\s+::\s+(.+)$'
    match = re.match(pattern, line.strip())
    
    if not match:
        return None
    
    bullet_id, helpful, harmful, content = match.groups()
    return {
        "id": int(bullet_id),
        "helpful": int(helpful),
        "harmful": int(harmful),
        "content": content.strip()
    }


def format_playbook_line(bullet_id: int, helpful: int, harmful: int, content: str) -> str:
    """
    Format a playbook bullet line.
    
    Args:
        bullet_id: Unique bullet identifier
        helpful: Number of times marked helpful
        harmful: Number of times marked harmful
        content: Bullet content text
        
    Returns:
        Formatted bullet string
    """
    return f"[{bullet_id}] helpful={helpful} harmful={harmful} :: {content}"


def update_bullet_counts(playbook_text: str, bullet_tags: List[Dict]) -> str:
    """
    Update helpful/harmful counts for tagged bullets.
    
    Args:
        playbook_text: Current playbook content
        bullet_tags: List of dicts with bullet_id and tag (helpful/harmful/neutral)
        
    Returns:
        Updated playbook text
    """
    lines = playbook_text.split('\n')
    updated_lines = []
    
    # Build lookup map for tags
    tag_map = {tag["bullet_id"]: tag["tag"] for tag in bullet_tags}
    
    for line in lines:
        parsed = parse_playbook_line(line)
        if parsed and parsed["id"] in tag_map:
            tag = tag_map[parsed["id"]]
            if tag == "helpful":
                parsed["helpful"] += 1
            elif tag == "harmful":
                parsed["harmful"] += 1
            # neutral: no change
            
            updated_lines.append(format_playbook_line(
                parsed["id"],
                parsed["helpful"],
                parsed["harmful"],
                parsed["content"]
            ))
        else:
            updated_lines.append(line)
    
    return '\n'.join(updated_lines)


def apply_curator_operations(
    playbook_text: str, 
    operations: List[Dict], 
    next_id: int
) -> Tuple[str, int]:
    """
    Apply curator operations to the playbook.
    
    Supports:
    - ADD: Insert new bullet into section
    - REMOVE: Delete bullet by ID
    - UPDATE: Modify bullet content
    
    Args:
        playbook_text: Current playbook
        operations: List of operation dicts
        next_id: Next available bullet ID
        
    Returns:
        (updated_playbook, next_global_id)
    """
    lines = playbook_text.split('\n')
    updated_id = next_id
    
    for op in operations:
        op_type = op.get("op", "").upper()
        
        if op_type == "ADD":
            section = op.get("section", "")
            content = op.get("content", "")
            section_slug = get_section_slug(section)
            
            # Find the section and add bullet after section header
            new_bullet = format_playbook_line(updated_id, 0, 0, content)
            section_found = False
            
            for i, line in enumerate(lines):
                # Look for section header (e.g., "## Task Decomposition (TSD)")
                if f"({section_slug})" in line or section.lower() in line.lower():
                    section_found = True
                    # Find insertion point (after header and comments)
                    insert_idx = i + 1
                    while insert_idx < len(lines) and (
                        lines[insert_idx].strip().startswith('<!--') or
                        lines[insert_idx].strip() == ''
                    ):
                        insert_idx += 1
                    lines.insert(insert_idx, new_bullet)
                    break
            
            if not section_found:
                # Append to end if section not found
                lines.append(f"\n## {section}")
                lines.append(new_bullet)
            
            updated_id += 1
            
        elif op_type == "REMOVE":
            bullet_id = op.get("bullet_id")
            if bullet_id:
                lines = [
                    line for line in lines 
                    if not (parse_playbook_line(line) and 
                           parse_playbook_line(line)["id"] == bullet_id)
                ]
        
        elif op_type == "UPDATE":
            bullet_id = op.get("bullet_id")
            new_content = op.get("new_content", "")
            if bullet_id and new_content:
                for i, line in enumerate(lines):
                    parsed = parse_playbook_line(line)
                    if parsed and parsed["id"] == bullet_id:
                        lines[i] = format_playbook_line(
                            parsed["id"],
                            parsed["helpful"],
                            parsed["harmful"],
                            new_content
                        )
                        break
    
    return '\n'.join(lines), updated_id


def get_playbook_stats(playbook_text: str) -> Dict:
    """
    Compute statistics about the playbook.
    
    Returns:
        Dict with total_bullets, high_performing, problematic, unused
    """
    lines = playbook_text.split('\n')
    bullets = [parse_playbook_line(line) for line in lines if parse_playbook_line(line)]
    
    total = len(bullets)
    high_performing = sum(1 for b in bullets if b["helpful"] >= 3 and b["harmful"] == 0)
    problematic = sum(1 for b in bullets if b["harmful"] >= 2)
    unused = sum(1 for b in bullets if b["helpful"] == 0 and b["harmful"] == 0)
    
    return {
        "total_bullets": total,
        "high_performing": high_performing,
        "problematic": problematic,
        "unused": unused
    }


def extract_playbook_bullets(playbook_text: str, bullet_ids: List[int]) -> str:
    """
    Extract specific bullets by ID from playbook.
    
    Args:
        playbook_text: Full playbook content
        bullet_ids: List of bullet IDs to extract
        
    Returns:
        Formatted string of bullets
    """
    lines = playbook_text.split('\n')
    found_bullets = []
    
    for line in lines:
        parsed = parse_playbook_line(line)
        if parsed and parsed["id"] in bullet_ids:
            found_bullets.append(line)
    
    if found_bullets:
        return '\n'.join(found_bullets)
    else:
        return f"Bullets {bullet_ids} not found in playbook"


def extract_json_from_text(text: str) -> Optional[Dict]:
    """
    Extract and parse JSON from text that may contain prose.
    
    Args:
        text: Text potentially containing JSON
        
    Returns:
        Parsed JSON dict or None if parsing fails
    """
    # Try parsing as-is first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    
    # Look for JSON code blocks
    json_pattern = r'```json\s*(\{.*?\})\s*```'
    match = re.search(json_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Look for bare JSON objects
    json_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})'
    matches = re.findall(json_pattern, text, re.DOTALL)
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    return None


def get_section_slug(section_name: str) -> str:
    """
    Map section name to 3-letter slug.
    
    Args:
        section_name: Full section name
        
    Returns:
        3-letter slug
    """
    slug_map = {
        "task_decomposition": "TSD",
        "task decomposition": "TSD",
        "error_handling": "ERR",
        "error handling": "ERR",
        "context_management": "CTX",
        "context management": "CTX",
        "reasoning_patterns": "RSN",
        "reasoning patterns": "RSN",
        "tool_usage": "TLS",
        "tool usage": "TLS",
        "communication": "COM",
    }
    
    return slug_map.get(section_name.lower(), "GEN")
