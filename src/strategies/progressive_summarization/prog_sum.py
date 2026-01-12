from typing import Dict, List, Optional
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.split_trace import process_and_split_trace_user

logger = get_logger("ProgressiveSummarization")


def _resolve_prompt_path(prompt_path: Optional[str]) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    if prompt_path:
        candidate = Path(prompt_path)
        if candidate.is_file():
            return candidate
        candidate_from_root = repo_root / prompt_path
        if candidate_from_root.is_file():
            return candidate_from_root
    return repo_root / "src/strategies/progressive_summarization/prog_sum.prompt.md"


def summarize_conv_history(
    messages: List[Dict],
    llm_client,
    summarizer_model: str = "gpt-4-1-mini",
    summary_prompt_path: Optional[str] = None,
) -> List[Dict]:
    if llm_client is None:
        raise ValueError("llm_client is required for progressive summarization")
    
    user_query, conversation_history = process_and_split_trace_user(messages)

    prompt_file = _resolve_prompt_path(summary_prompt_path)
    summarization_prompt = prompt_file.read_text(encoding="utf-8")

    # Build prompt for summarization
    prompt_messages = [ 
        {"role": "system", "content": summarization_prompt},
        {"role": "user", "content": f"Conversation history to compress:\n{conversation_history}"},
    ]

    # Call LLM to generate summary (let exceptions propagate)
    response = llm_client.generate_plain(
        input_messages=prompt_messages, 
        model=summarizer_model
    )

    # Extract summary text from response
    message = response.choices[0].message
    if isinstance(message, dict):
        summary_text = (message.get("content") or "").strip()
    else:
        summary_text = (getattr(message, "content", "") or "").strip()

    if not summary_text:
        raise ValueError("Summarization returned empty content")

    # Build final message list: [summary, user query]
    summary_message = {"role": "system", "content": summary_text}
    
    result = []
    if user_query:
        result.extend(user_query)
    result.extend([summary_message])
    
    return result