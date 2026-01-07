from typing import List, Dict, Tuple

from src.utils.logger import get_logger

logger = get_logger("HistoryUtils")

# Note: split_llm_trace and split_llm_trace_with_tools have been moved to src.utils.split_trace
# Use process_and_split_trace_user and process_and_split_trace_user_tool instead