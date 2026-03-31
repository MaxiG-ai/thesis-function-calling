#!/usr/bin/env python3
"""
Task Difficulty Scorer

This script evaluates the difficulty of user queries from ComplexFuncBench
by sending each task's user query to Claude Sonnet 4.6 for scoring.

The LLM scores each task on a scale from 1 (extremely easy) to 7 (unsolvable for an LLM),
considering factors like:
- Number of steps/API calls required
- Ambiguity in the request
- Complex reasoning or planning needs
- Dependencies between sub-tasks

Output: CSV file with columns [task_id, task_difficulty_score]
"""

import csv
import json
import sys
from pathlib import Path

import litellm

from memorch.utils.config import load_configs
from memorch.utils.logger import get_logger

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = get_logger("TaskDifficultyScorer")

# Configuration
BENCHMARK_FILE = (
    PROJECT_ROOT / "benchmarks/complex_func_bench/data/ComplexFuncBench.jsonl"
)
OUTPUT_FILE = PROJECT_ROOT / "tools/task_difficulty_scores.csv"
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs/model_config.toml"
EXPERIMENT_CONFIG_PATH = PROJECT_ROOT / "config.toml"

# Scoring prompt template
SCORING_PROMPT = """You are evaluating the difficulty of a user query for an LLM to solve using function calling. Score the difficulty from 1 (extremely easy) to 7 (unsolvable for an LLM).

Consider:
- Number of steps/API calls likely required
- Ambiguity in the request
- Need for complex reasoning or planning
- Dependencies between sub-tasks

Examples:
- Score 1: "Which flights leave from FRA to JFK on 2025-10-10?"
- Score 7: "I am visiting my mother over her birthday in South Carolina. We want to fly to the country, she always dreamt of going to and see all attractions there. Also get us an affordable car and nice hotels for max 200$ per night. Don't go more than 7 days I am on a budget."

User Query: {query}

Respond with ONLY a single integer from 1-7."""


def load_benchmark_tasks(file_path: Path) -> list[dict]:
    """
    Load tasks from ComplexFuncBench JSONL file.

    Returns:
        List of task dictionaries with 'id' and 'conversations' fields
    """
    tasks = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    logger.info(f"Loaded {len(tasks)} tasks from {file_path}")
    return tasks


def extract_user_query(task: dict) -> str:
    """
    Extract the first user message from a task's conversations.

    Args:
        task: Task dictionary with 'conversations' field

    Returns:
        The content of the first user message
    """
    for turn in task["conversations"]:
        if turn["role"] == "user":
            return turn["content"]
    raise ValueError(f"No user message found in task {task['id']}")


def score_task_difficulty(query: str, model_def, max_retries: int = 2) -> int:
    """
    Score the difficulty of a user query using Claude Sonnet 4.6.

    Args:
        query: The user query to score
        model_def: Model definition from config with credentials
        max_retries: Number of retries if invalid response is received

    Returns:
        Integer score from 1-7

    Raises:
        ValueError: If unable to get valid score after retries
    """
    prompt = SCORING_PROMPT.format(query=query)

    for attempt in range(max_retries + 1):
        try:
            # Prepare model kwargs
            model_kwargs = {}
            if hasattr(model_def, "temperature") and model_def.temperature is not None:
                model_kwargs["temperature"] = model_def.temperature

            # Call Claude via litellm
            response = litellm.completion(
                model=model_def.litellm_name,
                messages=[{"role": "user", "content": prompt}],
                api_base=model_def.api_base if hasattr(model_def, "api_base") else None,
                api_key=model_def.api_key if hasattr(model_def, "api_key") else None,
                drop_params=True,
                **model_kwargs,
            )

            # Extract and validate score
            content = response.choices[0].message.content.strip()
            score = int(content)

            if 1 <= score <= 7:
                return score
            else:
                logger.warning(
                    f"Score {score} out of range [1-7], attempt {attempt + 1}/{max_retries + 1}"
                )

        except (ValueError, AttributeError, IndexError) as e:
            logger.warning(
                f"Failed to parse score from response: {e}, attempt {attempt + 1}/{max_retries + 1}"
            )
            if attempt == max_retries:
                raise ValueError(
                    f"Unable to get valid score after {max_retries + 1} attempts"
                )

    raise ValueError(f"Unable to get valid score after {max_retries + 1} attempts")


def main():
    """
    Main execution function.

    Process:
    1. Load configuration for Claude credentials
    2. Load benchmark tasks
    3. For each task, extract user query and score it
    4. Write results to CSV
    """
    logger.info("=" * 60)
    logger.info("Task Difficulty Scorer - Starting")
    logger.info("=" * 60)

    # Load configuration
    logger.info("Loading configuration...")
    config = load_configs(str(EXPERIMENT_CONFIG_PATH), str(MODEL_CONFIG_PATH))

    # Get Claude Sonnet 4.6 model definition
    if "claude-sonnet-4-6" not in config.model_registry:
        logger.error("Model 'claude-sonnet-4-6' not found in model registry")
        sys.exit(1)

    model_def = config.model_registry["claude-sonnet-4-6"]
    logger.info(f"Using model: {model_def.litellm_name}")

    # Load benchmark tasks
    tasks = load_benchmark_tasks(BENCHMARK_FILE)
    total_tasks = len(tasks)

    # Process tasks and score them
    results = []
    failed_tasks = []

    logger.info(f"\nProcessing {total_tasks} tasks...")
    logger.info("-" * 60)

    for idx, task in enumerate(tasks, 1):
        task_id = task["id"]

        try:
            # Extract user query
            user_query = extract_user_query(task)

            # Score the task
            score = score_task_difficulty(user_query, model_def)

            # Store result
            results.append({"task_id": task_id, "task_difficulty_score": score})

            # Log progress every 10 tasks
            if idx % 10 == 0:
                logger.info(
                    f"Progress: {idx}/{total_tasks} tasks scored ({idx / total_tasks * 100:.1f}%)"
                )

        except Exception as e:
            logger.error(f"Failed to score task {task_id}: {e}")
            failed_tasks.append(task_id)
            # Continue with next task
            continue

    logger.info("-" * 60)
    logger.info(f"Completed: {len(results)}/{total_tasks} tasks scored successfully")

    if failed_tasks:
        logger.warning(f"Failed tasks ({len(failed_tasks)}): {', '.join(failed_tasks)}")

    # Write results to CSV
    logger.info(f"\nWriting results to {OUTPUT_FILE}...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["task_id", "task_difficulty_score"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    logger.info(f"✓ Results written to {OUTPUT_FILE}")
    logger.info("=" * 60)
    logger.info("Task Difficulty Scorer - Complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
