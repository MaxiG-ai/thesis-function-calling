# ACE for ComplexFuncBench: Implementation Specification

## 1. Project Overview

**Goal:** Adapt the **Agentic Context Engineering (ACE)** framework to serve as an **Intra-Task Memory** mechanism for the **ComplexFuncBench** benchmark.

[cite_start]Unlike the standard ACE implementation, which evolves a "Playbook" across thousands of different tasks [Cross-Task Learning](cite: 16), this implementation will use ACE to evolve a memory state *within* a single multi-step interaction. [cite_start]The memory must be **completely reset** between benchmark tasks to ensure no information leaks[cite: 1].

### Key References

* [cite_start]**Thesis/Context:** *Context Engineering for Multi-step Tool Use* [Maximilian Graf, KIT](cite: 1).
* [cite_start]**Framework:** ACE [Agentic Context Engineering](cite: 16).

---

## 2. Background & Architecture

### The Concept

[cite_start]The core idea is to replace the standard "Black Box" LLM in your research design [cite: 1] with an ACE-driven loop.

1. **Task Input:** The agent receives a complex function request.
2. **Interaction Loop:** The agent attempts to solve it using tools.
3. **ACE Role:** * **Generator:** Decides the next tool call using the current Memory (Playbook).
    * **Reflector/Curator:** Analyzes the result of the tool call and updates the Memory immediately.
4. **Reset:** Once the task is done, the Memory is wiped.

### Component Mapping

| Function | ACE Component | Responsibility in Benchmark |
| :--- | :--- | :--- |
| **Generate Interaction** | `ace.core.Generator` | Takes the current history + memory and outputs the next tool action. |
| **Create Memories** | `ace.core.Reflector` | Analyzes the last action and observation (Environment Feedback). |
| **Update Context** | `ace.core.Curator` | Edits the Memory string (Playbook) based on the Reflector's insights. |

---

## 3. Technical Specifications

### 3.1 Dependencies

You will need the `ace` package structure provided in the repository. Ensure the following are accessible in your python path:

* `ace.core.generator`
* `ace.core.reflector`
* `ace.core.curator`
* `utils` (for client initialization)

### 3.2 Data Structures

* **`task`**: An object from `ComplexFuncBench` containing the `instruction` (query) and `tools` definitions.
* **`interaction_history`**: A string growing with every step: `User: ... \n Assistant: ... \n Observation: ...`
* **`current_memory` (The Playbook)**: A string containing the evolved context. Initialize this as an empty template at the start of every task.

---

## 4. Implementation Guide

### Step 1: Initialization Script (`benchmark_adapter.py`)

First, setup the core agents. [cite_start]You can reuse the initialization logic from `utils.py` [cite: 15] or instantiate them manually.

```python
import os
from ace.core import Generator, Reflector, Curator
from utils import initialize_clients

# 1. Setup Clients (Sambanova, OpenAI, etc.)
# Ensure .env is set with API keys
api_provider = "openai" # or "sambanova", "together"
generator_client, reflector_client, curator_client = initialize_clients(api_provider)

# 2. Initialize ACE Agents
# We use the same model for all, or mix and match as per research needs
MODEL_NAME = "gpt-4o" 

generator = Generator(generator_client, api_provider, MODEL_NAME)
reflector = Reflector(reflector_client, api_provider, MODEL_NAME)
curator = Curator(curator_client, api_provider, MODEL_NAME)

# 3. Define the Empty Memory Template
# [cite_start]This matches the structure ACE expects [cite: 23]
EMPTY_MEMORY_TEMPLATE = """## STRATEGIES & INSIGHTS
(No insights yet)

## PREVIOUS MISTAKES
(No mistakes recorded)

## TOOL USAGE PATTERNS
(No patterns recorded)
"""

```

### Step 2: Function Definitions

#### Function A: `generate_next_interaction`

This function wraps the `Generator`. It takes the current state and decides the next move.

```python
def generate_next_interaction(task_input, history, current_memory):
    """
    Generates the next interaction/message/tool-call.
    
    Args:
        task_input (str): The original query/task from the benchmark.
        history (str): The conversation history so far (Context).
        current_memory (str): The current ACE Playbook string.
        
    Returns:
        str: The raw model response (reasoning + tool call).
    """
    # The ACE Generator expects a 'reflection' argument usually passed from 
    # the previous step, but in this live loop, the reflection is already 
    # embedded in the 'current_memory' by the Curator.
    
    response, _, _ = generator.generate(
        question=task_input,
        playbook=current_memory,     # The evolving memory
        context=history,             # The interaction history
        reflection="(See Playbook)", # Reflection is implicitly in the playbook
        call_id="benchmark_step"
    )
    
    return response

```

#### Function B: `create_and_update_memory`

This function chains the `Reflector` and `Curator` to update the memory based on the immediate past step.

```python
def create_and_update_memory(task_input, last_action, observation, current_memory, step_count):
    """
    Reflects on the last action and updates the memory (Playbook).
    
    Args:
        task_input (str): The original query.
        last_action (str): What the agent just did (Reasoning + Tool Call).
        observation (str): The output from the environment/tool.
        current_memory (str): The current Playbook.
        step_count (int): Current step number (for logging).
        
    Returns:
        str: The updated Playbook (Memory).
    """
    
    # --- Sub-step 1: Reflect ---
    # The Reflector analyzes if the action was successful based on the observation.
    # We treat the 'observation' as 'environment_feedback'.
    
    reflection_content, bullet_tags, _ = reflector.reflect(
        question=task_input,
        reasoning_trace=last_action,
        predicted_answer=last_action, # In tool use, the action is the "answer" so far
        ground_truth=None,            # We don't have final GT yet, we rely on observation
        environment_feedback=observation,
        bullets_used="",              # Optional: extract if needed
        use_ground_truth=False
    )

    # --- Sub-step 2: Curate ---
    # The Curator integrates the reflection into the persistent memory string.
    
    # [cite_start]Dummy stats required by curator signature [cite: 27]
    playbook_stats = {"length": len(current_memory)}
    
    updated_memory, _, _, _ = curator.curate(
        current_playbook=current_memory,
        recent_reflection=reflection_content,
        question_context=f"Task: {task_input}\nAction: {last_action}\nResult: {observation}",
        current_step=step_count,
        total_samples=100,      # Dummy value
        token_budget=4096,      # Limit memory size
        playbook_stats=playbook_stats,
        use_ground_truth=False
    )
    
    return updated_memory

```

### Step 3: The Benchmark Loop (`run_benchmark.py`)

This is the orchestration logic that ties it into `ComplexFuncBench`.

```python
# Pseudo-code for the benchmark runner
from benchmark_adapter import generate_next_interaction, create_and_update_memory, EMPTY_MEMORY_TEMPLATE

def run_complex_func_bench(dataset):
    results = []

    for task_id, task in enumerate(dataset):
        print(f"--- Starting Task {task_id} ---")
        
        # [cite_start]1. RESET MEMORY (Critical Requirement) [cite: 1]
        current_memory = EMPTY_MEMORY_TEMPLATE
        interaction_history = ""
        
        # 2. Reset Environment
        env = ComplexFuncBenchEnv(task) 
        done = False
        step_count = 0
        
        while not done:
            step_count += 1
            
            # A. Generate Next Action
            action = generate_next_interaction(
                task_input=task.instruction, 
                history=interaction_history, 
                current_memory=current_memory
            )
            
            # B. Execute Action in Benchmark Environment
            observation, done, info = env.step(action)
            
            # C. Update History
            # Append this turn to history so the Generator sees it next time
            interaction_history += f"\nStep {step_count}:\nAction: {action}\nObservation: {observation}\n"
            
            # D. Create Memories (Update ACE Context)
            # Only update if the task isn't finished, or update one last time to capture the final lesson
            if not done:
                print("Updating Memory...")
                current_memory = create_and_update_memory(
                    task_input=task.instruction,
                    last_action=action,
                    observation=observation,
                    current_memory=current_memory,
                    step_count=step_count
                )
                
        # 3. Evaluate Final Result
        score = env.evaluate()
        results.append(score)
        print(f"Task {task_id} Finished. Score: {score}")
        
    return results

```

---

## 5. Notes on ComplexFuncBench Integration

* **Prompting:** You may need to modify `ace/prompts/generator.py` slightly. The default prompt expects to solve a problem in one shot or output a final answer. For `ComplexFuncBench`, you might want to inject a system instruction like: *"You are a tool-using agent. Use the provided memory to guide your next tool selection."*
*

**Observation Formatting:** Ensure the `observation` string passed to `create_and_update_memory` is clear (e.g., "Error: Invalid Argument" or "Success: Output is 42"). The Reflector relies on this to judge success.
