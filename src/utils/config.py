import os
import tomllib as tomli
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError

class ModelDef(BaseModel):
    model_config = {"extra": "allow"}
    
    litellm_name: str
    context_window: int
    provider: str
    api_base: Optional[str] = None 
    api_key: Optional[str] = None


class MemoryDef(BaseModel):
    type: str
    target_summary_length: Optional[int] = None
    auto_compact_threshold: Optional[int] = None

    # New fields for MemoryBank
    embedding_model: Optional[str] = "BAAI/bge-small-en-v1.5"
    top_k: Optional[int] = 3

    # Fields for Progressive Summarization
    summary_prompt: Optional[str] = None
    summarizer_model: Optional[str] = None


class ExperimentConfig(BaseModel):
    experiment_name: str
    results_dir: str
    log_dir: str
    logging_level: str
    input_file: str
    proc_num: int = 1
    benchmark_sample_size: Optional[int]=None
    selected_test_cases: Optional[List[str]] = None
    enabled_models: List[str]
    enabled_memory_methods: List[str]
    compact_threshold: int

    # Maps strategy name -> config
    memory_strategies: Dict[str, MemoryDef]

    # Maps model key -> config (Populated from model_config.toml)
    model_registry: Dict[str, ModelDef] = Field(default_factory=dict)


def load_configs(
    exp_path="config.toml", model_path="model_config.toml"
) -> ExperimentConfig:
    """
    Loads and merges the experiment config with the model registry.
    Also sets the global logging level based on the config.
    """
    if not os.path.exists(exp_path) or not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing config files: {exp_path} or {model_path}")

    with open(model_path, "rb") as f:
        model_data = tomli.load(f)

    with open(exp_path, "rb") as f:
        exp_data = tomli.load(f)

    # Inject registry into experiment config for a single unified object
    # The 'models' key in toml becomes 'model_registry' in Pydantic
    exp_data["model_registry"] = model_data.get("models", {})

    try:
        config = ExperimentConfig(**exp_data)
        
        # Set global logging level from config
        from .logger import set_global_log_level
        set_global_log_level(config.logging_level)
        
        return config
    except ValidationError as e:
        print("❌ Configuration Error:")
        print(e)
        raise
