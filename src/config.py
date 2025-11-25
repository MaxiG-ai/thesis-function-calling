import os
import tomllib as tomli
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ValidationError

class ModelDef(BaseModel):
    litellm_name: str
    context_window: int
    provider: str
    api_base: Optional[str] = None 
    api_key: Optional[str] = None


class MemoryDef(BaseModel):
    type: str
    max_tokens: Optional[int] = None
    target_summary_length: Optional[int] = None
    summarizer_model: Optional[str] = None


class ExperimentConfig(BaseModel):
    experiment_name: str
    results_dir: str
    enabled_models: List[str]
    enabled_memory_methods: List[str]

    # Maps strategy name -> config
    memory_strategies: Dict[str, MemoryDef]

    # Maps model key -> config (Populated from model_config.toml)
    model_registry: Dict[str, ModelDef] = Field(default_factory=dict)


def load_configs(
    exp_path="config.toml", model_path="model_config.toml"
) -> ExperimentConfig:
    """
    Loads and merges the experiment config with the model registry.
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
        return ExperimentConfig(**exp_data)
    except ValidationError as e:
        print("❌ Configuration Error:")
        print(e)
        raise
