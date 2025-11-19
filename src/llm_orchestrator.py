import logging
from typing import List, Dict, Optional, Any, Union
from litellm import completion, ModelResponse
from .config import load_configs, ExperimentConfig, ModelDef, MemoryDef

# Configure Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Orchestrator")


class LLMOrchestrator:
    def __init__(self, exp_path="config.toml", model_path="model_config.toml"):
        # 1. Load static config
        self.cfg: ExperimentConfig = load_configs(exp_path, model_path)

        # 2. State variables (mutable)
        self.active_model_key: str = self.cfg.enabled_models[0]
        self.active_memory_key: str = self.cfg.enabled_memory_methods[0]

        logger.info(f"🚀 Orchestrator initialized for: {self.cfg.experiment_name}")
        self._log_active_state()

    def set_active_context(self, model_key: str, memory_key: str):
        """
        HOTSWAP: Updates the active configuration for the next request.
        Call this inside your experiment loop.
        """
        if model_key not in self.cfg.model_registry:
            raise ValueError(f"Model '{model_key}' not found in registry.")
        if memory_key not in self.cfg.memory_strategies:
            raise ValueError(f"Memory strategy '{memory_key}' not defined.")

        self.active_model_key = model_key
        self.active_memory_key = memory_key

        logger.info("🔄 Context Switched")
        self._log_active_state()

    def _get_active_model_def(self) -> ModelDef:
        return self.cfg.model_registry[self.active_model_key]

    def get_active_memory_config(self) -> MemoryDef:
        """Used by the Middleware to know how to compress context"""
        return self.cfg.memory_strategies[self.active_memory_key]

    def _log_active_state(self):
        m = self._get_active_model_def()
        logger.info(
            f"👉 Active: [Model: {self.active_model_key}] [Memory: {self.active_memory_key}]"
        )
        logger.info(f"   Target: {m.litellm_name}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = "auto",
        **kwargs,
    ) -> Union[ModelResponse, Any]:
        """
        Executes the request using the CURRENTLY ACTIVE model.
        """
        # 1. Get the definition for the active model
        model_def = self._get_active_model_def()
        target_litellm_name = model_def.litellm_name

        # 2. SANITIZATION:
        # Benchmarks (like MCP-Bench) often send 'model="gpt-4"' hardcoded.
        # We MUST override this to ensure we test the model we intend to test.
        kwargs.pop("model", None)

        try:
            # 3. Execute via LiteLLM
            response = completion(
                model=target_litellm_name,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                drop_params=True,  # Crucial: Ignores unsupported OpenAI params sent by benchmarks
                **kwargs,
            )
            return response

        except Exception as e:
            logger.error(f"💥 Generation Failed on {target_litellm_name}: {str(e)}")
            raise e
