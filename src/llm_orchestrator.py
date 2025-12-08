from typing import List, Dict, Optional, Any, Union
from openai import OpenAI
from openai.types.chat import ChatCompletion
from src.utils.config import load_configs, ExperimentConfig, ModelDef, MemoryDef
from src.memory_processing import MemoryProcessor
from src.utils.logger import get_logger
from src.utils.client_factory import ClientFactory

logger = get_logger("Orchestrator")


class LLMOrchestrator:
    def __init__(self, exp_path="config.toml", model_path="model_config.toml"):
        # 1. Load static config
        self.cfg: ExperimentConfig = load_configs(exp_path, model_path)

        self.memory_processor = MemoryProcessor(self.cfg)
        # 2. State variables (mutable)
        self.active_model_key: str = self.cfg.enabled_models[0]
        self.active_memory_key: str = self.cfg.enabled_memory_methods[0]

        # 3. Initialize OpenAI Client using centralized factory
        # Client configuration now comes from model_config.toml via ClientFactory
        self.client = ClientFactory.create_client()
        
        # Set initial model for memory processor
        self.memory_processor.set_current_model(self.active_model_key)

        logger.info(f"🚀 Orchestrator initialized for: {self.cfg.experiment_name}")
        self._log_active_state()

    def reset_session(self):
        """
        Clears memory state. Call this before starting a new benchmark conversation.
        """
        self.memory_processor.reset_state()
        logger.info("🔄 Session Reset")

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
        
        # Notify memory processor of the active model for context window calculations
        self.memory_processor.set_current_model(model_key)

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
        input_messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = "auto",
        **kwargs,
    ) -> Union[ChatCompletion, Any]:
        """
        Executes the request using the CURRENTLY ACTIVE model.
        """
        # Pass through the processor before sending to LLM
        
        messages = self.memory_processor.apply_strategy(
            input_messages, self.active_memory_key
        )

        # 1. Get the definition for the active model
        model_def = self._get_active_model_def()
        target_litellm_name = model_def.litellm_name

        # 👇 Prepare dynamic connection args
        connection_args = {}
        if model_def.api_base:
            connection_args["api_base"] = model_def.api_base
        if model_def.api_key:
            connection_args["api_key"] = model_def.api_key

        # 2. SANITIZATION (remove model from kwargs to avoid conflicts, when benchmarks sends model):
        kwargs.pop("model", None)

        try:
            # 3. Execute via OpenAI SDK
            # We simply pass the model name (e.g. "gpt-5-mini"). 
            response = self.client.chat.completions.create(
                model=self.active_model_key,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                # **kwargs,
            )
            return response

        except Exception as e:
            logger.error(f"💥 Generation Failed on {target_litellm_name}: {str(e)}")
            raise e
