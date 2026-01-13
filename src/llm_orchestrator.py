import litellm
import weave
import os

from typing import List, Dict, Optional, Any, Union, Iterable
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam, ChatCompletion
from src.utils.config import load_configs, ExperimentConfig, ModelDef
from src.memory_processing import MemoryProcessor
from src.utils.token_count import get_token_count
from src.utils.logger import get_logger

logger = get_logger("Orchestrator")

class LLMOrchestrator:
    """
    Centralized LLM interaction manager.
    
    Integrates:
    - LiteLLM for model interactions
    - Memory processing for context optimization
    - Comprehensive tracking with weave/wandb
    
    Usage:
        # Initialize
        orchestrator = LLMOrchestrator()
        orchestrator.set_active_context("gpt-5", "memory_bank")
        
        # Direct use
        response = orchestrator.generate(messages, tools=tools)
        
        # Benchmark integration (inject into model)
        runner = SAPGPTRunner(
            model_name="gpt-5",
            args=args,
            logger=logger,
            orchestrator=orchestrator  # Memory processing automatically applied
        )
    """
    
    def __init__(self, exp_path="config.toml", model_path="model_config.toml"):
        """
        Initialize orchestrator with configuration.
        
        Args:
            exp_path: Path to experiment config file
            model_path: Path to model registry config file
        """
        # Load static config
        self.cfg: ExperimentConfig = load_configs(exp_path, model_path)
        
        # Initialize memory processor
        self.memory_processor = MemoryProcessor(self.cfg)
        
        # State variables (mutable)
        self.active_model_key: str = self.cfg.enabled_models[0]
        self.active_memory_key: str = self.cfg.enabled_memory_methods[0]
        
        # Configure LiteLLM
        os.environ["LITELLM_LOG"] = "ERROR"
        litellm.suppress_debug_info = True
        litellm.success_callback = ["weave"]
        
        logger.info(f"🚀 Orchestrator initialized for: {self.cfg.experiment_name}")

    def get_exp_config(self) -> Dict[str, Any]:
        """
        Return only experiment configs (no model registry). Used for logging with weave.
        """
        exp_dict = self.cfg.model_dump()
        exp_dict.pop("model_registry", None)
        return exp_dict

    def reset_session(self):
        """
        Clear memory state. Call this before starting a new benchmark conversation.
        """
        self.memory_processor.reset_state()
        logger.info("🔄 Session Reset")
    
    def set_active_context(self, model_key: str, memory_key: str):
        """
        HOTSWAP: Updates the active configuration for the next request.
        Call this inside your experiment loop to switch configurations.
        
        Args:
            model_key: Model identifier from model_config.toml
            memory_key: Memory strategy from config.toml
            
        Raises:
            ValueError: If model or memory strategy not found
        """
        if model_key not in self.cfg.model_registry:
            raise ValueError(f"Model '{model_key}' not found in registry.")
        if memory_key not in self.cfg.memory_strategies:
            raise ValueError(f"Memory strategy '{memory_key}' not defined.")
        
        self.active_model_key = model_key
        self.active_memory_key = memory_key
        
        logger.info("🔄 Context Switched")

    def get_model_config(self) -> ModelDef:
        """Helper to get current model config"""
        model_def = self.cfg.model_registry.get(self.active_model_key)
        if not model_def:
            raise ValueError(f"Model '{self.active_model_key}' not found in registry.")
        return model_def

    def get_model_kwargs_from_config(self) -> Dict[str, Any]:
        """
        Retrieve model-specific kwargs from configuration.
        
        Returns:
            Dictionary of model parameters (e.g., temperature) from extra fields
        """
        model_def = self.get_model_config()
        
        # Get all fields from the model
        all_fields = model_def.model_dump()
        
        # Define the base/required fields that should not be passed as kwargs
        base_fields = {
            "litellm_name", 
            "context_window", 
            "provider", 
            "api_base", 
            "api_key"
        }
        
        # Extract only the extra fields (kwargs)
        model_kwargs = {
            key: value 
            for key, value in all_fields.items() 
            if key not in base_fields and value is not None
        }
        
        return model_kwargs
    
    @weave.op(
            postprocess_inputs=lambda inputs: {k: v for k, v in inputs.items() if k not in ["input_messages"]},
            enable_code_capture=False,
    )
    def generate_with_memory_applied(
        self,
        input_messages: List[Dict[str, str]],
        tools: Optional[List[ChatCompletionToolParam]] = None,
        tool_choice: Optional[str] = "auto",
        **kwargs,
    ) -> Union[ChatCompletion, Any]:
        """
        Execute LLM request with memory processing and comprehensive tracking.
        
        This is the core method that:
        1. Logs pre-processing metrics
        2. Applies memory strategy via memory processor
        3. Executes LLM call with timing
        4. Logs post-processing metrics
        5. Handles errors with tracking
        
        Args:
            input_messages: Conversation messages (will be processed by memory strategy)
            tools: Available function definitions
            tool_choice: Tool selection strategy ("auto", "required", "none")
            model: Optional override for the model used in this plain request.
            **kwargs: Additional parameters (max_tokens, etc.)
            
        Returns:
            ChatCompletion response from OpenAI API
            
        Raises:
            Exception: Any errors from OpenAI API (logged to wandb)
        """
        # Sync raw history (append only new messages)
        # TODO: Feature: Keep complete trace before truncation.

        logger.debug(
            f"🔄 Processing {len(input_messages)} messages with {self.active_memory_key}"
        )

        model_def = self.get_model_config()
        input_token_count = get_token_count(input_messages, model_name=model_def.litellm_name)

        # Apply memory processing
        compressed_view, _ = self.memory_processor.apply_strategy(
            input_messages,
            self.active_memory_key,
            input_token_count=input_token_count,
            llm_client=self,
        )
        
        # Sanitize kwargs (remove model if passed by benchmark)
        kwargs.pop("model", None)
        
        try:
            # Build request parameters
            request_params = {
                "model": model_def.litellm_name,
                "messages": compressed_view,
                "api_base": model_def.api_base,
                "api_key": model_def.api_key,
                "drop_params": True,
            }
            
            # Merge config kwargs
            request_params.update(self.get_model_kwargs_from_config())
            
            # Merge runtime kwargs (overrides config)
            request_params.update(kwargs)
            
            # Only add tools and tool_choice if provided
            if tools is not None:
                request_params["tools"] = tools
            if tool_choice is not None:
                request_params["tool_choice"] = tool_choice
            
            response = litellm.completion(**request_params)
            
            return response
            
        except Exception as e:
            logger.error(f"💥 Generation Failed: {str(e)}")
            raise e
        
    @weave.op(
            enable_code_capture=False,
    )
    def generate_plain(
        self,
        input_messages: Iterable[ChatCompletionMessageParam],
        **kwargs,
    ) -> Union[ChatCompletion, Any]:
        """
        Execute LLM request for evaluation. No memory processing applied. Model defaults to GPT-4.1 
        Exception: Any errors from OpenAI API (logged to wandb)
        """
        kwargs.pop("model", None)
        try:
            
            # Try to find in registry
            model_def = self.get_model_config()
            
            request_params = {
                "model": model_def.litellm_name,
                "messages": input_messages,
                "api_base": model_def.api_base,
                "api_key": model_def.api_key,
                "drop_params": True,
                **kwargs,
            }

            response = litellm.completion(**request_params)
            return response
        
        except Exception as e:
            logger.error(f"💥 Generation Failed: {str(e)}")
            raise e
