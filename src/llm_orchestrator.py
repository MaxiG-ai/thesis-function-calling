from typing import List, Dict, Optional, Any, Union, Iterable
import weave
import os
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.chat import ChatCompletion
import litellm
from litellm.files.main import ModelResponse
from src.utils.config import load_configs, ExperimentConfig, ModelDef
from src.memory_processing import MemoryProcessor, get_token_count
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
        os.environ["LITELLM_LOG"] = "ERROR"
        # 1. Load static config
        self.cfg: ExperimentConfig = load_configs(exp_path, model_path)
        
        # 2. Initialize memory processor
        self.memory_processor = MemoryProcessor(self.cfg)
        
        # 3. State variables (mutable)
        self.active_model_key: str = self.cfg.enabled_models[0]
        self.active_memory_key: str = self.cfg.enabled_memory_methods[0]
        self.raw_history: List[Dict] = []
        
        # 4. Configure LiteLLM
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

    def get_raw_history_token_count(self) -> int:
        """
        Calculate token count of the raw conversation history.
        
        Returns:
            Token count of raw history
        """
        return get_token_count(self.raw_history, model="gpt-4-1-mini")
    
    def reset_session(self):
        """
        Clear memory state. Call this before starting a new benchmark conversation.
        """
        self.memory_processor.reset_state()
        self.raw_history = []
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
    
    @weave.op()
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
        if not self.raw_history:
            self.raw_history = list(input_messages)
        else:
            self.raw_history.append(input_messages[-1])

        # Pre-processing metrics
        input_char_count = sum(len(str(m.get("content", ""))) for m in input_messages)
        
        logger.debug(
            f"🔄 Processing {len(input_messages)} messages "
            f"({input_char_count} chars) with {self.active_memory_key}"
        )
        
        #TODO: Needs to be filled correctly
        input_token_count = {
            "conversation_history_token_count": get_token_count(
                input_messages, model="gpt-4-1-mini"
            ),

        }

        # Apply memory processing
        compressed_view, _ = self.memory_processor.apply_strategy(
            input_messages,
            self.active_memory_key,
            input_token_info=input_token_count,
            llm_client=self,
        )
        
        processed_char_count = sum(len(str(m.get("content", ""))) for m in compressed_view)
        compression_ratio = processed_char_count / input_char_count if input_char_count > 0 else 1.0
        
        logger.debug(
            f"📊 Memory processed: {len(compressed_view)} messages "
            f"({processed_char_count} chars, {compression_ratio:.2%} of original)"
        )
        
        # Sanitize kwargs (remove model if passed by benchmark)
        kwargs.pop("model", None)
        
        try:
            model_def = self.get_model_config()
            
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
            
            # Append generated response to raw history
            if type(response) is ModelResponse:
                response_message = response.choices[0]
                # Convert to dict to match input_messages format
                self.raw_history.append(response_message.model_dump(exclude_none=True))
            else:
                logger.warning("⚠️ Response is not of type ChatCompletion; skipping raw history append.")
            return response
            
        except Exception as e:
            logger.error(f"💥 Generation Failed: {str(e)}")
            raise e
        
    @weave.op()
    def generate_plain(
        self,
        input_messages: Iterable[ChatCompletionMessageParam],
        **kwargs,
    ) -> Union[ChatCompletion, Any]:
        """
        Execute LLM request for evaluation. No memory processing applied. Model defaults to GPT-5 

        Args:
            input_messages: Conversation messages (will be processed by memory strategy)
            tools: Available function definitions
            tool_choice: Tool selection strategy ("auto", "required", "none")
            **kwargs: Additional parameters (max_tokens, temperature, etc.)
            
        Returns:
            ChatCompletion response from OpenAI API
            
        Raises:
            Exception: Any errors from OpenAI API (logged to wandb)
        """
        model_key = kwargs.pop("model", "gpt-4-1")
        
        # Try to find in registry
        model_def = self.cfg.model_registry.get(model_key)
        
        if model_def:
            model_name = model_def.litellm_name
            api_base = model_def.api_base
            api_key = model_def.api_key
        else:
            # Fallback: assume model_key is the model name, use default proxy
            model_name = model_key
            api_base = "http://localhost:3030/v1"
            api_key = "THINKTANK"

        create_kwargs = {
            "model": model_name,
            "messages": input_messages,
            "api_base": api_base,
            "api_key": api_key,
            "drop_params": True,
            **kwargs,
        }

        try:
            response = litellm.completion(**create_kwargs)
            return response
        except Exception as e:
            logger.error(f"💥 Generation Failed: {str(e)}")
            raise e
