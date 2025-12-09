from typing import List, Dict, Optional, Any, Union, Iterable
import weave
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.chat import ChatCompletion
from src.utils.config import load_configs, ExperimentConfig, ModelDef
from src.memory_processing import MemoryProcessor
from src.utils.logger import get_logger

logger = get_logger("Orchestrator")


class ClientFactory:
    """
    Factory for creating configured OpenAI clients.
    Integrated into orchestrator for centralized management.
    Kept as separate class for backward compatibility with benchmark code.
    """
    _client_cache = {}
    
    @classmethod
    def create_client(
        cls, 
        model_def: Optional[ModelDef] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None
    ) -> OpenAI:
        """
        Create or retrieve cached OpenAI client.
        
        Args:
            model_def: Model definition from config (preferred method)
            api_base: Override base URL (fallback if no model_def)
            api_key: Override API key (fallback if no model_def)
            
        Returns:
            Configured OpenAI client instance
        """
        # Use model_def if provided, else fallback to direct params
        if model_def:
            base_url = model_def.api_base or "http://localhost:4000/v1"
            key = model_def.api_key or "placeholder-key"
        else:
            base_url = api_base or "http://localhost:4000/v1"
            key = api_key or "placeholder-key"
        
        # Cache key for client reuse
        cache_key = f"{base_url}:{key}"
        
        if cache_key not in cls._client_cache:
            logger.debug(f"Creating new OpenAI client for {base_url}")
            cls._client_cache[cache_key] = OpenAI(
                base_url=base_url,
                api_key=key
            )
        else:
            logger.debug(f"Reusing cached OpenAI client for {base_url}")
        
        return cls._client_cache[cache_key]
    
    @classmethod
    def clear_cache(cls):
        """Clear client cache. Useful for testing or when connection parameters change."""
        logger.debug("Clearing client cache")
        cls._client_cache.clear()


class LLMOrchestrator:
    """
    Centralized LLM interaction manager.
    
    Integrates:
    - Client factory for OpenAI connections
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
        # 1. Load static config
        self.cfg: ExperimentConfig = load_configs(exp_path, model_path)
        
        # 2. Initialize memory processor
        self.memory_processor = MemoryProcessor(self.cfg)
        
        # 3. State variables (mutable)
        self.active_model_key: str = self.cfg.enabled_models[0]
        self.active_memory_key: str = self.cfg.enabled_memory_methods[0]
        
        # 4. Create OpenAI client
        self.client = ClientFactory.create_client()
        
        logger.info(f"🚀 Orchestrator initialized for: {self.cfg.experiment_name}")
    
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

    def get_model_kwargs_from_config(self) -> Dict[str, Any]:
        """
        Retrieve model-specific kwargs from configuration.
        
        Returns:
            Dictionary of model parameters (e.g., temperature) from extra fields
        """
        model_def = self.cfg.model_registry.get(self.active_model_key)
        if not model_def:
            raise ValueError(f"Model '{self.active_model_key}' not found in registry.")
        
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
            **kwargs: Additional parameters (max_tokens, temperature, etc.)
            
        Returns:
            ChatCompletion response from OpenAI API
            
        Raises:
            Exception: Any errors from OpenAI API (logged to wandb)
        """
        # Pre-processing metrics
        input_char_count = sum(len(str(m.get("content", ""))) for m in input_messages)
        
        logger.debug(
            f"🔄 Processing {len(input_messages)} messages "
            f"({input_char_count} chars) with {self.active_memory_key}"
        )
        
        # Apply memory processing
        messages = self.memory_processor.apply_strategy(
            input_messages, self.active_memory_key
        )
        
        processed_char_count = sum(len(str(m.get("content", ""))) for m in messages)
        compression_ratio = processed_char_count / input_char_count if input_char_count > 0 else 1.0
        
        logger.debug(
            f"📊 Memory processed: {len(messages)} messages "
            f"({processed_char_count} chars, {compression_ratio:.2%} of original)"
        )
        
        # Sanitize kwargs (remove model if passed by benchmark)
        kwargs.pop("model", None)
        
        try:
            # Build request parameters
            request_params = {
                "model": self.active_model_key,
                "messages": messages,
                **self.get_model_kwargs_from_config(),
            }
            
            # Only add tools and tool_choice if provided
            if tools is not None:
                request_params["tools"] = tools
            if tool_choice is not None:
                request_params["tool_choice"] = tool_choice
            
            response = self.client.chat.completions.create(**request_params)
            
            # Post-call metrics
            response_message = response.choices[0].message
            tool_call_count = len(response_message.tool_calls) if response_message.tool_calls else 0
            
            logger.debug(
                "✅ LLM call completed."
                f"(finish: {response.choices[0].finish_reason}, "
                f"tool_calls: {tool_call_count})"
            )
            
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
        # Sanitize kwargs (remove model if passed by benchmark)
        kwargs.pop("model", None)
        
        try:
            if "tools" in kwargs:
                tools = kwargs.get("tools", None)
                tool_choice = kwargs.get("tool_choice", None)
                create_kwargs = {
                    "model": "gpt-5",
                    "messages": input_messages,
                }
                if tools is not None:
                    create_kwargs["tools"] = tools
                if tool_choice is not None:
                    create_kwargs["tool_choice"] = tool_choice
                response = self.client.chat.completions.create(**create_kwargs)
            else:
                response = self.client.chat.completions.create(
                    model="gpt-5",
                    messages=input_messages,
                )
            
            return response
            
        except Exception as e:
            logger.error(f"💥 Generation Failed: {str(e)}")
            raise e
