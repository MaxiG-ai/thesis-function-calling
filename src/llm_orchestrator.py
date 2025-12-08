from typing import List, Dict, Optional, Any, Union
import time
import weave
import wandb
from openai import OpenAI
from openai.types.chat import ChatCompletion
from src.utils.config import load_configs, ExperimentConfig, ModelDef, MemoryDef
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
        
        # 5. Set initial model for memory processor
        self.memory_processor.set_current_model(self.active_model_key)
        
        logger.info(f"🚀 Orchestrator initialized for: {self.cfg.experiment_name}")
        self._log_active_state()
    
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
        
        # Notify memory processor of the active model for context window calculations
        self.memory_processor.set_current_model(model_key)
        
        logger.info("🔄 Context Switched")
        self._log_active_state()
    
    def _get_active_model_def(self) -> ModelDef:
        """Get model definition for currently active model."""
        return self.cfg.model_registry[self.active_model_key]
    
    def get_active_memory_config(self) -> MemoryDef:
        """
        Get memory configuration for currently active strategy.
        Used by middleware to understand how to compress context.
        """
        return self.cfg.memory_strategies[self.active_memory_key]
    
    def _log_active_state(self):
        """Log current active configuration."""
        m = self._get_active_model_def()
        logger.info(
            f"👉 Active: [Model: {self.active_model_key}] [Memory: {self.active_memory_key}]"
        )
        logger.info(f"   Target: {m.litellm_name}")
    
    @weave.op()
    def generate(
        self,
        input_messages: List[Dict[str, str]],
        tools: Optional[List[Dict]] = None,
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
        model_def = self._get_active_model_def()
        
        # Pre-processing metrics
        input_char_count = sum(len(str(m.get("content", ""))) for m in input_messages)
        
        wandb.log({
            "llm/call_timestamp": time.time(),
            "llm/model": self.active_model_key,
            "llm/memory_strategy": self.active_memory_key,
            "llm/input_messages": len(input_messages),
            "llm/input_chars": input_char_count,
            "llm/has_tools": tools is not None,
            "llm/tool_count": len(tools) if tools else 0,
        })
        
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
        
        wandb.log({
            "llm/processed_messages": len(messages),
            "llm/processed_chars": processed_char_count,
            "llm/compression_ratio": compression_ratio,
        })
        
        logger.debug(
            f"📊 Memory processed: {len(messages)} messages "
            f"({processed_char_count} chars, {compression_ratio:.2%} of original)"
        )
        
        # Sanitize kwargs (remove model if passed by benchmark)
        kwargs.pop("model", None)
        
        # Execute with timing
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.active_model_key,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
            )
            
            duration = time.time() - start_time
            
            # Post-call metrics
            message = response.choices[0].message
            tool_call_count = len(message.tool_calls) if message.tool_calls else 0
            
            wandb.log({
                "llm/duration_seconds": duration,
                "llm/finish_reason": response.choices[0].finish_reason,
                "llm/has_tool_calls": message.tool_calls is not None,
                "llm/tool_call_count": tool_call_count,
                "llm/has_content": message.content is not None,
            })
            
            logger.debug(
                f"✅ LLM call completed in {duration:.2f}s "
                f"(finish: {response.choices[0].finish_reason}, "
                f"tool_calls: {tool_call_count})"
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"💥 Generation Failed after {duration:.2f}s: {str(e)}")
            
            wandb.log({
                "llm/error": str(e),
                "llm/error_type": type(e).__name__,
                "llm/duration_seconds": duration,
            })
            
            raise e
