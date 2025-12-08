"""
Centralized OpenAI client factory.
Single source of truth for API connection configuration.
"""
from openai import OpenAI
from typing import Optional
from src.utils.config import ModelDef
from src.utils.logger import get_logger

logger = get_logger("ClientFactory")


class ClientFactory:
    """
    Factory for creating configured OpenAI clients.
    Provides caching and centralized configuration management.
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
            
        Example:
            # Using model definition from config
            model_def = config.model_registry["gpt-5"]
            client = ClientFactory.create_client(model_def=model_def)
            
            # Using direct parameters
            client = ClientFactory.create_client(
                api_base="http://localhost:4000/v1",
                api_key="my-key"
            )
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
        """
        Clear client cache.
        Useful for testing or when connection parameters change.
        """
        logger.debug("Clearing client cache")
        cls._client_cache.clear()
