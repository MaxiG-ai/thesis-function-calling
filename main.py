import time
import logging
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union

# Import your custom modules
from src.llm_orchestrator import LLMOrchestrator
from src.memory_processing import MemoryProcessor

# --- Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Proxy")

app = FastAPI(title="Thesis Memory Proxy", version="0.1.0")

# Global Singletons (Loaded on Startup)
orchestrator: LLMOrchestrator = None
memory_processor: MemoryProcessor = None


@app.on_event("startup")
async def startup_event():
    global orchestrator, memory_processor
    # Initialize the Orchestrator (loads env vars and configs)
    orchestrator = LLMOrchestrator()
    memory_processor = MemoryProcessor(orchestrator.cfg)
    logger.info(
        f"✅ Proxy Ready. Active Context: {orchestrator.active_model_key} / {orchestrator.active_memory_key}"
    )


# --- API Models (Standard OpenAI Schema) ---
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict], None]
    name: Optional[str] = None
    tool_calls: Optional[List] = None
    tool_call_id: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str  # We ignore this, but it's required by the client
    messages: List[ChatMessage]
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[Union[str, Dict]] = "auto"
    stream: Optional[bool] = False
    # Catch-all for other params (temperature, etc.)
    extra_body: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        extra = "allow"  # Allow unknown fields like 'temperature'


# --- Core Endpoint ---
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    Intercepts the chat completion request, applies memory strategy,
    and forwards to the active LLM.
    """
    try:
        # 1. Parse Request (We use raw Request to be flexible with extra params)
        body = await request.json()

        # Validation (Basic)
        messages_raw = body.get("messages", [])
        if not messages_raw:
            raise HTTPException(
                status_code=400, detail="Messages list cannot be empty."
            )

        # 2. 🧠 MEMORY INTERVENTION 🧠
        # This is the core of your thesis.
        optimized_messages = memory_processor.apply_strategy(
            messages=messages_raw, strategy_key=orchestrator.active_memory_key
        )

        # 3. Prepare Arguments for LiteLLM
        # We strip 'model' from body because Orchestrator decides the model
        api_args = {k: v for k, v in body.items() if k not in ["messages", "model"]}

        # 4. Generate Response
        start_time = time.time()

        if body.get("stream", False):
            return await _handle_streaming(optimized_messages, api_args)
        else:
            return await _handle_synchronous(optimized_messages, api_args, start_time)

    except Exception as e:
        logger.error(f"🔥 Proxy Error: {str(e)}")
        # Return 500 so the Benchmark knows something went wrong
        raise HTTPException(status_code=500, detail=str(e))


# --- Helpers ---


async def _handle_synchronous(messages, api_args, start_time):
    """Handles standard Request-Response"""
    response = orchestrator.generate(messages=messages, **api_args)

    # Add Thesis Metrics header (Latency)
    duration = time.time() - start_time

    # Convert LiteLLM object to Dict if needed, or return JSON
    # LiteLLM responses are usually Pydantic objects
    return JSONResponse(content=response.model_dump())


async def _handle_streaming(messages, api_args):
    """Handles Streaming (Server-Sent Events)"""

    # Get the generator from LiteLLM
    response_stream = orchestrator.generate(messages=messages, **api_args)

    async def stream_generator():
        # Iterate through the sync generator from LiteLLM
        # In production with async litellm, this would be 'async for'
        for chunk in response_stream:
            # Convert chunk to JSON string compatible with OpenAI stream format
            data = json.dumps(chunk.model_dump())
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


@app.get("/health")
async def health_check():
    return {
        "status": "active",
        "model": orchestrator.active_model_key,
        "memory": orchestrator.active_memory_key,
    }
def main():
    print("Hello from thesis-function-calling!")

# Conceptual example of your experiment runner
orchestrator = LLMOrchestrator()

if __name__ == "__main__":
    main()
    
    for model in orchestrator.cfg.enabled_models:
        for memory in orchestrator.cfg.enabled_memory_methods:
            # 1. Switch the Proxy State
            orchestrator.set_active_context(model, memory)

            # 2. Run the Benchmark (which hits your Proxy API)
            # run_benchmark_process(...)
