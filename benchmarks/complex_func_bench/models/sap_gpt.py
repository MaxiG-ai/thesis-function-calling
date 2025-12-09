from typing import Any
import json
import copy
import weave
from benchmarks.complex_func_bench.prompts.prompts import SimpleTemplatePrompt
from benchmarks.complex_func_bench.utils.utils import retry
from src.llm_orchestrator import LLMOrchestrator


class SAPGPTModel:
    def __init__(self, orchestrator:LLMOrchestrator):
        super().__init__()
        self.model_name = orchestrator.active_model_key
        # Use centralized client factory instead of hardcoded connection
        self.orchestrator = orchestrator
        

    def __call__(self, prefix, prompt: SimpleTemplatePrompt, **kwargs: Any):
        filled_prompt = prompt(**kwargs)
        prediction = self._predict(prefix, filled_prompt, **kwargs)
        return prediction
    
    @weave.op()
    @retry(max_attempts=10)
    def _predict(self, prefix, text, **kwargs):
        try:
            completion = self.orchestrator.generate_plain(
                model=self.model_name,
                input_messages=[
                    {"role": "system", "content": prefix},
                    {"role": "user", "content": text}
                ],
                )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Exception: {e}")
            return None


class FunctionCallSAPGPT(SAPGPTModel):
    """
    Function calling model that can optionally use LLMOrchestrator for memory processing.
    
    When orchestrator is provided, all calls are routed through it with memory techniques applied.
    When orchestrator is None, falls back to direct client calls (for evaluation/comparison).
    """
    
    def __init__(self, model_name, orchestrator:LLMOrchestrator):
        """
        Initialize function calling model.
        
        Args:
            model_name: Model identifier
            orchestrator: Optional LLMOrchestrator instance for memory processing
        """
        super().__init__(orchestrator=orchestrator)
        self.model_name = orchestrator.active_model_key
        self.messages = []
        self.orchestrator = orchestrator

    @weave.op()
    @retry(max_attempts=5, delay=10)
    def generate_response(self, messages, tools=None, **kwargs: Any):
        # The runner manages self.messages directly by appending assistant/tool messages
        # We should NOT overwrite it here - just use what the runner has built up
        # Only initialize on first call (when self.messages is empty)
        if "function_call" not in json.dumps(messages, ensure_ascii=False):
            self.messages = copy.deepcopy(messages)
        
        try:
            # Route through orchestrator if available (applies memory processing)
            response = self.orchestrator.generate_with_memory_applied(
                input_messages=self.messages,
                tools=tools,
                tool_choice=kwargs.get("tool_choice", "auto"),
                max_tokens=kwargs.get("max_tokens", 2048)
            )
            return response.choices[0].message
            
        except Exception as e:
            print(f"Exception: {e}")
            return None


if __name__ == "__main__":
    llmo = LLMOrchestrator()
    model = SAPGPTModel(llmo)
    response = model("You are a helpful assistant.", SimpleTemplatePrompt(template=("What is the capital of France?"), args_order=[]))
    print(response)
