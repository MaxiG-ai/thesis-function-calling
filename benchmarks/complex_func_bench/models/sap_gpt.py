from typing import Any, Optional
import json
import copy
import weave
from benchmarks.complex_func_bench.prompts.prompts import SimpleTemplatePrompt
from benchmarks.complex_func_bench.utils.utils import retry
from src.llm_orchestrator import ClientFactory


class SAPGPTModel:
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        # Use centralized client factory instead of hardcoded connection
        self.client = ClientFactory.create_client()
        

    def __call__(self, prefix, prompt: SimpleTemplatePrompt, **kwargs: Any):
        filled_prompt = prompt(**kwargs)
        prediction = self._predict(prefix, filled_prompt, **kwargs)
        return prediction
    
    @weave.op()
    @retry(max_attempts=10)
    def _predict(self, prefix, text, **kwargs):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
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
    
    def __init__(self, model_name, orchestrator=None):
        """
        Initialize function calling model.
        
        Args:
            model_name: Model identifier
            orchestrator: Optional LLMOrchestrator instance for memory processing
        """
        super().__init__(None)
        self.model_name = model_name
        self.messages = []
        self.orchestrator = orchestrator

    @weave.op()
    @retry(max_attempts=5, delay=10)
    def __call__(self, messages, tools=None, **kwargs: Any):
        # The runner manages self.messages directly by appending assistant/tool messages
        # We should NOT overwrite it here - just use what the runner has built up
        # Only initialize on first call (when self.messages is empty)
        if "function_call" not in json.dumps(messages, ensure_ascii=False):
            self.messages = copy.deepcopy(messages)
        
        try:
            # Route through orchestrator if available (applies memory processing)
            if self.orchestrator is not None:
                response = self.orchestrator.generate(
                    input_messages=self.messages,
                    tools=tools,
                    tool_choice=kwargs.get("tool_choice", "auto"),
                    max_tokens=kwargs.get("max_tokens", 2048)
                )
                return response.choices[0].message
            
            # Fallback to direct client call (for evaluation/comparison)
            else:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages,
                    tools=tools,
                    tool_choice="auto",
                    max_tokens=2048
                )
                return completion.choices[0].message
                
        except Exception as e:
            print(f"Exception: {e}")
            return None


if __name__ == "__main__":
    model = SAPGPTModel("gpt-5-mini")
    response = model("You are a helpful assistant.", SimpleTemplatePrompt(template=("What is the capital of France?"), args_order=[]))
    print(response)
