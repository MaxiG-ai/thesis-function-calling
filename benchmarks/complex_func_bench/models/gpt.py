from typing import Any
import os
from openai import OpenAI
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.complex_func_bench.prompts.prompts import SimpleTemplatePrompt
from benchmarks.complex_func_bench.utils.utils import retry


class GPTModel:
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.client = OpenAI(base_url="http://localhost:3030/v1", api_key=os.getenv("OPENAI_API_KEY", "just-a-placeholder-key"))
        

    def __call__(self, prefix, prompt: SimpleTemplatePrompt, **kwargs: Any):
        filled_prompt = prompt(**kwargs)
        prediction = self._predict(prefix, filled_prompt, **kwargs)
        return prediction
    
    @retry(max_attempts=10)
    def _predict(self, prefix, text, **kwargs):
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prefix},
                    {"role": "user", "content": text}
                ],
                # temperature=0.0, #disabled, because gpt-5-mini doesn't support it
                )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"Exception: {e}")
            return None


if __name__ == "__main__":
    model = GPTModel("gpt-4")
    response = model("You are a helpful assistant.", SimpleTemplatePrompt(template=("What is the capital of France?"), args_order=[]))
    print(response)