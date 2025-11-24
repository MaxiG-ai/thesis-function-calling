import requests
import json
import os
from types import SimpleNamespace
from typing import Any


class LocalAPIModel:
    def __init__(self, model_name, api_base="http://localhost:8000/v1"):
        self.model_name = model_name
        self.api_base = api_base
        self.headers = {
            "Content-Type": "application/json",
            # Add an API key if your proxy requires it
            "Authorization": f"Bearer {os.getenv('PROXY_API_KEY', 'dummy-key')}",
        }

    def __call__(self, messages, tools=None, **kwargs: Any):
        """
        Mimics the __call__ signature of FunctionCallGPT.
        Returns a SimpleNamespace matching OpenAI's 'message' object.
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "tools": tools,
            "temperature": 0.0,
            "max_tokens": 2048,
            "tool_choice": "auto" if tools else None,
        }

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions", headers=self.headers, json=payload
            )
            response.raise_for_status()
            resp_json = response.json()

            return self._adapt_response(resp_json)

        except Exception as e:
            print(f"❌ Proxy Request Failed: {e}")
            return None

    def _adapt_response(self, response_json):
        """
        Converts raw JSON to the object structure expected by GPTRunner.
        GPTRunner accesses: response.tool_calls and response.content
        """
        choice = response_json.get("choices", [{}])[0].get("message", {})

        content = choice.get("content")
        tool_calls_data = choice.get("tool_calls")

        tool_calls_objects = None
        if tool_calls_data:
            tool_calls_objects = []
            for tc in tool_calls_data:
                # ComplexFuncBench expects objects, not dicts, for tool_calls
                function_obj = SimpleNamespace(
                    name=tc["function"]["name"], arguments=tc["function"]["arguments"]
                )
                tool_calls_objects.append(
                    SimpleNamespace(
                        id=tc.get("id", "call_id"),
                        function=function_obj,
                        type="function",
                    )
                )

        return SimpleNamespace(content=content, tool_calls=tool_calls_objects)
