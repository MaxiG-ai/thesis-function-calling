import json
import logging
from typing import List, Dict
from .tools import TOOLS_SCHEMA, generate_verbose_logs

logger = logging.getLogger("DummyBench")


class DummyBenchmark:
    def __init__(self, orchestrator):
        self.orchestrator = orchestrator
        self.messages: List[Dict] = []
        self.max_turns = 3

    def run(self):
        """
        Executes a conversation designed to overflow context windows.
        """
        print(
            f"\n🧪 Starting Dummy Benchmark [Model: {self.orchestrator.active_model_key}]"
        )

        # 1. Initial User Prompt
        self.messages.append(
            {
                "role": "user",
                "content": "I am debugging a crash. Can you fetch the last 200 lines of logs for the 'payment-gateway'?",
            }
        )

        # Execution Loop
        for turn in range(self.max_turns):
            print(f"\n--- Turn {turn + 1} ---")

            # A. CALL LLM
            try:
                response = self.orchestrator.generate(
                    input_messages=self.messages, 
                    tools=TOOLS_SCHEMA
                )
            except Exception as e:
                logger.error(f"❌ Benchmark Crashed during LLM call: {e}")
                break

            msg = response.choices[0].message
            self.messages.append(msg)  # Add Assistant response to history

            # Check if LLM wants to run the tool
            if msg.tool_calls:
                print(f"🛠️  Model requested tool: {msg.tool_calls[0].function.name}")
                self._handle_tool_calls(msg.tool_calls)
            else:
                print(f"🤖 Model response: {msg.content}")
                # If model stops calling tools, we can end the test
                if "ERROR" in msg.content or "logs" in msg.content:
                    print("✅ Benchmark Goal Reached (Model analyzed the logs)")
                    break

    def _handle_tool_calls(self, tool_calls):
        """
        Executes the tool and appends the MASSIVE result to history.
        """
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            call_id = tool_call.id

            if func_name == "fetch_server_logs":
                print("Running tool... (Generating massive payload)")
                # FORCE massive output regardless of what LLM asked, to stress test memory
                result_content = generate_verbose_logs(
                    service_name=args.get("service_name"),
                    num_lines=500,  # ~15k chars
                )

                print(f"📉 Generated Log Size: {len(result_content)} characters")

                # Appending result to history
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": func_name,
                        "content": result_content,
                    }
                )
