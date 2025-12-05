import json
import copy
from benchmarks.complex_func_bench.utils.runner.base_runner import ModelRunner

class OrchestratorRunner(ModelRunner):
    def __init__(self, args, logger, orchestrator):
        super().__init__(args, logger)
        self.orchestrator = orchestrator

    def get_standard_functions(self, functions):
        # Convert CFB function format to OpenAI/LiteLLM format
        return [{"type": "function", "function": copy.deepcopy(func)} for func in functions]

    def get_standard_fc(self, tool_call):
        # Convert LiteLLM tool call back to CFB format for evaluation
        try:
            return {
                "name": tool_call.function.name,
                "arguments": json.loads(tool_call.function.arguments)
            }
        except Exception as e:
            self.logger.error(f"Failed to parse tool call: {e}")
            return None

    def run(self, data):
        """
        Running the actual benchmark.
        """
        # 1. Reset Memory for new conversation
        self.orchestrator.reset_session() 
        
        convs, functions = data['conversations'], data['functions']
        self.CompareClass.add_free_function(convs)
        
        # 1. Setup Initial Context
        messages = []
        query = convs[0]['content']
        messages.append({"role": "user", "content": query})
        
        # Initialize "Golden" (Ground Truth) Tracker
        self.init_golden(convs)

        # 2. Execution Loop
        while True:
            # --- THESIS INTERCEPTION POINT ---
            # We pass the full history to the orchestrator. 
            # It handles Memory/Compression internally before calling the LLM.
            try:
                response = self.orchestrator.generate(
                    input_messages=messages, 
                    tools=self.get_standard_functions(functions)
                )
                llm_message = response.choices[0].message
            except Exception as e:
                return self.return_result(messages, {"error_type": "llm_error", "content": str(e)})

            # 3. Handle Tool Calls
            if llm_message.tool_calls:
                # Safety check: Did we expect the model to stop?
                if self.golden_fcs == []:
                    return self.return_result(messages, {"error_type": "func_hallucination", "content": "Model generated tool calls when none were expected."})

                # Add Assistant Message to History
                messages.append(llm_message.model_dump()) 
                
                # Parse and Validate Tool Calls against Golden Truth
                function_calls = []
                for tool_call in llm_message.tool_calls:
                    fc = self.get_standard_fc(tool_call)
                    if fc: 
                        function_calls.append(fc)

                # CFB Evaluation Logic (Compares prediction vs Ground Truth)
                self.error_message, success_map, success_matched, format_error = self.CompareClass.compare_turn_prediction(
                    functions, messages[:-1], 
                    copy.deepcopy(function_calls), self.golden_fcs, 
                    self.golden_obs
                )

                # If totally wrong, stop
                if not success_map:
                    return self.return_result(messages, self.error_message)
                if len(success_map) == 0 and format_error == {}:
                    return self.return_result(messages, self.error_message)
                
                self.correct_count += len(success_map)

                # 4. Generate Observations
                # CRITICAL: We use the "Golden Observation" (Ground Truth Result) 
                # to ensure the benchmark proceeds correctly, even if the model compressed context.
                real_time_obs = []
                for t, _ in enumerate(function_calls):
                    if t in success_map:
                        temp_obs = success_map[t] # Use Ground Truth
                    elif t in format_error:
                        temp_obs = format_error[t]
                    else:
                        temp_obs = self.unexpect_call_resp
                    
                    real_time_obs.append(temp_obs)
                    
                    # Add Tool Output to History
                    messages.append({
                        "tool_call_id": llm_message.tool_calls[t].id,
                        "role": "tool",
                        "name": function_calls[t]['name'],
                        "content": json.dumps(temp_obs, ensure_ascii=False)
                    })

                self.process_matches(success_matched)

            # 5. Handle Final Text Response
            elif llm_message.content:
                messages.append({"role": "assistant", "content": llm_message.content})
                return self.return_result(messages, self.error_message)
            else:
                return self.return_result(messages, {"error_type": "unknown_error", "content": "Response was empty"})