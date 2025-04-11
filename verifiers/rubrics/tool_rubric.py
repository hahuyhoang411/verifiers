import json
from typing import List, Dict, Callable

from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
from verifiers.rubrics.math_grader import grade

class ToolRubric(Rubric):
    def __init__(self,
                 parser: XMLParser = XMLParser(fields=["reasoning", ("tool", "answer")]),
                 env_parser: XMLParser = XMLParser(fields=["result"]),
                 tools: List[Callable] = []):
        self.parser = parser
        self.env_parser = env_parser
        self.tools = {tool.__name__: tool for tool in tools}
        self.reward_funcs = [
            self.mc_reward_func,
            self.math_reward_func,
            self.code_reward_func,
            self.qa_reward_func,
            self.correct_answer_reward_func,
            self.no_tool_bonus_reward_func, # Answer correctly if no tools 
            self.tool_execution_reward_func,
            self.parser.get_format_reward_func(),
            self.parser.get_xml_reward_func(),
        ]
        self.reward_weights = [
            0.0,  # mc
            0.0,  # math
            0.0,  # code
            0.0,  # qa
            1.0,  # correct_answer
            0.5,  # no_tool_bonus
            0.5,  # tool_execution
            0.25, # format
            0.25, # xml
        ]
        for tool_name in self.tools.keys():
            self.reward_funcs.append(self.get_named_tool_reward_func(tool_name))
            self.reward_weights.append(0.0)
            self.reward_funcs.append(self.get_named_tool_count_reward_func(tool_name))
            self.reward_weights.append(0.0)
            self.reward_funcs.append(self.get_named_tool_attempt_reward_func(tool_name))
            self.reward_weights.append(0.0)

    def evaluate_code(self, code_str, answer, **kwargs) -> float:
        import io
        import sys
        import signal
        from contextlib import redirect_stdout
        
        try:
            test_cases = json.loads(answer)['test_cases']
        except:
            return 0.0
        # strip ```python and ``` if present at the beginning and end of the code
        code_str = code_str.strip()
        if code_str.startswith('```python'):
            code_str = code_str[9:]
        elif code_str.startswith('```'):
            code_str = code_str[3:]
        if code_str.endswith('```'):
            code_str = code_str[:-3]
        code_str = code_str.strip()

        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timed out")

        def normalize_output(output):
            # Normalize line endings and whitespace
            return '\n'.join(line.strip() for line in output.splitlines())
        
        total_cases = 0
        passed = 0
        
        for test in test_cases:
            output = io.StringIO()
            sys.stdin = io.StringIO(test['input'])
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(10)
                with redirect_stdout(output):
                    exec(code_str)
                signal.alarm(0)
                actual = normalize_output(output.getvalue())
                expected = normalize_output(test['output'])
                
                # Compare each line individually
                actual_lines = actual.splitlines()
                expected_lines = expected.splitlines()
                total_cases += len(expected_lines)
                for a, e in zip(actual_lines, expected_lines):
                    if a == e:
                        passed += 1
                    
            except Exception as e:
                sys.stdin = sys.__stdin__
                return 0.0
            sys.stdin = sys.__stdin__
        
        return passed / total_cases if total_cases else 0.0
        

    def code_reward_func(self, completions, answer, task, **kwargs) -> List[float | None]:
        """Reward function that checks if the final answer matches the expected answer."""
        rewards = []
        for completion, ans, t in zip(completions, answer, task):
            if t == "code":
                response = str(self.get_last_answer(completion))
                reward = self.evaluate_code(response, ans, **kwargs)
            else:
                reward = None
            rewards.append(reward)
        return rewards
    
    def mc_reward_func(self, completions, answer, task, **kwargs) -> List[float | None]:
        """Reward function that checks if the final answer matches the expected answer."""
        rewards = []
        for completion, ans, t in zip(completions, answer, task):
            if t == "mc":
                response = str(self.get_last_answer(completion)) #[0]
                if len(response.strip()) > 0 and isinstance(response, str):
                    response = response.strip()[0]
                reward = 1.0 if response == ans.strip() else 0.0
            else:
                reward = None
            rewards.append(reward)
        return rewards

    def math_reward_func(self, completions, answer, task, **kwargs) -> List[float | None]:
        """Reward function that checks if the final answer matches the expected answer."""
        rewards = []
        for completion, ans, t in zip(completions, answer, task):
            if t == "math":
                response = str(self.get_last_answer(completion))
                try:
                    reward = 1.0 if grade(response, ans) else 0.0
                except:
                    reward = 0.0
            else:
                reward = None
            rewards.append(reward)
        return rewards

    def qa_reward_func(self, completions, answer, task, **kwargs) -> List[float | None]:
        """Reward function for simple QA - checks for exact or near-exact match."""
        rewards = []
        for completion, ans, t in zip(completions, answer, task):
            if t == "qa": # Use your chosen task tag
                response = self.get_last_answer(completion)
                if response is None:
                    reward = 0.0
                else:
                    # Implement your desired comparison logic here:
                    # Option A: Simple exact match (case-insensitive)
                    # reward = 1.0 if str(response).strip().lower() == str(ans).strip().lower() else 0.0
                    # Option B: Check if answer is substring (case-insensitive)
                    reward = 1.0 if str(ans).strip().lower() in str(response).strip().lower() else 0.0
                    # Add more sophisticated logic if needed
            else:
                reward = None # Important: Return None if the task doesn't match
            rewards.append(reward)
        return rewards
        
    def correct_answer_reward_func(self, completions, answer, task, **kwargs) -> List[float | None]:
        """Reward function that checks if the final answer matches the expected answer."""
        rewards = []
        for completion, ans, t in zip(completions, answer, task):
            reward = None
            if t == "mc":
                try:
                    reward = self.mc_reward_func([completion], [ans], [t], **kwargs)[0]
                except:
                    reward = None
            elif t == "math":
                try:
                    reward = self.math_reward_func([completion], [ans], [t], **kwargs)[0]
                except:
                    reward = None
            elif t == "code":
                try:
                    reward = self.code_reward_func([completion], [ans], [t], **kwargs)[0]
                except:
                    reward = None
            elif t == "qa":
                try:
                    reward = self.qa_reward_func([completion], [ans], [t], **kwargs)[0]
                except:
                    reward = None
            else:
                reward = None
            rewards.append(reward)
        return rewards

    def no_tool_bonus_reward_func(self, completions: List[List[Dict[str, str]]], answer: List[str], task: List[str], **kwargs) -> List[float]:
        """
        Reward function that gives a bonus if the correct answer is achieved without successful tool use.
        """
        rewards = []
        correctness_scores = self.correct_answer_reward_func(completions, answer, task, **kwargs)

        for i, trajectory in enumerate(completions):
            correctness = correctness_scores[i]
            tool_successfully_used = False

            # Check for successful tool execution in the trajectory
            for j, msg in enumerate(trajectory):
                if msg['role'] == 'assistant':
                    parsed = self.parser.parse(msg['content'])
                    if hasattr(parsed, 'tool') and parsed.tool is not None:
                        # Found a tool call, check if it was successful
                        if j + 1 < len(trajectory) and trajectory[j + 1]['role'] == 'user':
                            parsed_response = self.env_parser.parse(trajectory[j + 1]['content'])
                            if hasattr(parsed_response, 'result') and parsed_response.result is not None and not parsed_response.result.startswith("Error:"):
                                tool_successfully_used = True
                                break # Found one successful use, no need to check further

            # Award bonus only if correct AND no tool was successfully used
            if correctness == 1.0 and not tool_successfully_used:
                rewards.append(1.0) # The weight (e.g., 0.1) will scale this during training
            else:
                rewards.append(0.0)

        return rewards

    def tool_execution_reward_func(self, completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
        """
        Reward function that checks tool execution success.

        Uses XMLParser to identify proper tool calls.
        """
        def check_execution(trajectory):
            tool_attempts = 0
            successful_executions = 0
            
            # Find assistant messages with tools and their responses
            for i, msg in enumerate(trajectory):
                if msg['role'] == 'assistant':
                    # Use parser to check for tool tag
                    parsed = self.parser.parse(msg['content'])
                    if hasattr(parsed, 'tool') and parsed.tool is not None:
                        # Found a properly formatted tool message
                        if i + 1 < len(trajectory) and trajectory[i + 1]['role'] == 'user':
                            tool_attempts += 1
                            # Check response with env_parser
                            multiplier = 1.0 
                            response = str(parsed.tool)
                            if (("sympy" in response) or ("numpy" in response)) and len(response) > 100:
                                multiplier = 1.5
                            else:
                                multiplier = 0.5
                            parsed_response = self.env_parser.parse(trajectory[i + 1]['content'])
                            if hasattr(parsed_response, 'result') and parsed_response.result is not None and not parsed_response.result.startswith("Error:"):
                                successful_executions += 1 * multiplier
            
            # Calculate reward
            if tool_attempts == 0:
                return 0.0
            return (successful_executions / tool_attempts)
        
        return [check_execution(c) for c in completions]
    
    def get_named_tool_reward_func(self, tool_name: str) -> Callable:
        """
        Returns a reward function that checks tool execution success for a specific tool.

        Uses XMLParser to identify proper tool calls.
        """
        def tool_reward_func(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
            """
            Reward function that checks execution success for the {tool_name} tool.
            
            Uses XMLParser to identify proper tool calls for the specified tool.
            """
            import json
            
            def check_tool_execution(trajectory: List[Dict[str, str]]) -> float:
                tool_attempts = 0
                successful_executions = 0
                
                # Find assistant messages with the specific tool and their responses
                for i, msg in enumerate(trajectory):
                    if msg['role'] == 'assistant':
                        # Use parser to check for tool tag
                        parsed = self.parser.parse(msg['content'])
                        if hasattr(parsed, 'tool') and parsed.tool is not None:
                            try:
                                command = json.loads(parsed.tool)
                                if isinstance(command, dict) and command.get("name") == tool_name:
                                    # Found a properly formatted tool message for the specific tool
                                    if i + 1 < len(trajectory) and trajectory[i + 1]['role'] == 'user':
                                        tool_attempts += 1
                                        # Check response with env_parser
                                        parsed_response = self.env_parser.parse(trajectory[i + 1]['content'])
                                        if hasattr(parsed_response, 'result') and parsed_response.result is not None and not parsed_response.result.startswith("Error:"):
                                            successful_executions += 1
                            except json.JSONDecodeError:
                                pass
                
                # Calculate reward
                if tool_attempts == 0:
                    return 0.0
                return (successful_executions / tool_attempts)
            
            return [check_tool_execution(c) for c in completions]
        
        # Create a function with the dynamic name based on tool_name
        tool_reward_func.__name__ = f"{tool_name}_reward_func"
        return tool_reward_func
    
    def get_named_tool_count_reward_func(self, tool_name: str) -> Callable:
        """
        Returns a reward function that counts the number of times the {tool_name} tool is used.
        """
        def tool_count_reward_func(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
            """
            Reward function that counts the number of times the {tool_name} tool is used.
            """
            import json

            def count_tool_executions(trajectory: List[Dict[str, str]]) -> float:
                successful_executions = 0.0
                for i, msg in enumerate(trajectory):
                    if msg['role'] == 'assistant':
                        parsed = self.parser.parse(msg['content'])
                        if hasattr(parsed, 'tool') and parsed.tool is not None:
                            try:
                                command = json.loads(parsed.tool)
                                if isinstance(command, dict) and command.get("name") == tool_name:
                                    # Found a properly formatted tool message for the specific tool
                                    if i + 1 < len(trajectory) and trajectory[i + 1]['role'] == 'user':
                                        parsed_response = self.env_parser.parse(trajectory[i + 1]['content'])
                                        if hasattr(parsed_response, 'result') and parsed_response.result is not None and not parsed_response.result.startswith("Error:"):
                                            successful_executions += 1
                            except json.JSONDecodeError:
                                pass
                return successful_executions
            
            return [count_tool_executions(c) for c in completions]
        
        tool_count_reward_func.__name__ = f"{tool_name}_count_reward_func"
        return tool_count_reward_func

    def get_named_tool_attempt_reward_func(self, tool_name: str) -> Callable:
        """
        Returns a reward function that counts the number of times the {tool_name} tool is used.
        """
        def tool_attempt_reward_func(completions: List[List[Dict[str, str]]], **kwargs) -> List[float]:
            """
            Reward function that counts the number of times the {tool_name} tool is used.
            """
            import json

            def count_tool_executions(trajectory: List[Dict[str, str]]) -> float:
                attempted_executions = 0.0
                for i, msg in enumerate(trajectory):
                    if msg['role'] == 'assistant':
                        parsed = self.parser.parse(msg['content'])
                        if hasattr(parsed, 'tool') and parsed.tool is not None:
                            try:
                                command = json.loads(parsed.tool)
                                if isinstance(command, dict) and command.get("name") == tool_name:
                                    attempted_executions += 1
                            except json.JSONDecodeError:
                                pass
                return attempted_executions
            
            return [count_tool_executions(c) for c in completions]
            
        tool_attempt_reward_func.__name__ = f"{tool_name}_attempt_reward_func"
        return tool_attempt_reward_func