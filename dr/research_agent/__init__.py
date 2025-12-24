from pathlib import Path
import json
from agent import Agent
from state import State, Role
from utils import get_today_str, load_prompt, render_tools_for_prompt
from backend import call_llm
from tools import call_tool, parse_tool_calls

MAX_TOOL_CALLS = 3


class ResearchAgent(Agent):

    def __init__(self, state: State):
        super().__init__(state)
        today = get_today_str()
        tools_txt = render_tools_for_prompt(self.context_path / "tools.json")
        system_prompt = load_prompt(
            self.context_path / "prompt.txt", date=today, tools=tools_txt
        )
        self.dialog = state["dialog"]
        self.dialog.add_message(role=Role.SYSTEM, content=system_prompt)

    def run(self):
        messages = self.dialog.to_messages()
        output = call_llm(messages=messages)
        tool_calls = parse_tool_calls(output, strict=False)
        # 1. If there are multiple tool calls, keep the non think_tool ones, because at this point, think_tool thoughts are just LLM hallucinations.
        # 2. If there is only one tool call, think_tool or otherwise, proceed normally.
        # 3. If there are no tool calls, just add the output as assistant message. (ALTERNATIVE: route to a full-fledged summarization node)
        if len(tool_calls) > 1:
            tool_calls = [
                tc for tc in tool_calls if tc.get("tool_name", None) != "think_tool"
            ]

        if tool_calls:
            # TODO: asynchronous tool calls
            for tc in tool_calls:
                tc_msg = "Invoking tool call:\n" + json.dumps(tc, indent=2)
                self.dialog.add_message(role=Role.ASSISTANT, content=tc_msg)
                yield tc_msg

                tool_name = tc.get("tool_name", None)
                # think_tool does NOT count toward limit
                if tool_name == "think_tool":
                    continue

                # real tool calls → check budget
                if self.state["tool_call_iterations"] >= MAX_TOOL_CALLS:
                    # Stop tool loop cleanly
                    final_msg = (
                        "I have reached the maximum number of tool calls and will now "
                        "provide a final response based on the information gathered."
                    )
                    self.dialog.add_message(role=Role.ASSISTANT, content=final_msg)
                    yield final_msg
                    return

                # execute tool call
                tool_result = call_tool(tc)
                self.state["tool_call_iterations"] += 1
                self.dialog.add_message(
                    role=Role.TOOL,
                    content=tool_result,
                )
                yield tool_result

            # recursive call to continue the dialog
            yield from self.run()
        else:
            output = output.strip()
            self.dialog.add_message(role=Role.ASSISTANT, content=output)
            yield output
