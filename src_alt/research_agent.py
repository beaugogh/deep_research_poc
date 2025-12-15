"""Research Agent Implementation (src_alt)

This is a copy of `src/research_agent.py` adapted to reference the local
`src_alt.langgraph` graph implementation and otherwise keep the logic
identical.
"""

try:
    from typing_extensions import Literal
except Exception:  # pragma: no cover - typing_extensions not installed
    from typing import Literal

# This module uses an imperative implementation (ResearcherAgent) instead of
# a graph framework. No langgraph import is required.
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from langchain.chat_models import init_chat_model

from src_alt.state_research import ResearcherState, ResearcherOutputState
from src_alt.utils import tavily_search, get_today_str, think_tool
from src_alt.prompts import research_agent_prompt, compress_research_system_prompt, compress_research_human_message


# ===== CONFIGURATION =====

tools = [tavily_search, think_tool]
tools_by_name = {tool.name: tool for tool in tools}

model = init_chat_model(model="openai:gpt-5")
model_with_tools = model.bind_tools(tools)
summarization_model = init_chat_model(model="openai:gpt-5")
compress_model = init_chat_model(model="openai:gpt-5", max_tokens=32000)


def llm_call(state: ResearcherState):
    """Call the researcher model and return the updated researcher_messages.

    This helper wraps the model invocation so tests and the imperative
    loop remain easy to read.
    """
    return {
        "researcher_messages": [
            model_with_tools.invoke(
                [SystemMessage(content=research_agent_prompt)] + state["researcher_messages"]
            )
        ]
    }


def tool_node(state: ResearcherState):
    tool_calls = state["researcher_messages"][-1].tool_calls

    observations = []
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observations.append(tool.invoke(tool_call["args"]))

    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]

    return {"researcher_messages": tool_outputs}


def compress_research(state: ResearcherState) -> dict:
    system_message = compress_research_system_prompt.format(date=get_today_str())
    messages = [SystemMessage(content=system_message)] + state.get("researcher_messages", []) + [HumanMessage(content=compress_research_human_message)]
    response = compress_model.invoke(messages)

    raw_notes = [
        str(m.content) for m in filter_messages(
            state["researcher_messages"], 
            include_types=["tool", "ai"]
        )
    ]

    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)]
    }


def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """Return the next node name based on whether the last message issued tool calls."""
    messages = state["researcher_messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tool_node"
    return "compress_research"


class ResearcherAgent:
    """Imperative researcher agent with sync and async interfaces."""

    def __init__(self, max_iterations: int = 15):
        self.max_iterations = max_iterations

    def invoke(self, state: dict) -> dict:
        state = dict(state or {})
        iterations = 0

        while iterations < self.max_iterations:
            response = model_with_tools.invoke([SystemMessage(content=research_agent_prompt)] + state.get("researcher_messages", []))
            state["researcher_messages"] = [response]

            if response.tool_calls:
                tool_calls = response.tool_calls
                observations = []
                for tool_call in tool_calls:
                    tool = tools_by_name[tool_call["name"]]
                    observations.append(tool.invoke(tool_call["args"]))

                tool_outputs = [
                    ToolMessage(content=observation, name=tool_call["name"], tool_call_id=tool_call["id"]) 
                    for observation, tool_call in zip(observations, tool_calls)
                ]
                state["researcher_messages"] = tool_outputs
                iterations += 1
                continue

            system_message = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=system_message)] + state.get("researcher_messages", []) + [HumanMessage(content=compress_research_human_message)]
            compress_response = compress_model.invoke(messages)

            raw_notes = [str(m.content) for m in filter_messages(state["researcher_messages"], include_types=["tool", "ai"])]

            return {
                "compressed_research": str(compress_response.content),
                "raw_notes": ["\n".join(raw_notes)],
                "researcher_messages": state.get("researcher_messages", [])
            }

        return {"compressed_research": "", "raw_notes": [], "researcher_messages": state.get("researcher_messages", [])}

    async def ainvoke(self, state: dict) -> dict:
        state = dict(state or {})
        iterations = 0

        while iterations < self.max_iterations:
            response = await model_with_tools.ainvoke([SystemMessage(content=research_agent_prompt)] + state.get("researcher_messages", []))
            state["researcher_messages"] = [response]

            if response.tool_calls:
                tool_calls = response.tool_calls
                observations = []
                for tool_call in tool_calls:
                    tool = tools_by_name[tool_call["name"]]
                    observations.append(tool.invoke(tool_call["args"]))

                tool_outputs = [
                    ToolMessage(content=observation, name=tool_call["name"], tool_call_id=tool_call["id"]) 
                    for observation, tool_call in zip(observations, tool_calls)
                ]
                state["researcher_messages"] = tool_outputs
                iterations += 1
                continue

            system_message = compress_research_system_prompt.format(date=get_today_str())
            messages = [SystemMessage(content=system_message)] + state.get("researcher_messages", []) + [HumanMessage(content=compress_research_human_message)]
            compress_response = await compress_model.ainvoke(messages)

            raw_notes = [str(m.content) for m in filter_messages(state["researcher_messages"], include_types=["tool", "ai"])]

            return {
                "compressed_research": str(compress_response.content),
                "raw_notes": ["\n".join(raw_notes)],
                "researcher_messages": state.get("researcher_messages", [])
            }

        return {"compressed_research": "", "raw_notes": [], "researcher_messages": state.get("researcher_messages", [])}


researcher_agent = ResearcherAgent()
