"""Multi-agent supervisor (src_alt) — uses local StateGraph replacement.

This is a copy of `src/multi_agent_supervisor.py` adapted to run with the
plain-Python `src_alt.langgraph` replacement. The internal logic and prompts
are unchanged.
"""

import asyncio

try:
    from typing_extensions import Literal
except Exception:  # pragma: no cover - typing_extensions not installed
    from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    HumanMessage, 
    BaseMessage, 
    SystemMessage, 
    ToolMessage,
    filter_messages
)
from src_alt.research_agent import researcher_agent

from src_alt.prompts import lead_researcher_with_multiple_steps_diffusion_double_check_prompt
from src_alt.state_multi_agent_supervisor import (
    SupervisorState,
    ConductResearch,
    ResearchComplete
)
from src_alt.utils import get_today_str, think_tool, refine_draft_report


def get_notes_from_tool_calls(messages: list[BaseMessage]) -> list[str]:
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]


# Ensure async compatibility for Jupyter environments
try:
    import nest_asyncio
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            nest_asyncio.apply()
    except ImportError:
        pass
except ImportError:
    pass


# ===== CONFIGURATION =====

supervisor_tools = [ConductResearch, ResearchComplete, think_tool,refine_draft_report]
supervisor_model = init_chat_model(model="openai:gpt-5")
supervisor_model_with_tools = supervisor_model.bind_tools(supervisor_tools)

# System constants
max_researcher_iterations = 15
max_concurrent_researchers = 3


# ===== SUPERVISOR NODES =====

async def _supervisor_node(state: SupervisorState) -> dict:
    """Run the supervisor model to produce the next supervisory message.

    Returns a dict with updated 'supervisor_messages' and increments the
    'research_iterations' counter.
    """
    supervisor_messages = state.get("supervisor_messages", [])

    system_message = lead_researcher_with_multiple_steps_diffusion_double_check_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=max_concurrent_researchers,
        max_researcher_iterations=max_researcher_iterations,
    )
    messages = [SystemMessage(content=system_message)] + supervisor_messages

    response = await supervisor_model_with_tools.ainvoke(messages)

    return {"supervisor_messages": [response], "research_iterations": state.get("research_iterations", 0) + 1}


async def _supervisor_tools_node(state: SupervisorState) -> tuple[dict, bool]:
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    tool_messages = []
    all_raw_notes = []
    draft_report = ""
    should_end = False

    exceeded_iterations = research_iterations >= max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete = any(tool_call["name"] == "ResearchComplete" for tool_call in most_recent_message.tool_calls)

    if exceeded_iterations or no_tool_calls or research_complete:
        # If we should finish, return the accumulated notes and brief
        should_end = True
        return (
            {
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", ""),
            },
            True,
        )

    try:
        think_tool_calls = [tool_call for tool_call in most_recent_message.tool_calls if tool_call["name"] == "think_tool"]
        conduct_research_calls = [tool_call for tool_call in most_recent_message.tool_calls if tool_call["name"] == "ConductResearch"]
        refine_report_calls = [tool_call for tool_call in most_recent_message.tool_calls if tool_call["name"] == "refine_draft_report"]

        for tool_call in think_tool_calls:
            observation = think_tool.invoke(tool_call["args"])
            tool_messages.append(ToolMessage(content=observation, name=tool_call["name"], tool_call_id=tool_call["id"]))

        if conduct_research_calls:
            coros = [
                researcher_agent.ainvoke({
                    "researcher_messages": [HumanMessage(content=tool_call["args"]["research_topic"])],
                    "research_topic": tool_call["args"]["research_topic"]
                })
                for tool_call in conduct_research_calls
            ]

            tool_results = await asyncio.gather(*coros)

            research_tool_messages = [
                ToolMessage(content=result.get("compressed_research", "Error synthesizing research report"), name=tool_call["name"], tool_call_id=tool_call["id"]) 
                for result, tool_call in zip(tool_results, conduct_research_calls)
            ]

            tool_messages.extend(research_tool_messages)

            all_raw_notes = ["\n".join(result.get("raw_notes", [])) for result in tool_results]

        for tool_call in refine_report_calls:
            notes = get_notes_from_tool_calls(supervisor_messages)
            findings = "\n".join(notes)

            draft_report = refine_draft_report.invoke({
                "research_brief": state.get("research_brief", ""),
                "findings": findings,
                "draft_report": state.get("draft_report", "")
            })

            tool_messages.append(ToolMessage(content=draft_report, name=tool_call["name"], tool_call_id=tool_call["id"]))

    except Exception:
        return ({}, True)

    if len(refine_report_calls) > 0:
        return ({"supervisor_messages": tool_messages, "raw_notes": all_raw_notes, "draft_report": draft_report}, False)
    return ({"supervisor_messages": tool_messages, "raw_notes": all_raw_notes}, False)


class SupervisorAgent:
    """Imperative supervisor agent that will run supervisor and its tool
    handling node in a loop until research completes or an exit condition is met."""

    def __init__(self):
        self.max_iterations = max_researcher_iterations

    async def ainvoke(self, state: dict) -> dict:
        state = dict(state or {})

        while True:
            # Supervisor node: get next supervisory message
            supervisor_update = await _supervisor_node(state)
            state.update(supervisor_update)

            # Supervisor tools node: evaluate and execute tool calls
            tools_update, should_end = await _supervisor_tools_node(state)
            state.update(tools_update)

            if should_end:
                return state

            # otherwise loop to continue supervision


supervisor_agent = SupervisorAgent()
