"""User Clarification and Research Brief Generation (src_alt)

This is a copy of `src/research_agent_scope.py` adapted to use the plain
Python `src_alt.langgraph` implementation instead of the external
`langgraph` dependency.
"""

from datetime import datetime
try:
    from typing_extensions import Literal
except Exception:  # pragma: no cover - typing_extensions not installed
    from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, get_buffer_string

from src_alt.prompts import (
    transform_messages_into_research_topic_human_msg_prompt,
    draft_report_generation_prompt,
    clarify_with_user_instructions,
)
from src_alt.state_scope import AgentState, ResearchQuestion, AgentInputState, DraftReport
from src_alt.utils import get_today_str


# (Using `get_today_str` from `src_alt.utils` for consistency across the package)


# ===== CONFIGURATION =====

# Initialize model
model = init_chat_model(model="openai:gpt-5")
creative_model = init_chat_model(model="openai:gpt-5")


# ===== WORKFLOW NODES =====

def clarify_with_user(state: AgentState):
    #uncomment if you want to enable this module  
    """
    Determine if the user's request contains sufficient information to proceed with research.

    Uses structured output to make deterministic decisions and avoid hallucination.
    Routes to either research brief generation or ends with a clarification question.
    """

    # Simplified logic: proceed to brief creation
    return {"goto": "write_research_brief"}


def write_research_brief(state: AgentState):
    """
    Transform the conversation history into a comprehensive research brief.

    Uses structured output to ensure the brief follows the required format
    and contains all necessary details for effective research.
    """
    # Set up structured output model
    structured_output_model = model.with_structured_output(ResearchQuestion)

    # Generate research brief from conversation history
    response = structured_output_model.invoke([
        HumanMessage(content=transform_messages_into_research_topic_human_msg_prompt.format(
            messages=get_buffer_string(state.get("messages", [])),
            date=get_today_str()
        ))
    ])

    # Update state with generated research brief and pass it to the supervisor
    return {"goto": "write_draft_report", "update": {"research_brief": response.research_brief}}


def write_draft_report(state: AgentState):
    """
    Final report generation node.

    Synthesizes all research findings into a comprehensive final report
    """
    # Set up structured output model
    structured_output_model = creative_model.with_structured_output(DraftReport)
    research_brief = state.get("research_brief", "")
    draft_report_prompt = draft_report_generation_prompt.format(
        research_brief=research_brief,
        date=get_today_str()
    )

    response = structured_output_model.invoke([HumanMessage(content=draft_report_prompt)])

    return {
        "research_brief": research_brief,
        "draft_report": response.draft_report,
        "supervisor_messages": ["Here is the draft report: " + response.draft_report, research_brief]
    }


class ScopeAgent:
    """Imperative scoping agent that runs the three scoping nodes sequentially."""

    def invoke(self, state: dict) -> dict:
        state = dict(state or {})

        # Clarify with user
        res = clarify_with_user(state)
        if isinstance(res, dict) and res.get("goto") == "write_research_brief":
            # Write research brief
            res2 = write_research_brief(state)
            if isinstance(res2, dict) and "update" in res2:
                state.update(res2["update"])

            # Write draft report
            res3 = write_draft_report(state)
            if isinstance(res3, dict):
                state.update(res3)

        return state

    async def ainvoke(self, state: dict) -> dict:
        # Nodes are synchronous in this implementation so just reuse invoke
        return self.invoke(state)


scope_research = ScopeAgent()
