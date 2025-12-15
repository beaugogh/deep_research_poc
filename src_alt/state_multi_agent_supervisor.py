"""Supervisor state & tools (src_alt)

Small adaptation of `src/state_multi_agent_supervisor.py` pointing to our
local message marker implementation.
"""

import operator
try:
    from typing_extensions import Annotated, TypedDict, Sequence
except Exception:  # pragma: no cover - typing_extensions not installed
    from typing import Annotated, TypedDict, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field


class SupervisorState(TypedDict):
    supervisor_messages: Sequence[BaseMessage]
    research_brief: str
    notes: list[str]
    research_iterations: int
    raw_notes: list[str]
    draft_report: str


@tool
class ConductResearch(BaseModel):
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )


@tool
class ResearchComplete(BaseModel):
    pass
