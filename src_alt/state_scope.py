"""State Definitions and Pydantic Schemas for Research Scoping (src_alt)

This is a near-copy of `src/state_scope.py`. It imports the local
`src_alt.langgraph` utilities (MessagesState/add_messages) instead of the
external package.
"""

import operator
try:
    from typing_extensions import Optional, Annotated, List, Sequence
except Exception:  # pragma: no cover - typing_extensions not installed
    from typing import Optional, Annotated, List, Sequence

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


# ===== STATE DEFINITIONS =====

class AgentInputState(dict):
    """Lightweight dict-based input state for the scoping agent."""


class AgentState(dict):
    """Dict-based main agent state.

    Expected keys include (but are not limited to):
    - 'messages': list of BaseMessage
    - 'research_brief': Optional[str]
    - 'supervisor_messages': list of BaseMessage
    - 'raw_notes', 'notes': list[str]
    - 'draft_report', 'final_report': str
    """


# ===== STRUCTURED OUTPUT SCHEMAS =====

class ClarifyWithUser(BaseModel):
    need_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question.",
    )
    question: str = Field(
        description="A question to ask the user to clarify the report scope",
    )
    verification: str = Field(
        description="Verify message that we will start research after the user has provided the necessary information.",
    )


class ResearchQuestion(BaseModel):
    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )


class DraftReport(BaseModel):
    draft_report: str = Field(
        description="A draft report that will be used to guide the research.",
    )
