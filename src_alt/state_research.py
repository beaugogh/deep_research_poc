"""State Definitions and Pydantic Schemas for Research Agent (src_alt)

Copied from `src/state_research.py` and modified to reference the local
`src_alt.langgraph` message marker.
"""

import operator
try:
    from typing_extensions import TypedDict, Annotated, List, Sequence
except Exception:  # pragma: no cover - typing_extensions not installed
    from typing import TypedDict, Annotated, List, Sequence
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage


class ResearcherState(TypedDict):
    researcher_messages: Sequence[BaseMessage]
    tool_call_iterations: int
    research_topic: str
    compressed_research: str
    raw_notes: List[str]


class ResearcherOutputState(TypedDict):
    compressed_research: str
    raw_notes: List[str]
    researcher_messages: Sequence[BaseMessage]


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


class Summary(BaseModel):
    summary: str = Field(description="Concise summary of the webpage content")
    key_excerpts: str = Field(description="Important quotes and excerpts from the content")
