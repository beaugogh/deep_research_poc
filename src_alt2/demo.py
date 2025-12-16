
"""Command-line demo for the ThinkDepth.ai deep research agent.

This script mirrors the workflow showcased in ``thinkdepthai_deepresearch.ipynb``:

1. Build the multi-agent research graph with an in-memory checkpoint.
2. Run the agent asynchronously against a research question.
3. Print the final report in Markdown (with Rich when available).

Set ``MOCK_DEEP_RESEARCH=1`` or leave the API keys unset to run the demo
with lightweight mock models and search results.

Usage:
    uv run python demo.py

Edit ``PROMPT``, ``THREAD_ID``, or ``RECURSION_LIMIT`` in this file to change
the demo behavior without passing CLI arguments.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Dict

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.research_agent_full import deep_researcher_builder
from src.services import USING_MOCK_SERVICES


DEFAULT_PROMPT = (
    "Write a paper to discuss the influence of AI interaction on interpersonal "
    "relations, considering AI's potential to fundamentally change how and why "
    "individuals relate to each other."
)

# Update these constants to change the demo behavior without CLI args.
PROMPT = DEFAULT_PROMPT
THREAD_ID = "demo-thread"
RECURSION_LIMIT = 2


def ensure_env_var(name: str) -> None:
    """Exit early if the required environment variable is missing."""
    if USING_MOCK_SERVICES:
        return
    if not os.environ.get(name):
        print(f"Environment variable {name} must be set before running the demo.", file=sys.stderr)
        sys.exit(1)


async def run_agent(prompt: str, *, thread_id: str, recursion_limit: int) -> Dict[str, Any]:
    """Build the agent with a checkpoint and run it on the provided prompt."""
    checkpointer = InMemorySaver()
    full_agent = deep_researcher_builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id, "recursion_limit": recursion_limit}}
    return await full_agent.ainvoke({"messages": [HumanMessage(content=prompt)]}, config=config)


def render_report(report: str) -> None:
    """Render the final report as Markdown if Rich is installed, plain text otherwise."""
    if not report:
        print("Agent finished without returning a final report.")
        return

    try:
        from rich.console import Console
        from rich.markdown import Markdown

        Console().print(Markdown(report))
    except Exception:
        # Rich isn't available or failed to render; fall back to simple printing.
        print(report)


async def async_main() -> None:
    ensure_env_var("OPENAI_API_KEY")
    ensure_env_var("TAVILY_API_KEY")

    print("Building full research agent graph ...", flush=True)
    result = await run_agent(
        PROMPT,
        thread_id=THREAD_ID,
        recursion_limit=RECURSION_LIMIT,
    )

    print("\nFinal report:\n")
    render_report(result.get("final_report", ""))


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.", file=sys.stderr)


if __name__ == "__main__":
    main()
