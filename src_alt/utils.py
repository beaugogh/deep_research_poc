"""Lightweight utilities for `src_alt` that avoid heavy external dependencies.

This module provides small, self-contained replacements for functions and
tools used by the alt agents: `get_today_str`, `tavily_search`, `think_tool`,
and `refine_draft_report`.
"""
from datetime import datetime
from types import SimpleNamespace


def get_today_str() -> str:
    return datetime.now().strftime("%a %b %-d, %Y")


class TavilySearchTool:
    name = "tavily_search"

    def invoke(self, query: str, max_results: int = 3, topic: str = "general") -> str:
        # Return a stable mock search output useful for local demos/tests
        return f"Search results for '{query}' (mock): No internet in demo mode."


class ThinkTool:
    name = "think_tool"

    def invoke(self, reflection: str) -> str:
        return f"Reflection recorded: {reflection}"


class RefineDraftReportTool:
    name = "refine_draft_report"

    def invoke(self, payload: dict) -> str:
        research_brief = payload.get("research_brief", "")
        findings = payload.get("findings", "")
        draft_report = payload.get("draft_report", "")
        # Simple refinement: append findings to draft
        return (draft_report or "") + "\n\nRefined with findings:\n" + findings


# Expose instances mirroring the `deep_research.utils` interface
tavily_search = TavilySearchTool()
think_tool = ThinkTool()
refine_draft_report = RefineDraftReportTool()
