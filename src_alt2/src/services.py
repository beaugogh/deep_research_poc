"""
Service configuration utilities for chat models and Tavily.

This module centralizes initialization logic so we can transparently switch
between real API-backed services and lightweight mock implementations when
API keys are unavailable. Set the environment variable
``MOCK_DEEP_RESEARCH=1`` to force mock mode, or simply omit the API keys.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from langchain.chat_models import init_chat_model as real_init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

try:
    from tavily import TavilyClient as RealTavilyClient
except Exception:  # pragma: no cover - tavily might be absent in mock runs
    RealTavilyClient = None  # type: ignore


def _is_truthy(value: Optional[str]) -> bool:
    return str(value).lower() in {"1", "true", "yes", "on"} if value is not None else False


def _determine_mock_mode() -> bool:
    if "MOCK_DEEP_RESEARCH" in os.environ:
        return _is_truthy(os.environ.get("MOCK_DEEP_RESEARCH"))
    # Fall back to mock mode when either API key is missing.
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    has_tavily = bool(os.environ.get("TAVILY_API_KEY"))
    return not (has_openai and has_tavily)


USING_MOCK_SERVICES = _determine_mock_mode()


class MockChatModel:
    """Minimal chat model drop-in replacement for offline demos."""

    def __init__(self, model: Optional[str] = None, **_: Any) -> None:
        self.model = model or "mock-model"

    def _build_content(self, messages: Sequence[BaseMessage]) -> str:
        last_human = next(
            (
                getattr(message, "content", "")
                for message in reversed(messages)
                if isinstance(message, HumanMessage)
            ),
            "",
        )
        suffix = f" about '{last_human}'" if last_human else ""
        return f"[{self.model}] Mock response{suffix}"

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        return AIMessage(content=self._build_content(messages), tool_calls=[])

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        await asyncio.sleep(0)  # keep signature compatible
        return self.invoke(messages)

    def with_structured_output(self, schema: Any):
        return MockStructuredModel(schema=schema, response_builder=self._build_content)

    def bind_tools(self, tools: Sequence[Any]):
        return MockToolChatModel(
            model_name=self.model,
            response_builder=self._build_content,
            tool_names=[getattr(tool, "name", "tool") for tool in tools],
        )


class MockStructuredModel:
    """Structured output wrapper returning deterministic schema instances."""

    def __init__(self, schema: Any, response_builder: Any) -> None:
        self.schema = schema
        self._response_builder = response_builder

    def _build_payload(self, messages: Sequence[BaseMessage]) -> Dict[str, str]:
        base_text = self._response_builder(messages)
        try:
            fields = self.schema.model_fields  # type: ignore[attr-defined]
        except AttributeError:  # pragma: no cover - fallback safety
            return {}
        return {field_name: f"{base_text} ({field_name})" for field_name in fields}

    def invoke(self, messages: Sequence[BaseMessage]):
        payload = self._build_payload(messages)
        return self.schema(**payload)

    async def ainvoke(self, messages: Sequence[BaseMessage]):
        await asyncio.sleep(0)
        return self.invoke(messages)


class MockToolChatModel:
    """Tool-aware variant that never emits tool calls."""

    def __init__(self, model_name: str, response_builder: Any, tool_names: Sequence[str]) -> None:
        self.model_name = model_name
        self._response_builder = response_builder
        self.tool_names = list(tool_names)

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        # We purposely skip tool calls to keep the mock execution simple.
        content = self._response_builder(messages)
        return AIMessage(content=f"{content} (tools available: {', '.join(self.tool_names)})", tool_calls=[])

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        await asyncio.sleep(0)
        return self.invoke(messages)


@dataclass
class MockTavilyResult:
    url: str
    title: str
    content: str
    raw_content: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "raw_content": self.raw_content,
        }


class MockTavilyClient:
    """Simple Tavily replacement returning canned text."""

    def search(
        self,
        query: str,
        max_results: int = 3,
        include_raw_content: bool = True,
        topic: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, str]]]:
        results: List[Dict[str, str]] = []
        for idx in range(1, max(1, max_results) + 1):
            result = MockTavilyResult(
                url=f"https://example.com/mock-{idx}",
                title=f"Mock result {idx} for '{query}'",
                content=(
                    f"Summary for {query}. Topic={topic or 'general'}. "
                    "This data is generated by the mock Tavily client."
                ),
                raw_content=(
                    f"Full mock content #{idx} discussing {query}. "
                    "Use this text as placeholder external research."
                )
                if include_raw_content
                else "",
            )
            results.append(result.to_dict())
        return {"results": results}


def init_chat_model(*args: Any, **kwargs: Any):
    """Return either the real LangChain chat model or the mock equivalent."""
    if USING_MOCK_SERVICES:
        return MockChatModel(**kwargs)
    return real_init_chat_model(*args, **kwargs)


def get_tavily_client():
    """Return either the real Tavily client or a mock replacement."""
    if USING_MOCK_SERVICES or RealTavilyClient is None:
        return MockTavilyClient()
    return RealTavilyClient()
