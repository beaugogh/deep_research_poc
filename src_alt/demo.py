"""Demo runner for `src_alt` agents.

Provides a mock-based runner so you can exercise the FullAgent without
installing LLM dependencies. Run as a script:

    python -m src_alt.demo --mock

This will inject small fake `langchain`/`langchain_core` modules into
sys.modules so the agent imports succeed and return deterministic outputs.
"""
from __future__ import annotations

import sys
import argparse
from types import SimpleNamespace
import asyncio
from typing import Sequence

# Optional siliconflow integration
try:
    from siliconflow import call_siliconflow_llm, API_KEY as SILICONFLOW_API_KEY
except Exception:
    call_siliconflow_llm = None
    SILICONFLOW_API_KEY = None


class FakeModel:
    def __init__(self, name="fake"):
        self.name = name

    def with_structured_output(self, schema):
        # Return an object whose .invoke returns a SimpleNamespace matching
        # the fields expected by the code.
        class S:
            def invoke(self_inner, messages):
                # Provide fields used by the agent code conservatively
                return SimpleNamespace(research_brief="Mocked research brief", draft_report="Mocked draft report")
        return S()

    def bind_tools(self, tools):
        # Return an object with .ainvoke/.invoke that returns a SimpleNamespace
        class Bound:
            def invoke(self_inner, messages):
                return SimpleNamespace(content="mock content", tool_calls=[])

            async def ainvoke(self_inner, messages):
                return SimpleNamespace(content="mock content", tool_calls=[])
        return Bound()

    def invoke(self, messages):
        return SimpleNamespace(content="mock content", tool_calls=[])

    async def ainvoke(self, messages):
        return SimpleNamespace(content="mock content", tool_calls=[])


def inject_fakes():
    """Inject minimal fake packages into sys.modules so imports work.

    For more advanced usage you can use `inject_fakes_impl` to request a
    SiliconFlow-backed model instead of the fake model (e.g. for integration
    testing with a real API key).
    """
    return inject_fakes_impl(use_siliconflow=False, api_key=None)


def inject_fakes_impl(use_siliconflow: bool = False, api_key: str | None = None):
    """Inject fakes; optionally use a siliconflow-backed model when
    `use_siliconflow` is True and `siliconflow.call_siliconflow_llm` is
    importable.
    """
    # fake langchain.chat_models.init_chat_model - returns either a fake
    # model or a RealSiliconFlowModel depending on availability.
    def init_chat_model_factory(*args, **kwargs):
        if use_siliconflow and call_siliconflow_llm is not None:
            return RealSiliconFlowModel(api_key=api_key or SILICONFLOW_API_KEY, model=kwargs.get("model"))
        return FakeModel()

    fake_chat_models = SimpleNamespace(init_chat_model=init_chat_model_factory)
    sys.modules.setdefault("langchain", SimpleNamespace())
    sys.modules["langchain.chat_models"] = fake_chat_models

    # fake langchain_core.messages with HumanMessage
    class HumanMessage:
        def __init__(self, content=""):
            self.content = content

    class SystemMessage(HumanMessage):
        pass

    class ToolMessage(HumanMessage):
        def __init__(self, content="", tool_name=""):
            super().__init__(content)
            self.tool_name = tool_name

    def get_buffer_string(messages, *args, **kwargs):
        # Simple utility to join message contents for demos
        return "\n".join(getattr(m, "content", str(m)) for m in messages)

    def filter_messages(messages, include_types="tool"):
        if include_types == "tool":
            return [m for m in messages if isinstance(m, ToolMessage)]
        return messages

    fake_messages_mod = SimpleNamespace(
        HumanMessage=HumanMessage,
        BaseMessage=HumanMessage,
        SystemMessage=SystemMessage,
        ToolMessage=ToolMessage,
        get_buffer_string=get_buffer_string,
        filter_messages=filter_messages,
    )
    sys.modules["langchain_core.messages"] = fake_messages_mod

    # Fake tools module
    def tool(cls_or_func=None):
        # identity decorator that returns the class or function unchanged
        if cls_or_func is None:
            return lambda x: x
        return cls_or_func

    fake_tools_mod = SimpleNamespace(tool=tool)
    sys.modules["langchain_core.tools"] = fake_tools_mod
    sys.modules.setdefault("langchain_core", SimpleNamespace())
    sys.modules["langchain_core.messages"] = fake_messages_mod

    return True


class RealSiliconFlowModel:
    """Thin adapter that calls `call_siliconflow_llm` while providing the
    minimal interface expected by the agents (sync/async invoke, bind_tools,
    with_structured_output).
    """

    def __init__(self, api_key: str | None = None, model: str | None = None):
        if call_siliconflow_llm is None:
            raise RuntimeError("siliconflow.call_siliconflow_llm not available")
        self.api_key = api_key or SILICONFLOW_API_KEY
        self.model = model or "deepseek-ai/DeepSeek-V3.2"

    def _messages_to_payload(self, messages: Sequence[object]):
        out = []
        for m in messages:
            clsname = getattr(m, "__class__", type(m)).__name__
            if clsname == "SystemMessage":
                role = "system"
            elif clsname == "ToolMessage":
                role = "assistant"
            else:
                role = "user"
            out.append({"role": role, "content": getattr(m, "content", str(m))})
        return out

    def with_structured_output(self, schema):
        class S:
            def __init__(self_inner, parent: "RealSiliconFlowModel"):
                self_inner.parent = parent

            def invoke(self_inner, messages):
                payload = self_inner.parent._messages_to_payload(messages)
                resp = call_siliconflow_llm(api_key=self_inner.parent.api_key, messages=payload, model=self_inner.parent.model)
                return SimpleNamespace(research_brief=resp, draft_report=resp)

        return S(self)

    def bind_tools(self, tools):
        class Bound:
            def __init__(self_inner, parent: "RealSiliconFlowModel"):
                self_inner.parent = parent

            def invoke(self_inner, messages):
                payload = self_inner.parent._messages_to_payload(messages)
                resp = call_siliconflow_llm(api_key=self_inner.parent.api_key, messages=payload, model=self_inner.parent.model)
                return SimpleNamespace(content=resp, tool_calls=[])

            async def ainvoke(self_inner, messages):
                result = await asyncio.to_thread(lambda: Bound(self_inner.parent).invoke(messages))
                return result

        return Bound(self)

    def invoke(self, messages):
        payload = self._messages_to_payload(messages)
        resp = call_siliconflow_llm(api_key=self.api_key, messages=payload, model=self.model)
        return SimpleNamespace(content=resp, tool_calls=[])

    async def ainvoke(self, messages):
        result = await asyncio.to_thread(lambda: self.invoke(messages))
        return result


def run_mock(use_siliconflow: bool = False, siliconflow_api_key: str | None = None):
    inject_fakes_impl(use_siliconflow=use_siliconflow, api_key=siliconflow_api_key)

    # Import the full agent and run it
    # Ensure the main 'src' package (deep_research) is importable from repo root
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    from src_alt.research_agent_full import agent

    # Minimal input state
    state = {"messages": ["User request: Mock the flow"], "user_request": "Write a short mock report"}

    print("Running FullAgent (mock)... (siliconflow=" + str(use_siliconflow) + ")")
    out = agent.invoke(state)
    print("Result:")
    for k, v in out.items():
        print(k, "=", v)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mock", action="store_true", help="Run with mock LLMs/tools")
    p.add_argument("--siliconflow", action="store_true", help="Use SiliconFlow API for model calls (requires siliconflow package and API key)")
    p.add_argument("--siliconflow-api-key", type=str, default=None, help="Explicit SiliconFlow API key (optional)")
    args = p.parse_args(argv)

    if args.mock:
        run_mock(use_siliconflow=args.siliconflow, siliconflow_api_key=args.siliconflow_api_key)
    else:
        print("Non-mock run requires langchain and LLM configuration. Use --mock to run a local demo.")


if __name__ == "__main__":
    main()
