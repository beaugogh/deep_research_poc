"""
Demo runner for `src_alt` agents.

Goal: step-wise demo output (NOT token streaming)
- As soon as each major step finishes, show its outputs in Gradio.
- Do not wait for the whole workflow to complete before showing anything.

Backends:
  - fake        : deterministic local fake model
  - pangu       : llm_calls.call_llm (OpenAI-compatible endpoint configured in config.yaml)
  - siliconflow : llm_calls.call_siliconflow_llm (REST)

Run:
  python -m src_alt.demo --gradio --backend fake
  python -m src_alt.demo --gradio --backend pangu
  python -m src_alt.demo --gradio --backend siliconflow
"""

from __future__ import annotations

import os
import sys
import argparse
import asyncio
from types import SimpleNamespace
from typing import Sequence, Literal
from urllib.request import getproxies


from llm_calls import call_llm as call_pangu_llm, call_siliconflow_llm
from llm_calls import initialize as initialize_llm

Backend = Literal["fake", "pangu", "siliconflow"]


# =============================================================================
# Models (minimal interfaces expected by your agents)
# =============================================================================

class FakeModel:
    """Deterministic fake model. No tool calls; stable structured outputs."""
    def with_structured_output(self, schema):
        class S:
            def invoke(self_inner, messages):
                return SimpleNamespace(
                    research_brief="Mocked research brief",
                    draft_report="Mocked draft report",
                )
        return S()

    def bind_tools(self, tools):
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


class _BaseRemoteModel:
    """Shared message->payload conversion."""
    def _messages_to_payload(self, messages: Sequence[object]):
        out = []
        for m in messages:
            cls = m.__class__.__name__
            if cls == "SystemMessage":
                role = "system"
            elif cls == "ToolMessage":
                role = "tool"
            elif cls in ("HumanMessage",):
                role = "user"
            else:
                # Any other message-ish object is treated as assistant.
                role = "assistant"
            out.append({"role": role, "content": getattr(m, "content", str(m))})
        return out


class RealPanguModel(_BaseRemoteModel):
    """Adapter for llm_calls.call_llm(messages=[...])."""

    def with_structured_output(self, schema):
        parent = self

        class S:
            def invoke(self_inner, messages):
                payload = parent._messages_to_payload(messages)
                resp = call_pangu_llm(messages=payload)
                return SimpleNamespace(research_brief=resp, draft_report=resp)

        return S()

    def bind_tools(self, tools):
        parent = self

        class Bound:
            def invoke(self_inner, messages):
                payload = parent._messages_to_payload(messages)
                resp = call_pangu_llm(messages=payload)
                # Tool calling is not implemented in this shim; tool_calls stays empty.
                return SimpleNamespace(content=resp, tool_calls=[])

            async def ainvoke(self_inner, messages):
                return await asyncio.to_thread(self_inner.invoke, messages)

        return Bound()

    def invoke(self, messages):
        payload = self._messages_to_payload(messages)
        resp = call_pangu_llm(messages=payload)
        return SimpleNamespace(content=resp, tool_calls=[])

    async def ainvoke(self, messages):
        return await asyncio.to_thread(self.invoke, messages)


class RealSiliconFlowModel(_BaseRemoteModel):
    """Adapter for llm_calls.call_siliconflow_llm(messages=[...])."""

    def with_structured_output(self, schema):
        parent = self

        class S:
            def invoke(self_inner, messages):
                payload = parent._messages_to_payload(messages)
                resp = call_siliconflow_llm(messages=payload)
                return SimpleNamespace(research_brief=resp, draft_report=resp)

        return S()

    def bind_tools(self, tools):
        parent = self

        class Bound:
            def invoke(self_inner, messages):
                payload = parent._messages_to_payload(messages)
                resp = call_siliconflow_llm(messages=payload)
                return SimpleNamespace(content=resp, tool_calls=[])

            async def ainvoke(self_inner, messages):
                return await asyncio.to_thread(self_inner.invoke, messages)

        return Bound()

    def invoke(self, messages):
        payload = self._messages_to_payload(messages)
        resp = call_siliconflow_llm(messages=payload)
        return SimpleNamespace(content=resp, tool_calls=[])

    async def ainvoke(self, messages):
        return await asyncio.to_thread(self.invoke, messages)


# =============================================================================
# Fake LangChain injection (must match your agent call-sites)
# =============================================================================

def inject_fakes(backend: Backend) -> None:
    """
    Inject minimal fake `langchain` and `langchain_core` modules.

    This must support your usage patterns:
      - init_chat_model(...)
      - HumanMessage/SystemMessage/ToolMessage classes
      - get_buffer_string(...)
      - filter_messages(messages, include_types="tool")
      - filter_messages(messages, include_types=["tool", "ai"])
    """

    def init_chat_model(*args, **kwargs):
        if backend == "fake":
            return FakeModel()
        if backend == "pangu":
            return RealPanguModel()
        return RealSiliconFlowModel()

    # ---- langchain.chat_models ----
    langchain_pkg = sys.modules.setdefault("langchain", SimpleNamespace())
    chat_models_mod = SimpleNamespace(init_chat_model=init_chat_model)
    langchain_pkg.chat_models = chat_models_mod
    sys.modules["langchain.chat_models"] = chat_models_mod

    # ---- langchain_core.messages ----
    class HumanMessage:
        def __init__(self, content=""):
            self.content = content

    class SystemMessage(HumanMessage):
        pass

    class ToolMessage(HumanMessage):
        # Your agents call: ToolMessage(content=..., name=..., tool_call_id=...)
        def __init__(self, content="", name="", tool_call_id=None, tool_name=None):
            super().__init__(content)
            # tolerate both `name` and legacy `tool_name`
            self.name = name or (tool_name or "")
            self.tool_call_id = tool_call_id

    def get_buffer_string(messages, *_, **__):
        return "\n".join(getattr(m, "content", str(m)) for m in messages)

    def filter_messages(messages, include_types=None):
        """
        Compatibility shim for:
          - include_types="tool"                   (multi_agent_supervisor.py)
          - include_types=["tool", "ai"]           (research_agent.py)
          - include_types=(ToolMessage,)           (LangChain style)
          - include_types=ToolMessage              (type)
        """
        msgs = list(messages) if messages is not None else []

        if include_types is None:
            return msgs

        # normalize include_types into a list of selectors
        if isinstance(include_types, (str, type)):
            selectors = [include_types]
        elif isinstance(include_types, (list, tuple)):
            selectors = list(include_types)
        else:
            raise TypeError(f"Unsupported include_types: {include_types!r}")

        out = []
        for m in msgs:
            cls = m.__class__.__name__
            for sel in selectors:
                # String selectors used by your src_alt agents
                if isinstance(sel, str):
                    if sel == "tool" and cls == "ToolMessage":
                        out.append(m)
                        break
                    if sel == "ai":
                        # In this shim, any non-human/system/tool message is treated as AI output
                        if cls not in ("HumanMessage", "SystemMessage", "ToolMessage"):
                            out.append(m)
                            break

                # Type selectors
                elif isinstance(sel, type):
                    if isinstance(m, sel):
                        out.append(m)
                        break

        return out

    messages_mod = SimpleNamespace(
        HumanMessage=HumanMessage,
        BaseMessage=HumanMessage,
        SystemMessage=SystemMessage,
        ToolMessage=ToolMessage,
        get_buffer_string=get_buffer_string,
        filter_messages=filter_messages,
    )

    lc_core_pkg = sys.modules.setdefault("langchain_core", SimpleNamespace())
    lc_core_pkg.messages = messages_mod
    sys.modules["langchain_core.messages"] = messages_mod

    # ---- langchain_core.tools ----
    def tool(cls_or_func=None):
        # Identity decorator; supports @tool on classes in your state files
        if cls_or_func is None:
            return lambda x: x
        return cls_or_func

    tools_mod = SimpleNamespace(tool=tool)
    lc_core_pkg.tools = tools_mod
    sys.modules["langchain_core.tools"] = tools_mod


# =============================================================================
# Step-wise execution (yields after each major step completes)
# =============================================================================

def _ensure_src_alt_on_path() -> None:
    """
    Defensive sys.path handling:
    - Running as `python -m src_alt.demo` should already be fine.
    - This helps if someone runs the file directly.
    """
    try:
        import pathlib
        here = pathlib.Path(__file__).resolve()
        repo_root = here.parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
    except Exception:
        # Best-effort only
        pass


async def run_research_stepwise(user_request: str, backend: Backend):
    """
    Yields tuples (research_brief, draft_report, notes, final_report)
    after each step finishes:
      1) scope_research
      2) supervisor_agent
      3) final_report_generation
    """
    _ensure_src_alt_on_path()
    inject_fakes(backend)

    # Import AFTER injection so init_chat_model resolves to our shim.
    from src_alt.research_agent_scope import scope_research
    from src_alt.multi_agent_supervisor import supervisor_agent
    from src_alt.research_agent_full import final_report_generation

    state = {
        "messages": [f"User request: {user_request}"],
        "user_request": user_request,
    }

    # STEP 1: scoping
    state = await scope_research.ainvoke(state)
    yield (
        state.get("research_brief", "") or "",
        state.get("draft_report", "") or "",
        "",
        "",
    )

    # STEP 2: supervisor loop (multi-agent research)
    state = await supervisor_agent.ainvoke(state)
    notes = state.get("notes") or []
    yield (
        state.get("research_brief", "") or "",
        state.get("draft_report", "") or "",
        "\n\n".join(str(x) for x in notes),
        "",
    )

    # STEP 3: final report generation
    final = await final_report_generation(state)
    state.update(final)
    yield (
        state.get("research_brief", "") or "",
        state.get("draft_report", "") or "",
        "\n\n".join(str(x) for x in (state.get("notes") or [])),
        state.get("final_report", "") or "",
    )


# =============================================================================
# Gradio
# =============================================================================

def launch_gradio(backend: Backend):
    try:
        import gradio as gr
    except Exception as exc:
        raise RuntimeError("Gradio is required. Install with `pip install gradio`.") from exc

    async def on_submit(user_request: str):
        # Always show *something* quickly, even if user_request is empty
        req = (user_request or "").strip() or "Write a short mock report"

        try:
            async for update in run_research_stepwise(req, backend):
                yield update
        except Exception as e:
            # Surface error in UI immediately
            err = f"ERROR: {type(e).__name__}: {e}"
            yield ("", "", err, "")

    with gr.Blocks(title="Deep Research Demo (Step-wise)") as demo:
        gr.Markdown("## Deep Research Demo (Step-wise Updates)\nResults update after each major step completes.")

        user_request = gr.Textbox(lines=4, label="User request")
        run_btn = gr.Button("Run")

        research_brief = gr.Textbox(label="Research brief", lines=6)
        draft_report = gr.Textbox(label="Draft report", lines=10)
        notes = gr.Textbox(label="Notes / Progress", lines=12)
        final_report = gr.Textbox(label="Final report", lines=16)

        run_btn.click(
            fn=on_submit,
            inputs=[user_request],
            outputs=[research_brief, draft_report, notes, final_report],
        )

    demo.launch()


# =============================================================================
# Proxy helpers (optional; llm_calls.initialize already bypasses proxies)
# =============================================================================

def bypass_proxies():
    proxies = getproxies()
    http_proxy = proxies.get("http") or proxies.get("https")
    if http_proxy:
        # Respect existing env if set
        os.environ.setdefault("HTTP_PROXY", http_proxy)
        os.environ.setdefault("http_proxy", http_proxy)
        os.environ.setdefault("HTTPS_PROXY", http_proxy)
        os.environ.setdefault("https_proxy", http_proxy)

    no_proxy = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    for host in (".huawei.com", "localhost", "127.0.0.1", "::1"):
        if host not in no_proxy:
            no_proxy = (no_proxy + "," + host) if no_proxy else host
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = no_proxy


# =============================================================================
# Main
# =============================================================================

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["fake", "pangu", "siliconflow"], default="fake")
    p.add_argument("--gradio", action="store_true", help="Launch Gradio UI")
    args = p.parse_args(argv)

    bypass_proxies()

    if args.backend != "fake":
        # Both pangu and siliconflow use CONFIG loaded by initialize_llm()
        initialize_llm()

    if args.gradio:
        launch_gradio(args.backend)
    else:
        print("This demo is intended to be run with --gradio for step-wise updates.")
        print("Example: python -m src_alt.demo --gradio --backend pangu")


if __name__ == "__main__":
    # python -m src_alt.demo --gradio --backend pangu
    main()
