import asyncio

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command


def test_simple_flow():
    builder = StateGraph()

    def node_a(state):
        return {"x": 1}

    def node_b(state):
        return Command(goto=END, update={"y": state.get("x", 0) + 2})

    builder.add_node("a", node_a)
    builder.add_node("b", node_b)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")

    graph = builder.compile()
    out = graph.invoke({})
    assert out["x"] == 1
    assert out["y"] == 3


def test_conditional_edges():
    builder = StateGraph()

    def decide(state):
        return "one" if state.get("flag") else "two"

    def node_start(state):
        return {"flag": True}

    def node_one(state):
        return {"result": "one"}

    def node_two(state):
        return {"result": "two"}

    builder.add_node("start", node_start)
    builder.add_node("one", node_one)
    builder.add_node("two", node_two)
    builder.add_edge(START, "start")
    builder.add_conditional_edges("start", decide, {"one": "one", "two": "two"})

    graph = builder.compile()
    out = graph.invoke({})
    assert out["result"] == "one"


def test_async_node():
    builder = StateGraph()

    async def a(state):
        await asyncio.sleep(0)
        return {"ok": True}

    def b(state):
        return Command(goto=END)

    builder.add_node("a", a)
    builder.add_node("b", b)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")

    graph = builder.compile()
    out = graph.invoke({})
    assert out.get("ok") is True
