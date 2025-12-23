import json
from typing import List, Dict

from .tavily_search_tool import tavily_search_tool

TOOL_REGISTRY = {
    "tavily_search_tool": tavily_search_tool,
}


def call_tool(input_dict: dict):
    tool_name = input_dict["tool_name"]
    arguments = input_dict.get("arguments", {})

    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")

    tool_fn = TOOL_REGISTRY[tool_name]

    return tool_fn(**arguments)


class ToolParseError(ValueError):
    pass


def parse_tool_calls(llm_output: str, *, strict: bool = False) -> List[Dict]:
    """
    Parse tool calls from an LLM output.

    Modes:
      strict=True   → Parse the FIRST valid tool-call object (if any) - OPENAI style
      strict=False  → Parse ALL valid tool-call objects

    Returns:
      List of parsed tool-call dictionaries.
      Returns [] if no valid tool calls are found.
    """

    if not llm_output or not llm_output.strip():
        return []

    text = llm_output.strip()

    def is_valid_tool_call(obj: Dict) -> bool:
        return (
            isinstance(obj, dict)
            and "tool_name" in obj
            and "arguments" in obj
            and isinstance(obj["arguments"], dict)
        )

    tool_calls: List[Dict] = []
    length = len(text)
    idx = 0

    while idx < length:
        if text[idx] != "{":
            idx += 1
            continue

        brace_count = 0
        start = idx
        end = idx

        while end < length:
            if text[end] == "{":
                brace_count += 1
            elif text[end] == "}":
                brace_count -= 1

            if brace_count == 0:
                break
            end += 1

        if brace_count != 0:
            idx += 1
            continue

        candidate = text[start : end + 1]

        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            idx = end + 1
            continue

        if is_valid_tool_call(parsed):
            tool_calls.append(parsed)
            if strict:
                return tool_calls  # FIRST valid tool call only

        idx = end + 1

    return tool_calls
