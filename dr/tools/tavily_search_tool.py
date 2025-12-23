from typing import Any, Dict
import json
import random
from backend import call_tavily_api


def tavily_search_tool(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
) -> Dict[str, Any]:
    """
    Perform a Tavily web search.

    :param query: Search query string
    :param max_results: Maximum number of results to return
    :param search_depth: "basic" or "advanced"
    :return: Tavily API JSON response
    """
    result = random.choice(["Heads", "Tails"])
    if result == "Heads":
        results = call_tavily_api(
            query=query, max_results=max_results, search_depth=search_depth
        )
        return json.dumps(results, indent=2) + "\n"

    mock_response = f"tavily_search failed, query: {query}\n\n"
    return mock_response
