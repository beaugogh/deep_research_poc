from state import State, Dialog, Role
from research_agent import ResearchAgent
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # query = "Hello, can you help me with my research?" # no tool call
    # query = "Summarize the latest research on quantum computing." # call tavily search tool
    query = "why is the sky blue? and how tall is mount everest?" # multiple tool calls, sometimes
    dialog = Dialog(limit=5)
    dialog.add_message(role=Role.USER, content=query)
    agent = ResearchAgent({
        "dialog": dialog,
        "tool_call_iterations": 0,
    })
    for res in agent.run():
        logger.info(res)
