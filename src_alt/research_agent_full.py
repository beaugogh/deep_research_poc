"""Full Multi-Agent Research System (src_alt)

Copied from `src/research_agent_full.py` and adapted to use the local
`src_alt.langgraph` replacement when constructing the graph in this
alternative package.
"""

from langchain_core.messages import HumanMessage

from src_alt.utils import get_today_str
from src_alt.prompts import final_report_generation_with_helpfulness_insightfulness_hit_citation_prompt
from src_alt.state_scope import AgentState, AgentInputState
from src_alt.research_agent_scope import scope_research
from src_alt.multi_agent_supervisor import supervisor_agent

from langchain.chat_models import init_chat_model
writer_model = init_chat_model(model="openai:gpt-5", max_tokens=40000)


async def final_report_generation(state: AgentState):
    notes = state.get("notes", [])

    findings = "\n".join(notes)

    final_report_prompt = final_report_generation_with_helpfulness_insightfulness_hit_citation_prompt.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str(),
        draft_report=state.get("draft_report", ""),
        user_request=state.get("user_request", "")
    )

    final_report = await writer_model.ainvoke([HumanMessage(content=final_report_prompt)])

    return {
        "final_report": final_report.content, 
        "messages": ["Here is the final report: " + final_report.content],
    }


class FullAgent:
    """Top-level agent that runs scoping, supervisor and final report
    generation sequentially.
    """

    async def ainvoke(self, input_state: dict) -> dict:
        state = dict(input_state or {})

        # Scoping phase
        state = await scope_research.ainvoke(state)

        # Supervision / multi-agent research
        state = await supervisor_agent.ainvoke(state)

        # Final report generation
        final = await final_report_generation(state)
        state.update(final)
        return state

    def invoke(self, input_state: dict) -> dict:
        import asyncio
        return asyncio.run(self.ainvoke(input_state))


agent = FullAgent()
