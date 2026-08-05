"""
Multi-agent graph: a supervisor routes each turn to one of three
specialist nodes - retrieval_agent (SOP search), api_agent (inventory
gap tools), or response_agent (formatting/summarization, no tools) -
looping back through the supervisor until it's ready to respond.

Note the API shift from pre-1.0 LangChain: there's no more
AgentExecutor + create_tool_calling_agent + manual prompt templates
with agent_scratchpad placeholders. create_agent() builds a LangGraph
ReAct loop for you - you just hand it a model, tools, and a system
prompt. Each specialist here is one of those compiled ReAct loops,
added as a single node in the parent supervisor graph.
"""
from datetime import datetime
from typing import Literal

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel

from app.llm.client import get_chat_model
from app.tools.inventory_tool import check_inventory_gap, check_inventory_gap_for_sku, check_top_inventory_gap
from app.tools.knowledge_tool import search_sops

CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")

TOOLS_RETRIEVAL = [search_sops]
TOOLS_API = [check_inventory_gap, check_inventory_gap_for_sku, check_top_inventory_gap]

SUPERVISOR_SYSTEM_PROMPT = f"""You route each message to exactly one specialist - you never
answer the user directly and you never call tools yourself.

Current date: {CURRENT_DATE}

Route to retrieval_agent when the user asks how to do something in
the office, or what the procedure/policy is for something (SOPs).

Route to api_agent when the user asks about an inventory gap, stock
variance, or SCR (stock comparison report) - by storage location
(SLOC), by date, or by SKU.

Route to response_agent once enough information has already been
gathered to answer, or for anything that doesn't need a tool at all
(greetings, clarifying questions, general conversation).
"""

RETRIEVAL_SYSTEM_PROMPT = f"""You are the retrieval specialist on a data analytics assistant
for internal office use. Current date: {CURRENT_DATE}

Use search_sops to find relevant SOP (standard operating procedure)
excerpts for the user's question. Answer from those excerpts and cite
the source document title; if nothing relevant comes back, say so
instead of guessing. Keep tool calls minimal - search once, and only
search again if the first query clearly used the wrong terms.
"""

API_SYSTEM_PROMPT = f"""You are the inventory-data specialist on a data analytics assistant
for internal office use. Current date: {CURRENT_DATE}

You have three inventory-gap tools - check_inventory_gap,
check_inventory_gap_for_sku, and check_top_inventory_gap. Each tool's
own docstring explains exactly when to use it (by date/SLOC, by SKU,
or top gaps for a SLOC) - read those before picking one.

Keep tool calls minimal: describe once, then query directly. If a
query errors, read the error, fix the argument, and retry once. If
you still can't get it after one retry, explain what went wrong
instead of retrying indefinitely.
"""

RESPONSE_SYSTEM_PROMPT = """You are the response specialist on a data analytics assistant for
internal office use. You have no tools - you cannot call anything.

Given the user's question and whatever information other specialists
already gathered earlier in this conversation, write a clear,
well-formatted final answer. Synthesize and summarize - don't just
repeat raw tool output verbatim. Cite SOP document titles when
present. If the conversation doesn't contain enough information to
answer, say so plainly instead of guessing.
"""


class Route(BaseModel):
    next: Literal["retrieval_agent", "api_agent", "response_agent"]


class AgentGraphState(MessagesState):
    next: str


def _supervisor_node(state: AgentGraphState) -> dict:
    model = get_chat_model().with_structured_output(Route)
    messages = [SystemMessage(SUPERVISOR_SYSTEM_PROMPT), *state["messages"]]
    decision = model.invoke(messages)
    # Deliberately no "messages" key here - the routing decision is
    # bookkeeping, not conversation content. Keeping it out of
    # "messages" means the specialists and response_agent never see
    # it and can't mistake it for something the user or another agent
    # said.
    return {"next": decision.next}


def _response_node(state: AgentGraphState) -> dict:
    model = get_chat_model()
    messages = [SystemMessage(RESPONSE_SYSTEM_PROMPT), *state["messages"]]
    result = model.invoke(messages)
    return {"messages": [result]}


def build_graph(checkpointer=None):
    """
    Builds and compiles the supervisor + 3-specialist graph. Pass a
    checkpointer for multi-turn memory (see app/agent/memory.py); omit
    it for a one-shot, memory-less graph (see build_agent() below).
    """
    retrieval_agent = create_agent(
        model=get_chat_model(),
        tools=TOOLS_RETRIEVAL,
        system_prompt=RETRIEVAL_SYSTEM_PROMPT,
    )
    api_agent = create_agent(
        model=get_chat_model(),
        tools=TOOLS_API,
        system_prompt=API_SYSTEM_PROMPT,
    )

    graph = StateGraph(AgentGraphState)
    graph.add_node("supervisor", _supervisor_node)
    graph.add_node("retrieval_agent", retrieval_agent)
    graph.add_node("api_agent", api_agent)
    graph.add_node("response_agent", _response_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda state: state["next"],
        {
            "retrieval_agent": "retrieval_agent",
            "api_agent": "api_agent",
            "response_agent": "response_agent",
        },
    )
    graph.add_edge("retrieval_agent", "supervisor")
    graph.add_edge("api_agent", "supervisor")
    graph.add_edge("response_agent", END)

    return graph.compile(checkpointer=checkpointer)


def build_agent():
    """
    Returns a compiled, memory-less LangGraph graph. Call
    agent.invoke(...) or agent.stream(...) on the result - see
    app/agent/memory.py for the checkpointed (multi-turn) version used
    by the API.
    """
    return build_graph()
