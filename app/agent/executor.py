"""
LangChain 1.0 agent setup.

Note the API shift from pre-1.0 LangChain: there's no more
AgentExecutor + create_tool_calling_agent + manual prompt templates
with agent_scratchpad placeholders. create_agent() builds a LangGraph
ReAct loop for you - you just hand it a model, tools, and a system
prompt.
"""
from datetime import datetime
from langchain.agents import create_agent
from app.llm.client import get_chat_model
from app.tools.inventory_tool import check_inventory_gap, check_inventory_gap_for_sku, check_top_inventory_gap
from app.tools.knowledge_tool import search_sops




SYSTEM_PROMPT = f"""You are a data analytics assistant for internal office use.

Current date: {datetime.now().strftime('%Y-%m-%d')}
Always use this date as your current date when relevant 

Use check_inventory_gap when the user asks about an inventory gap or
stock variance for a specific storage location (SLOC) and date - this
calls the external inventory system directly and does not need a
dataset to be uploaded first.

Use search_sops when the user asks how to do something in the office
or what the procedure is for operation e.g. "how to quick set up inventory 
audit or how to do inventory adjustment. Answer from those excerpts and
cite the source document title; if nothing relevant comes back, say so
instead of guessing.

Keep tool calls minimal: describe once, then query directly. If a query
errors, read the error, fix the column name or syntax, and retry once.
If you still can't get it after one retry, explain what went wrong to
the user instead of retrying indefinitely.
"""

# Keep the toolset small and focused. Local models degrade in tool-call
# reliability once you hand them many tools at once - three well-scoped
# tools beats ten loosely-scoped ones.
TOOLS = [check_inventory_gap_for_sku,check_inventory_gap,check_top_inventory_gap, search_sops]


def build_agent():
    """
    Returns a compiled LangGraph agent. Call agent.invoke(...) or
    agent.stream(...) on the result - see app/api/routes.py for usage.
    """
    model = get_chat_model()
    agent = create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
    )
    return agent
