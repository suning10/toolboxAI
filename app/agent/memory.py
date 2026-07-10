"""
Conversation memory for the agent.

create_agent() is built on LangGraph, so multi-turn memory works via a
checkpointer keyed by thread_id - not the old ConversationBufferMemory
object pattern from pre-1.0 LangChain.

InMemorySaver is fine for a single-process, low-volume deployment
(your <1000 calls/day case). If you later run multiple app instances
behind a load balancer, swap this for a Redis or Postgres checkpointer
so sessions survive across instances/restarts.
"""
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from app.llm.client import get_chat_model
from app.tools.pandas_tool import describe_dataset, run_pandas_query
from app.tools.sql_tool import run_sql_query
from app.agent.executor import SYSTEM_PROMPT, TOOLS


def build_agent_with_memory():
    model = get_chat_model()
    checkpointer = InMemorySaver()
    agent = create_agent(
        model=model,
        tools=TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
    return agent


def invoke_with_history(agent, message: str, session_id: str):
    """
    session_id ties a conversation to a specific user/session -
    e.g. a browser session ID or API caller's user ID.
    """
    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
    )
    return result["messages"][-1].content
