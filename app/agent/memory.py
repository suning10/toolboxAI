"""
Conversation memory for the agent.

create_agent() is built on LangGraph, so multi-turn memory works via a
checkpointer keyed by thread_id - not the old ConversationBufferMemory
object pattern from pre-1.0 LangChain.

SqliteSaver persists checkpoints to a real file, so conversation state
survives process restarts (InMemorySaver loses everything on restart).
Still a single file on one box - if you later run multiple app
instances behind a load balancer, swap this for a Postgres checkpointer
(langgraph-checkpoint-postgres) so sessions survive across instances.
"""
import os
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langchain.agents import create_agent
from app.config import CHECKPOINT_DB_PATH
from app.llm.client import get_chat_model
from app.tools.pandas_tool import describe_dataset, run_pandas_query
from app.tools.sql_tool import run_sql_query
from app.agent.executor import SYSTEM_PROMPT, TOOLS


def build_agent_with_memory():
    model = get_chat_model()

    checkpoint_dir = os.path.dirname(CHECKPOINT_DB_PATH)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()

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


# Maps LangChain's internal message.type to the role names a frontend
# chat UI expects. Tool-call messages are deliberately excluded - a
# ChatGPT-style history view only shows the user/assistant turns.
_ROLE_BY_TYPE = {"human": "user", "ai": "assistant"}


def get_message_history(agent, session_id: str) -> list[dict]:
    """
    Returns the full conversation for session_id as a flat list of
    {"role": "user"|"assistant", "content": str} dicts, oldest first -
    read from the checkpointer's stored graph state, not re-run through
    the model. Empty list if the session doesn't exist or has no
    checkpoint yet.
    """
    config = {"configurable": {"thread_id": session_id}}
    state = agent.get_state(config)
    messages = state.values.get("messages", []) if state.values else []

    history = []
    for msg in messages:
        role = _ROLE_BY_TYPE.get(msg.type)
        if role and msg.content:
            history.append({"role": role, "content": msg.content})
    return history
