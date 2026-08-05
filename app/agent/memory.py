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
from langchain_core.messages import AIMessageChunk, ToolMessage
from app.config import CHECKPOINT_DB_PATH
from app.llm.client import get_chat_model
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
#        middleware=
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


def stream_agent_events(agent, message: str, session_id: str, include_reasoning: bool = False):
    """
    Rich event stream for a single chat turn - yields dicts describing
    everything the agent does, not just the final answer text:

      {"type": "tool_call", "tool": "<name>"}                 - agent started calling a tool
      {"type": "tool_result", "tool": "<name>", "content": str} - that tool's result
      {"type": "reasoning", "content": str}                    - thinking/reasoning tokens
                                                                    (only if include_reasoning=True)
      {"type": "answer", "content": str}                       - final answer tokens

    stream_mode="messages" yields (chunk, metadata) pairs for every LLM
    call inside the graph, not just the final answer. Tool calls stream
    in as partial tool_call_chunks (accumulating name/args across
    several chunks) - each is announced once, the first time its name
    appears, keyed by id (falling back to index if a provider omits
    id). The tool's result itself arrives as one complete ToolMessage,
    not a stream of chunks.

    Reasoning tokens only show up in `additional_kwargs["reasoning_content"]`
    if the underlying ChatOllama call was made with reasoning=True (see
    app/llm/client.py) - with the default reasoning=None, "thinking"
    models still think, but the tokens aren't separated out and are
    silently skipped here instead (same as before this option existed).
    """
    config = {"configurable": {"thread_id": session_id}}
    announced_tool_calls: set = set()

    for chunk, metadata in agent.stream(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
        stream_mode="messages",
    ):
        if isinstance(chunk, ToolMessage) and metadata.get("langgraph_node") == "tools":
            yield {"type": "tool_result", "tool": chunk.name, "content": chunk.content}
            continue

        if not isinstance(chunk, AIMessageChunk) or metadata.get("langgraph_node") != "model":
            continue

        for tool_call_chunk in chunk.tool_call_chunks:
            call_key = tool_call_chunk.get("id") or tool_call_chunk.get("index")
            name = tool_call_chunk.get("name")
            if name and call_key not in announced_tool_calls:
                announced_tool_calls.add(call_key)
                yield {"type": "tool_call", "tool": name}

        reasoning_text = chunk.additional_kwargs.get("reasoning_content")
        if include_reasoning and reasoning_text:
            yield {"type": "reasoning", "content": reasoning_text}
        elif not chunk.tool_call_chunks and chunk.content:
            yield {"type": "answer", "content": chunk.content}


def stream_with_history(agent, message: str, session_id: str):
    """
    Same contract as invoke_with_history, but yields the final answer's
    content token-by-token instead of returning it all at once. Thin
    wrapper over stream_agent_events for callers that only want plain
    answer text - see stream_agent_events for tool-call/reasoning events.
    """
    for event in stream_agent_events(agent, message, session_id):
        if event["type"] == "answer":
            yield event["content"]


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
