"""
Conversation memory for the agent.

create_agent() is built on LangGraph, so multi-turn memory works via a
checkpointer keyed by thread_id - not the old ConversationBufferMemory
object pattern from pre-1.0 LangChain. The supervisor graph in
app/agent/executor.py is checkpointed the same way; the retrieval_agent
and api_agent specialist sub-graphs are not checkpointed individually -
they're one-shot steps within a single turn, and cross-turn memory
lives entirely in the outer graph's checkpointer.

SqliteSaver persists checkpoints to a real file, so conversation state
survives process restarts (InMemorySaver loses everything on restart).
Still a single file on one box - if you later run multiple app
instances behind a load balancer, swap this for a Postgres checkpointer
(langgraph-checkpoint-postgres) so sessions survive across instances.
"""
import os
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from app.config import CHECKPOINT_DB_PATH
from app.agent.executor import build_graph

# Node whose streamed/final output is the true user-facing answer -
# retrieval_agent/api_agent produce their own wrap-up commentary too,
# but that's internal handoff text for the supervisor, not the answer.
_ANSWER_NODE = "response_agent"
_SPECIALIST_NODES = {"retrieval_agent", "api_agent"}


def build_agent_with_memory():
    checkpoint_dir = os.path.dirname(CHECKPOINT_DB_PATH)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()

    return build_graph(checkpointer=checkpointer)


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

      {"type": "tool_call", "tool": "<name>"}                 - a specialist started calling a tool
      {"type": "tool_result", "tool": "<name>", "content": str} - that tool's result
      {"type": "reasoning", "content": str}                    - thinking/reasoning tokens
                                                                    (only if include_reasoning=True)
      {"type": "answer", "content": str}                       - final answer tokens

    stream_mode="messages" yields (chunk, metadata) pairs for every LLM
    call inside the graph. Nested create_agent() nodes (retrieval_agent,
    api_agent) don't stream token-by-token by default - each shows up
    as one complete AIMessage (with a fully-formed .tool_calls list,
    not partial tool_call_chunks) followed by one complete ToolMessage,
    tagged with that specialist's own node name. Only the top-level
    response_agent node streams real token-by-token AIMessageChunks -
    that's the only node whose content becomes "answer" events; a
    specialist's own wrap-up commentary (e.g. "Found relevant SOP
    info.") is intentionally not treated as user-facing answer text.

    Reasoning tokens only show up in `additional_kwargs["reasoning_content"]`
    if the underlying ChatOllama call was made with reasoning=True (see
    app/llm/client.py) - with the default reasoning=None, "thinking"
    models still think, but the tokens aren't separated out and are
    silently skipped here instead (same as before this option existed).
    """
    config = {"configurable": {"thread_id": session_id}}

    for chunk, metadata in agent.stream(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
        stream_mode="messages",
    ):
        node = metadata.get("langgraph_node")

        if isinstance(chunk, ToolMessage) and node in _SPECIALIST_NODES:
            yield {"type": "tool_result", "tool": chunk.name, "content": chunk.content}
            continue

        if isinstance(chunk, AIMessage) and not isinstance(chunk, AIMessageChunk) and node in _SPECIALIST_NODES:
            for tool_call in chunk.tool_calls:
                yield {"type": "tool_call", "tool": tool_call["name"]}
            continue

        if not isinstance(chunk, AIMessageChunk) or node != _ANSWER_NODE:
            continue

        reasoning_text = chunk.additional_kwargs.get("reasoning_content")
        if include_reasoning and reasoning_text:
            yield {"type": "reasoning", "content": reasoning_text}
        elif chunk.content:
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

    A single turn can contain multiple "ai"-typed messages - a
    specialist's wrap-up commentary (e.g. from retrieval_agent) plus
    response_agent's real final answer, in that order. Only the last
    assistant message in each turn is kept: response_agent is always
    the graph's terminal node, so its message is always the last one
    written before the next human message (or end of history) - a
    specialist's earlier commentary is dropped rather than shown as a
    separate turn. This is a no-op for a turn that only ever produced
    one assistant message.
    """
    config = {"configurable": {"thread_id": session_id}}
    state = agent.get_state(config)
    messages = state.values.get("messages", []) if state.values else []

    history = []
    for msg in messages:
        role = _ROLE_BY_TYPE.get(msg.type)
        if not role or not msg.content:
            continue
        if role == "assistant" and history and history[-1]["role"] == "assistant":
            history[-1] = {"role": role, "content": msg.content}
        else:
            history.append({"role": role, "content": msg.content})
    return history
