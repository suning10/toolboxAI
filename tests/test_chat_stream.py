"""
Test the multi-agent graph's streaming (stream_agent_events,
stream_with_history) and blocking (invoke_with_history) paths with
fake chat models - no Ollama required, same rationale as the other
tool tests.

Doesn't go through app.agent.executor.build_graph() directly, since
that always calls get_chat_model() (hardcoded to Ollama) for every
node - instead it rebuilds the exact same graph topology here with
fake models injected directly, reusing the real AgentGraphState/Route
schema from app.agent.executor so the wiring under test matches
production.

Run: pytest tests/test_chat_stream.py -v
"""
import sqlite3
from typing import Iterator

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from app.agent.executor import AgentGraphState, Route
from app.agent.memory import invoke_with_history, stream_agent_events, stream_with_history


@tool
def dummy_retrieval_tool(query: str) -> str:
    """A dummy retrieval tool."""
    return "retrieved: some SOP text"


@tool
def dummy_api_tool(sloc: str) -> str:
    """A dummy api tool."""
    return "api result: gap=5"


class _FakeSupervisorModel:
    """Not a BaseChatModel - stands in for get_chat_model().with_structured_output(Route),
    returning pre-scripted routing decisions in order, one per call."""

    def __init__(self, routes: list[str]):
        self._routes = iter(routes)

    def invoke(self, messages):
        return Route(next=next(self._routes))


class _FakeSpecialistModel(BaseChatModel):
    """First call: calls its one tool. Second call: a short wrap-up
    message - mimics a create_agent() ReAct loop's two turns."""

    tool_name: str
    tool_args: dict
    wrapup: str
    call_count: int = 0

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "fake-specialist"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.call_count += 1
        if self.call_count == 1:
            msg = AIMessage(
                content="",
                tool_calls=[{"name": self.tool_name, "args": self.tool_args, "id": "call1"}],
            )
        else:
            msg = AIMessage(content=self.wrapup)
        return ChatResult(generations=[ChatGeneration(message=msg)])


class _FakeResponseModel(BaseChatModel):
    """Streams real token-by-token content, optionally with reasoning
    tokens interleaved first - mimics response_agent's plain model call."""

    reply: str
    reasoning: str | None = None

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "fake-response"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])

    def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]:
        if self.reasoning:
            for word in self.reasoning.split(" "):
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content="", additional_kwargs={"reasoning_content": word + " "})
                )
        for word in self.reply.split(" "):
            yield ChatGenerationChunk(message=AIMessageChunk(content=word + " "))


def _build_fake_graph(supervisor_model, response_model, retrieval_model=None, api_model=None):
    retrieval_agent = create_agent(
        model=retrieval_model
        or _FakeSpecialistModel(tool_name="dummy_retrieval_tool", tool_args={"query": "q"}, wrapup="unused"),
        tools=[dummy_retrieval_tool],
        system_prompt="retrieval",
    )
    api_agent = create_agent(
        model=api_model
        or _FakeSpecialistModel(tool_name="dummy_api_tool", tool_args={"sloc": "WC1E"}, wrapup="unused"),
        tools=[dummy_api_tool],
        system_prompt="api",
    )

    def supervisor_node(state):
        decision = supervisor_model.invoke(state["messages"])
        return {"next": decision.next}

    def response_node(state):
        result = response_model.invoke(state["messages"])
        return {"messages": [result]}

    graph = StateGraph(AgentGraphState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("retrieval_agent", retrieval_agent)
    graph.add_node("api_agent", api_agent)
    graph.add_node("response_agent", response_node)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        lambda s: s["next"],
        {"retrieval_agent": "retrieval_agent", "api_agent": "api_agent", "response_agent": "response_agent"},
    )
    graph.add_edge("retrieval_agent", "supervisor")
    graph.add_edge("api_agent", "supervisor")
    graph.add_edge("response_agent", END)

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    return graph.compile(checkpointer=checkpointer)


def test_routes_straight_to_response_when_no_tool_needed():
    agent = _build_fake_graph(
        supervisor_model=_FakeSupervisorModel(["response_agent"]),
        response_model=_FakeResponseModel(reply="Hello there."),
    )

    events = list(stream_agent_events(agent, "hi", "s1"))
    types = [e["type"] for e in events]
    assert "tool_call" not in types
    assert "tool_result" not in types
    assert types == ["answer"] * len(types)
    assert "".join(e["content"] for e in events) == "Hello there. "


def test_routes_through_retrieval_then_response():
    agent = _build_fake_graph(
        supervisor_model=_FakeSupervisorModel(["retrieval_agent", "response_agent"]),
        retrieval_model=_FakeSpecialistModel(
            tool_name="dummy_retrieval_tool", tool_args={"query": "badge"}, wrapup="Found relevant SOP info."
        ),
        response_model=_FakeResponseModel(reply="Here is the SOP answer."),
    )

    events = list(stream_agent_events(agent, "how do I badge in?", "s1"))
    types = [e["type"] for e in events]

    assert types[0] == "tool_call"
    assert events[0] == {"type": "tool_call", "tool": "dummy_retrieval_tool"}
    assert types[1] == "tool_result"
    assert events[1] == {"type": "tool_result", "tool": "dummy_retrieval_tool", "content": "retrieved: some SOP text"}
    assert all(t == "answer" for t in types[2:])

    answer_text = "".join(e["content"] for e in events if e["type"] == "answer")
    assert answer_text == "Here is the SOP answer. "
    # the retrieval specialist's own wrap-up commentary must never leak
    # into the user-facing answer stream
    assert "Found relevant SOP info." not in answer_text


def test_routes_through_api_agent():
    agent = _build_fake_graph(
        supervisor_model=_FakeSupervisorModel(["api_agent", "response_agent"]),
        api_model=_FakeSpecialistModel(tool_name="dummy_api_tool", tool_args={"sloc": "WC1E"}, wrapup="Gap found."),
        response_model=_FakeResponseModel(reply="The gap is 5."),
    )

    events = list(stream_agent_events(agent, "what's the gap for WC1E?", "s1"))
    assert events[0] == {"type": "tool_call", "tool": "dummy_api_tool"}
    assert events[1] == {"type": "tool_result", "tool": "dummy_api_tool", "content": "api result: gap=5"}
    assert "".join(e["content"] for e in events if e["type"] == "answer") == "The gap is 5. "


def test_reasoning_only_surfaces_from_response_agent():
    agent = _build_fake_graph(
        supervisor_model=_FakeSupervisorModel(["response_agent"]),
        response_model=_FakeResponseModel(reply="Final answer.", reasoning="Let me think."),
    )

    events = list(stream_agent_events(agent, "hi", "s1", include_reasoning=True))
    types = [e["type"] for e in events]
    assert types[0] == "reasoning"
    assert "".join(e["content"] for e in events if e["type"] == "reasoning") == "Let me think. "
    assert "".join(e["content"] for e in events if e["type"] == "answer") == "Final answer. "


def test_reasoning_hidden_by_default():
    agent = _build_fake_graph(
        supervisor_model=_FakeSupervisorModel(["response_agent"]),
        response_model=_FakeResponseModel(reply="Final answer.", reasoning="Let me think."),
    )

    events = list(stream_agent_events(agent, "hi", "s1"))
    assert "reasoning" not in [e["type"] for e in events]


def test_stream_with_history_yields_only_answer_text():
    agent = _build_fake_graph(
        supervisor_model=_FakeSupervisorModel(["retrieval_agent", "response_agent"]),
        retrieval_model=_FakeSpecialistModel(
            tool_name="dummy_retrieval_tool", tool_args={"query": "q"}, wrapup="internal note"
        ),
        response_model=_FakeResponseModel(reply="Clean final answer."),
    )

    tokens = list(stream_with_history(agent, "hi", "s1"))
    assert "".join(tokens) == "Clean final answer. "


def test_invoke_with_history_returns_response_agent_output():
    agent = _build_fake_graph(
        supervisor_model=_FakeSupervisorModel(["api_agent", "response_agent"]),
        api_model=_FakeSpecialistModel(tool_name="dummy_api_tool", tool_args={"sloc": "WC1E"}, wrapup="internal note"),
        response_model=_FakeResponseModel(reply="Blocking final answer."),
    )

    answer = invoke_with_history(agent, "hi", "s1")
    assert answer == "Blocking final answer."
