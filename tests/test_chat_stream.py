"""
Test token-by-token streaming (stream_with_history, stream_agent_events)
with fake chat models - no Ollama required, same rationale as the
other tool tests.

Run: pytest tests/test_chat_stream.py -v
"""
import sqlite3
from typing import Iterator

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver

from app.agent.memory import stream_agent_events, stream_with_history


@pytest.fixture
def fake_agent():
    model = FakeListChatModel(responses=["Hello world this is a test response"])
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    return create_agent(model=model, tools=[], system_prompt="test", checkpointer=checkpointer)


def test_stream_with_history_yields_tokens_in_order(fake_agent):
    tokens = list(stream_with_history(fake_agent, "hi", "s1"))
    assert "".join(tokens) == "Hello world this is a test response"
    assert len(tokens) > 1  # actually streamed, not one big chunk


class _FakeToolCallingModel(BaseChatModel):
    """Emits a tool-call chunk on its first call, then real answer
    tokens on its second call - mimics a ReAct loop's two model turns
    so the stream_with_history filter can be tested against both."""

    call_count: int = 0

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError

    def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]:
        self.call_count += 1
        if self.call_count == 1:
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[{"name": "dummy_tool", "args": "{}", "id": "call1", "index": 0}],
                )
            )
        else:
            for word in ["Final ", "answer ", "here"]:
                yield ChatGenerationChunk(message=AIMessageChunk(content=word))


@tool
def dummy_tool() -> str:
    """A dummy tool."""
    return "tool result"


def test_stream_with_history_excludes_tool_call_chunks():
    model = _FakeToolCallingModel()
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    agent = create_agent(model=model, tools=[dummy_tool], system_prompt="test", checkpointer=checkpointer)

    tokens = list(stream_with_history(agent, "hi", "s1"))
    assert "".join(tokens) == "Final answer here"


def test_stream_agent_events_surfaces_tool_call_and_result():
    model = _FakeToolCallingModel()
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    agent = create_agent(model=model, tools=[dummy_tool], system_prompt="test", checkpointer=checkpointer)

    events = list(stream_agent_events(agent, "hi", "s1"))
    types = [e["type"] for e in events]
    assert types == ["tool_call", "tool_result", "answer", "answer", "answer"]
    assert events[0] == {"type": "tool_call", "tool": "dummy_tool"}
    assert events[1] == {"type": "tool_result", "tool": "dummy_tool", "content": "tool result"}
    assert "".join(e["content"] for e in events if e["type"] == "answer") == "Final answer here"


class _FakeReasoningToolModel(BaseChatModel):
    """First call: thinking tokens, then a tool call split across two
    chunks (name arrives on one chunk, args on the next, same id) -
    exercises both reasoning surfacing and tool-call de-duplication.
    Second call: plain answer tokens."""

    call_count: int = 0

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "fake-reasoning-tool"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise NotImplementedError

    def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]:
        self.call_count += 1
        if self.call_count == 1:
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="", additional_kwargs={"reasoning_content": "Let me "})
            )
            yield ChatGenerationChunk(
                message=AIMessageChunk(content="", additional_kwargs={"reasoning_content": "check the tool."})
            )
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[{"name": "dummy_tool", "args": "", "id": "call1", "index": 0}],
                )
            )
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[{"name": None, "args": "{}", "id": "call1", "index": 0}],
                )
            )
        else:
            for word in ["Final ", "answer ", "here"]:
                yield ChatGenerationChunk(message=AIMessageChunk(content=word))


def test_stream_agent_events_include_reasoning_true():
    model = _FakeReasoningToolModel()
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    agent = create_agent(model=model, tools=[dummy_tool], system_prompt="test", checkpointer=checkpointer)

    events = list(stream_agent_events(agent, "hi", "s1", include_reasoning=True))
    types = [e["type"] for e in events]
    assert types == ["reasoning", "reasoning", "tool_call", "tool_result", "answer", "answer", "answer"]
    assert "".join(e["content"] for e in events if e["type"] == "reasoning") == "Let me check the tool."
    # tool_call announced exactly once even though its chunks span two deltas
    assert types.count("tool_call") == 1


def test_stream_agent_events_include_reasoning_false_by_default():
    model = _FakeReasoningToolModel()
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    agent = create_agent(model=model, tools=[dummy_tool], system_prompt="test", checkpointer=checkpointer)

    events = list(stream_agent_events(agent, "hi", "s1"))
    types = [e["type"] for e in events]
    assert "reasoning" not in types
    assert types == ["tool_call", "tool_result", "answer", "answer", "answer"]
