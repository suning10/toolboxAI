"""
Test token-by-token streaming (stream_with_history) with fake chat
models - no Ollama required, same rationale as the other tool tests.

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

from app.agent.memory import stream_with_history


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
