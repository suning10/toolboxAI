"""
Test conversation history retrieval (title-from-first-message and
get_message_history) with a fake chat model - no Ollama required,
same rationale as the other tool tests.

Run: pytest tests/test_chat_history.py -v
"""
import sqlite3

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph

from app.agent.memory import get_message_history
from app.services.chat_session_service import _title_from_message, touch_session


@pytest.fixture
def fake_agent():
    model = FakeListChatModel(responses=["Hi there!", "Sure, 42."])
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    return create_agent(model=model, tools=[], system_prompt="test", checkpointer=checkpointer)


def test_get_message_history_returns_user_and_assistant_turns(fake_agent):
    config = {"configurable": {"thread_id": "s1"}}
    fake_agent.invoke({"messages": [{"role": "user", "content": "hello"}]}, config=config)
    fake_agent.invoke({"messages": [{"role": "user", "content": "what is the answer?"}]}, config=config)

    history = get_message_history(fake_agent, "s1")
    assert history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "what is the answer?"},
        {"role": "assistant", "content": "Sure, 42."},
    ]


def test_get_message_history_empty_for_unknown_session(fake_agent):
    assert get_message_history(fake_agent, "never-existed") == []


def test_get_message_history_collapses_multiple_assistant_messages_per_turn():
    """A turn that produces several AI messages (e.g. a specialist's
    wrap-up commentary, then response_agent's real final answer) must
    only surface the last one - see app/agent/memory.py's
    get_message_history docstring."""

    def specialist_wrapup_node(state):
        return {"messages": [AIMessage(content="internal specialist commentary")]}

    def response_node(state):
        return {"messages": [AIMessage(content="the real final answer")]}

    graph = StateGraph(MessagesState)
    graph.add_node("specialist", specialist_wrapup_node)
    graph.add_node("response", response_node)
    graph.add_edge(START, "specialist")
    graph.add_edge("specialist", "response")
    graph.add_edge("response", END)

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    agent = graph.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "s1"}}
    agent.invoke({"messages": [{"role": "user", "content": "first turn"}]}, config=config)
    agent.invoke({"messages": [{"role": "user", "content": "second turn"}]}, config=config)

    history = get_message_history(agent, "s1")
    assert history == [
        {"role": "user", "content": "first turn"},
        {"role": "assistant", "content": "the real final answer"},
        {"role": "user", "content": "second turn"},
        {"role": "assistant", "content": "the real final answer"},
    ]


def test_title_from_message_truncates_long_text():
    long_message = "word " * 30
    title = _title_from_message(long_message)
    assert title.endswith("...")
    assert len(title) <= 63


def test_title_from_message_keeps_short_text_as_is():
    assert _title_from_message("how do I request a badge?") == "how do I request a badge?"


def test_touch_session_sets_title_only_on_creation(tmp_path, monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.db.session import Base

    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()

    session = touch_session(db, "s1", first_message="How do I reset my password?")
    assert session.title == "How do I reset my password?"

    session = touch_session(db, "s1", first_message="this should be ignored")
    assert session.title == "How do I reset my password?"
    assert session.message_count == 2
