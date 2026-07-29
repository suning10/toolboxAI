"""
Test that get_chat_model() builds the right LangChain chat model class
per MODEL_PROVIDER, and never passes temperature to Anthropic (which
rejects it with a 400 on current models) or Google (which degrades
reasoning quality if forced low on Gemini 3.0+) - no real API calls,
just verifying construction.

Run: pytest tests/test_llm_client.py -v
"""
from app.llm import client


def test_get_chat_model_defaults_to_ollama(monkeypatch):
    monkeypatch.setattr(client, "MODEL_PROVIDER", "ollama")
    from langchain_ollama import ChatOllama

    model = client.get_chat_model()
    assert isinstance(model, ChatOllama)


def test_get_chat_model_anthropic(monkeypatch):
    monkeypatch.setattr(client, "MODEL_PROVIDER", "anthropic")
    monkeypatch.setattr(client, "ANTHROPIC_API_KEY", "sk-test-fake-key")
    monkeypatch.setattr(client, "ANTHROPIC_MODEL", "claude-opus-4-8")
    from langchain_anthropic import ChatAnthropic

    model = client.get_chat_model()
    assert isinstance(model, ChatAnthropic)
    assert model.model == "claude-opus-4-8"
    # temperature must never be set explicitly - Opus 4.7+/4.8 reject it (400)
    assert model.temperature is None


def test_get_chat_model_openai(monkeypatch):
    monkeypatch.setattr(client, "MODEL_PROVIDER", "openai")
    monkeypatch.setattr(client, "OPENAI_API_KEY", "sk-test-fake-key")
    monkeypatch.setattr(client, "OPENAI_MODEL", "gpt-4.1")
    from langchain_openai import ChatOpenAI

    model = client.get_chat_model()
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-4.1"


def test_get_chat_model_google(monkeypatch):
    monkeypatch.setattr(client, "MODEL_PROVIDER", "google")
    monkeypatch.setattr(client, "GOOGLE_API_KEY", "sk-test-fake-key")
    monkeypatch.setattr(client, "GOOGLE_MODEL", "gemini-3.5-flash")
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = client.get_chat_model()
    assert isinstance(model, ChatGoogleGenerativeAI)
    assert model.model == "gemini-3.5-flash"
    # temperature must never be set explicitly - Gemini 3.0+ defaults to
    # 1.0 and forcing a low value degrades reasoning per Google's docs
    assert model.temperature is None


def test_get_chat_model_unknown_provider_raises(monkeypatch):
    monkeypatch.setattr(client, "MODEL_PROVIDER", "not-a-real-provider")
    try:
        client.get_chat_model()
        assert False, "expected ValueError"
    except ValueError as e:
        assert "not-a-real-provider" in str(e)
