"""
Builds the LangChain chat model this app runs on. This is the ONLY
place that knows which provider is in use - agent/tool/streaming code
just gets a LangChain BaseChatModel back and works the same regardless
of MODEL_PROVIDER (see app/config.py).

Switching providers later: set MODEL_PROVIDER + that provider's API
key/model env vars, nothing else changes. One caveat that doesn't
generalize automatically: app/agent/memory.py's stream_agent_events
surfaces reasoning/thinking tokens by reading
chunk.additional_kwargs["reasoning_content"], which is specific to how
langchain-ollama exposes ChatOllama's reasoning=True. Anthropic,
OpenAI, and Google's reasoning/thinking content all show up
differently (Anthropic: separate "thinking"-type content blocks;
OpenAI: not exposed via the Chat Completions API at all; Gemini:
separate "thought" content parts) - include_reasoning currently only
works end-to-end against Ollama.
"""
from langchain_core.language_models.chat_models import BaseChatModel

from app.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    GOOGLE_API_KEY,
    GOOGLE_MODEL,
    MODEL_PROVIDER,
    MODEL_TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)


def get_chat_model() -> BaseChatModel:
    if MODEL_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=MODEL_TEMPERATURE,
            # Cleanly separates a "thinking" model's reasoning into
            # additional_kwargs["reasoning_content"] instead of leaving
            # it mixed into the main response content. Only affects
            # models that support reasoning
            # (https://ollama.com/search?c=thinking) - ignored
            # otherwise. Whether reasoning is actually *shown* to a
            # caller is a per-request choice (see stream_agent_events'
            # include_reasoning param), not controlled here.
            reasoning=True,
        )

    if MODEL_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # No temperature param - Claude Opus 4.7+/4.8/Sonnet 5 reject a
        # temperature override with a 400 rather than ignoring it.
        kwargs = {"model": ANTHROPIC_MODEL}
        if ANTHROPIC_API_KEY:
            kwargs["api_key"] = ANTHROPIC_API_KEY
        return ChatAnthropic(**kwargs)

    if MODEL_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        kwargs = {"model": OPENAI_MODEL}
        if OPENAI_API_KEY:
            kwargs["api_key"] = OPENAI_API_KEY
        return ChatOpenAI(**kwargs)

    if MODEL_PROVIDER == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # No temperature param - Gemini 3.0+ defaults to temperature=1.0
        # when unset; Google's own docs warn that forcing a low value
        # (0 or 0.7) "can cause infinite loops, degraded reasoning
        # performance, and failure on complex tasks."
        kwargs = {"model": GOOGLE_MODEL}
        if GOOGLE_API_KEY:
            kwargs["api_key"] = GOOGLE_API_KEY
        return ChatGoogleGenerativeAI(**kwargs)

    raise ValueError(
        f"Unknown MODEL_PROVIDER '{MODEL_PROVIDER}' - use 'ollama', 'anthropic', 'openai', or 'google'."
    )
