"""
Wraps the self-hosted Ollama model as a LangChain chat model.
This is the ONLY place that knows about Ollama specifically - swapping
to vLLM, Bedrock, or a hosted API later means changing only this file.
"""
from langchain_ollama import ChatOllama
from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL, MODEL_TEMPERATURE


def get_chat_model() -> ChatOllama:
    """
    Returns a LangChain-compatible chat model backed by your self-hosted
    Ollama instance. bind_tools() and create_agent() both work against
    this the same way they would against ChatOpenAI or ChatAnthropic.
    """
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=MODEL_TEMPERATURE,

    )
