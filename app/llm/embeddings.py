"""
Wraps the self-hosted Ollama embedding model. Mirrors app/llm/client.py:
this is the only place that knows about Ollama specifically for
embeddings, so swapping providers later means changing only this file.
"""
from langchain_ollama import OllamaEmbeddings
from app.config import OLLAMA_BASE_URL, EMBEDDING_MODEL


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
