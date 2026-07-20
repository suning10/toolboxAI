"""
Central configuration. Reads from environment variables (.env locally,
real env vars on EC2/Windows Server).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Point this at your Ollama endpoint - localhost if Ollama runs on the same
# box as the app, or the EC2 private IP / DNS if the app and model are on
# separate hosts.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Must be a model you've pulled AND that supports tool calling.
# Run `ollama show <model>` and check for "tools" under Capabilities
# before assuming it works. Qwen2.5/Qwen3 (14B+) are currently the most
# reliable open models for tool calling.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.0:14b")

# Set temperature low for analytics/tool-calling tasks - you want
# deterministic tool selection, not creative variation.
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0"))

# Simple API key check for the FastAPI layer. Replace with real auth
# (OAuth, corporate SSO, etc.) before exposing beyond localhost.
API_KEY = os.getenv("API_KEY", "change-me-in-production")

DATA_UPLOAD_DIR = os.getenv("DATA_UPLOAD_DIR", "data/uploads")

# SQLite by default - stores application metadata (e.g. chat session
# records, see app/models/chat_session.py). Point at Postgres
# ("postgresql://...") if you scale beyond one instance.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/app.db")

# Separate SQLite file for LangGraph's own checkpoint storage (the full
# conversation state). Kept apart from DATABASE_URL because LangGraph
# manages this schema itself via SqliteSaver, not through our models.
CHECKPOINT_DB_PATH = os.getenv("CHECKPOINT_DB_PATH", "data/checkpoints.sqlite")

# External inventory system's REST API, used by
# app/tools/inventory_tool.py to check stock gaps by date + SLOC.
# PLACEHOLDER - point this at the real inventory service and update the
# request/response contract in inventory_tool.py to match its API.
INVENTORY_API_BASE_URL = os.getenv("INVENTORY_API_BASE_URL", "http://localhost:9000/api")
INVENTORY_API_KEY = os.getenv("INVENTORY_API_KEY", "")

# SOP knowledge base (RAG over office procedure docs) - see
# app/knowledge/ and app/tools/knowledge_tool.py.
# Must be an embedding model you've pulled: `ollama pull nomic-embed-text`.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))

SOP_DOCS_DIR = os.getenv("SOP_DOCS_DIR", "data/sops")
# Separate SQLite file (sqlite-vec extension) holding chunk text +
# embeddings for the SOP knowledge base - kept apart from DATABASE_URL
# since it's queried via raw sqlite3 + the vec0 virtual table, not SQLAlchemy.
SOP_VECTOR_DB_PATH = os.getenv("SOP_VECTOR_DB_PATH", "data/sop_vectors.sqlite")

SOP_CHUNK_SIZE = int(os.getenv("SOP_CHUNK_SIZE", "800"))
SOP_CHUNK_OVERLAP = int(os.getenv("SOP_CHUNK_OVERLAP", "150"))
SOP_SEARCH_TOP_K = int(os.getenv("SOP_SEARCH_TOP_K", "4"))
