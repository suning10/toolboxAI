"""
Central configuration. Reads from environment variables (.env locally,
real env vars on EC2/Windows Server).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Which chat model provider app/llm/client.py builds - "ollama" (default,
# self-hosted/free), "anthropic", "openai", or "google". Switching
# providers later is just changing this one var (plus that provider's
# API key) - nothing else in the app (agent, tools, streaming) is
# provider-specific.
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "ollama")

# Point this at your Ollama endpoint - localhost if Ollama runs on the same
# box as the app, or the EC2 private IP / DNS if the app and model are on
# separate hosts.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Must be a model you've pulled AND that supports tool calling.
# Run `ollama show <model>` and check for "tools" under Capabilities
# before assuming it works. Qwen2.5/Qwen3 (14B+) are currently the most
# reliable open models for tool calling.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:0.8b")

# Only used when MODEL_PROVIDER=anthropic. Reads ANTHROPIC_API_KEY from
# the environment automatically if unset here - see langchain-anthropic's
# ChatAnthropic. claude-opus-4-8 is Anthropic's current recommended
# default model as of this writing; check platform.claude.com for
# updates before assuming it's still current.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-8")

# Only used when MODEL_PROVIDER=openai. Reads OPENAI_API_KEY from the
# environment automatically if unset here. gpt-4.1 is a conservative,
# known-stable default - check platform.openai.com/docs/models for
# OpenAI's current recommended flagship before assuming it's still best.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")

# Only used when MODEL_PROVIDER=google. Reads GOOGLE_API_KEY from the
# environment automatically if unset here - see langchain-google-genai's
# ChatGoogleGenerativeAI. Check ai.google.dev/gemini-api/docs/models for
# Google's current recommended model before assuming gemini-3.5-flash
# is still current.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-3.5-flash")

# Set temperature low for analytics/tool-calling tasks - you want
# deterministic tool selection, not creative variation. Only applied
# for MODEL_PROVIDER=ollama:
# - Claude Opus 4.7+/4.8/Sonnet 5 reject a temperature override outright
#   (400 error) rather than ignoring it.
# - Gemini 3.0+ models default to temperature=1.0 when unset, and
#   Google's own docs warn that forcing a low value like 0 or 0.7 "can
#   cause infinite loops, degraded reasoning performance, and failure
#   on complex tasks."
# Recent OpenAI reasoning models are similarly restrictive. app/llm/client.py
# never passes this param to ChatAnthropic/ChatGoogleGenerativeAI/ChatOpenAI.
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

# Root log level for the app's own loggers (app.*) - see
# app/logging_config.py. DEBUG is noisy (logs tool inputs/outputs);
# INFO is the usual default; use WARNING in production if you only
# want problems surfaced.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# MySQL access via the mysql_mcp_server MCP server (spawned over stdio
# through `uvx`, see app/tools/mysql_mcp_tool.py). Leave MYSQL_HOST unset
# to skip loading MySQL tools entirely - the app runs fine without it,
# same as the other optional integrations above.
MYSQL_HOST = os.getenv("MYSQL_HOST", "")
MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
MYSQL_USER = os.getenv("MYSQL_USER", "")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "")
