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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")

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
