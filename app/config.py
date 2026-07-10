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
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

# Set temperature low for analytics/tool-calling tasks - you want
# deterministic tool selection, not creative variation.
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0"))

# Simple API key check for the FastAPI layer. Replace with real auth
# (OAuth, corporate SSO, etc.) before exposing beyond localhost.
API_KEY = os.getenv("API_KEY", "change-me-in-production")

DATA_UPLOAD_DIR = os.getenv("DATA_UPLOAD_DIR", "data/uploads")
