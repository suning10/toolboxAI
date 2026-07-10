# Office Analytics Agent

Self-hosted LangChain 1.0 agent over a self-hosted Ollama model, built to
answer questions over uploaded CSV/Excel data using pandas and SQL tools.

## Prerequisites

1. Ollama running and reachable (locally or on your EC2/Windows Server box)
2. A tool-calling-capable model pulled:
   ```
   ollama pull qwen2.5:14b
   ```
   Verify tool support before relying on it:
   ```
   ollama show qwen2.5:14b
   ```
   Look for `tools` under Capabilities. If it's missing, tool calls will
   silently fail or the model will just respond with text instead of
   invoking your functions.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env      # edit OLLAMA_BASE_URL / OLLAMA_MODEL / API_KEY as needed
```

## Run tests first (no model required)

```bash
pytest tests/ -v
```

This validates the pandas/SQL tools work correctly in isolation before
you debug them through the slower agent tool-calling loop.

## Run the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Try it

```bash
# 1. Upload a dataset
curl -X POST http://localhost:8000/upload \
  -H "x-api-key: change-me-in-production" \
  -F "file=@your_data.csv"

# 2. Ask a question
curl -X POST http://localhost:8000/chat \
  -H "x-api-key: change-me-in-production" \
  -H "Content-Type: application/json" \
  -d '{"message": "what are total sales by region?"}'
```

The response includes a `session_id` - pass it back on the next request
to continue the same conversation with memory.

## Project structure

```
app/
├── main.py              FastAPI entrypoint
├── config.py             env var loading
├── llm/client.py         Ollama connection (swap models/providers here only)
├── agent/
│   ├── executor.py        agent definition (create_agent + system prompt + tools)
│   └── memory.py          conversation memory via LangGraph checkpointer
├── tools/
│   ├── pandas_tool.py      dataframe query tool
│   └── sql_tool.py         SQL query tool (in-memory SQLite)
└── api/
    ├── routes.py           /upload, /chat, /health endpoints
    └── auth.py             API key check
```

## Build order this followed (recommended if extending)

1. Get `llm/client.py` talking to Ollama - test with a plain `.invoke("hi")`
2. Build and test tools in isolation (`tests/test_tools.py`) - no agent involved
3. Wire tools into `create_agent()` once each works standalone
4. Add memory (`agent/memory.py`)
5. Wrap in FastAPI

## Known limitations / next steps

- `run_pandas_query` uses a restricted `eval()`, not a real sandbox. Fine
  for a single trusted internal user; for multi-user production, move
  execution into a subprocess or container with resource limits.
- `InMemorySaver` for conversation memory means history is lost on
  restart and doesn't work across multiple app instances. Swap for a
  Redis or Postgres checkpointer (`langgraph-checkpoint-redis` /
  `langgraph-checkpoint-postgres`) if you scale beyond one process -
  you already have Redis experience, so that's a natural upgrade path.
- No RAG/vector store yet - add if the agent needs to search across many
  documents rather than just query one structured dataset at a time.
- No LangSmith tracing wired in - useful once you're debugging why the
  agent picked a specific tool call; set `LANGCHAIN_TRACING_V2=true` and
  `LANGCHAIN_API_KEY` env vars to enable it with zero code changes.
# toolboxAI
