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
curl -X POST http://localhost:8000/api/v1/upload \
  -H "x-api-key: change-me-in-production" \
  -F "file=@your_data.csv"

# 2. Ask a question
curl -X POST http://localhost:8000/api/v1/chat \
  -H "x-api-key: change-me-in-production" \
  -H "Content-Type: application/json" \
  -d '{"message": "what are total sales by region?"}'
```

`/health` is unversioned (no `/api/v1` prefix) since load balancers and
container orchestrators expect a stable health path.

The response includes a `session_id` - pass it back on the next request
to continue the same conversation with memory.

```bash
# List known sessions / inspect one
curl http://localhost:8000/api/v1/chat/sessions -H "x-api-key: change-me-in-production"
curl http://localhost:8000/api/v1/chat/sessions/<session_id> -H "x-api-key: change-me-in-production"
```

## Project structure

```
app/
├── main.py              FastAPI entrypoint - mounts /health and /api/v1
├── config.py             env var loading
├── llm/client.py         Ollama connection (swap models/providers here only)
├── agent/
│   ├── executor.py        agent definition (create_agent + system prompt + tools)
│   └── memory.py          conversation memory via LangGraph checkpointer
├── tools/
│   ├── pandas_tool.py      dataframe query tool
│   ├── sql_tool.py         SQL query tool (in-memory SQLite)
│   └── inventory_tool.py   inventory gap check by date + SLOC (calls external REST API)
├── db/
│   └── session.py          SQLAlchemy engine/session + init_db()
├── models/
│   └── chat_session.py     ChatSession entity (SQLAlchemy)
├── services/
│   └── chat_session_service.py   session create/touch/list/get
├── schemas/
│   ├── chat.py             ChatRequest / ChatResponse
│   ├── chat_session.py     ChatSessionRead (entity -> API mapping)
│   └── upload.py           UploadResponse
└── api/
    ├── health.py           unversioned GET /health
    ├── deps.py             API key auth dependency
    ├── exceptions.py       centralized error handler (no leaked tracebacks)
    └── v1/
        ├── router.py       aggregates endpoint routers under /api/v1
        └── endpoints/
            ├── upload.py    POST /api/v1/upload
            └── chat.py      POST/GET /api/v1/chat, /api/v1/chat/sessions[/{id}]
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
- Conversation memory now persists to a SQLite file (`CHECKPOINT_DB_PATH`,
  via `langgraph-checkpoint-sqlite`) instead of `InMemorySaver`, so it
  survives restarts - but it's still one file on one box. Swap for
  `langgraph-checkpoint-postgres` if you run multiple app instances
  behind a load balancer.
- `ChatSession` metadata (`app/models/chat_session.py`) also lives in
  SQLite (`DATABASE_URL`) with tables created via `Base.metadata.create_all()`
  at startup. Move to Alembic migrations once the schema needs versioned
  changes rather than just new tables/columns.
- `check_inventory_gap` (`app/tools/inventory_tool.py`) is wired against a
  **placeholder** contract - path, auth header, and response field names
  in that file, plus `INVENTORY_API_BASE_URL`/`INVENTORY_API_KEY` in
  `.env.example`, need to be updated to match the real inventory API.
- No RAG/vector store yet - add if the agent needs to search across many
  documents rather than just query one structured dataset at a time.
- No LangSmith tracing wired in - useful once you're debugging why the
  agent picked a specific tool call; set `LANGCHAIN_TRACING_V2=true` and
  `LANGCHAIN_API_KEY` env vars to enable it with zero code changes.
# toolboxAI
