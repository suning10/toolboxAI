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
cp .env .env      # edit OLLAMA_BASE_URL / OLLAMA_MODEL / API_KEY as needed
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
curl -X POST http://localhost:8000/admin/upload \
  -H "x-api-key: change-me-in-production" \
  -F "file=@your_data.csv"

# 2. Ask a question
curl -X POST http://localhost:8000/admin/chat \
  -H "x-api-key: change-me-in-production" \
  -H "Content-Type: application/json" \
  -d '{"message": "what are total sales by region?"}'
```

`/health` is unversioned (no `/admin` prefix) since load balancers and
container orchestrators expect a stable health path.

The response includes a `session_id` - pass it back on the next request
to continue the same conversation with memory.

```bash
# List known sessions / inspect one
curl http://localhost:8000/admin/ai/chat/sessions -H "x-api-key: change-me-in-production"
curl http://localhost:8000/admin/ai/chat/sessions/<session_id> -H "x-api-key: change-me-in-production"
curl http://localhost:8000/admin/ai/chat/sessions/<session_id>/messages -H "x-api-key: change-me-in-production"
```

### Streaming (SSE)

`POST /admin/ai/chat/stream` streams the answer token-by-token via
Server-Sent Events instead of waiting for the full response - built
for a reactive client (e.g. Spring's `WebClient`) that wants to render
tokens as they arrive:

```bash
curl -N -X POST http://localhost:8000/admin/ai/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "what are total sales by region?"}'
```

Event stream shape (same `ChatRequest` body as `/chat`, plus an optional
`include_reasoning` flag; `session_id` is optional and generated if
omitted, same as the non-streaming endpoint):

```
event: session
data: {"session_id": "..."}

event: reasoning              (only sent if include_reasoning: true in the request)
data: {"content": "Let me check..."}

event: tool_call              (sent once per tool call, as soon as the agent decides to call it)
data: {"tool": "check_inventory_gap"}

event: tool_result            (sent once that tool finishes, with its raw output)
data: {"tool": "check_inventory_gap", "content": "..."}

event: token
data: {"content": "Hel"}

event: token
data: {"content": "lo"}

...

event: done
data: {"response": "Hello ..."}
```

- `reasoning` events only appear if the request body sets
  `"include_reasoning": true` **and** the configured model actually
  supports reasoning/thinking (e.g. `qwen3.5:0.8b`) - see `reasoning=True`
  on `ChatOllama` in `app/llm/client.py`, which separates a thinking
  model's reasoning into its own field instead of mixing it into the
  final answer. Omit the flag (or leave it `false`, the default) to get
  the old behavior exactly - no reasoning events, nothing else changes.
- `tool_call`/`tool_result` events let the client show something like
  *"Checking inventory..."* while a tool runs, instead of the UI going
  quiet until the whole answer is ready.
- If the model call fails partway through, an `error` event (`data:
  {"detail": "..."}`) is sent instead of `done` and the stream ends -
  the HTTP status is already `200` by the time any tokens are sent, so
  errors can't be surfaced as an HTTP error status and are signaled
  in-band instead. The session's message count/title is still recorded
  via `chat_session_service.touch_session` even if the stream errors
  partway through.

### SOP knowledge base (RAG)

Ingest office SOPs (`.txt`/`.md`) so the agent can answer "how do I..."
questions by searching them, instead of guessing:

```bash
# Pull an embedding model once, alongside your chat model
ollama pull nomic-embed-text

curl -X POST http://localhost:8000/admin/sops/upload \
  -H "x-api-key: change-me-in-production" \
  -F "file=@badge_request.md"
```

This chunks the doc, embeds each chunk (`EMBEDDING_MODEL`), and stores
it in a `sqlite-vec` index (`SOP_VECTOR_DB_PATH`) - a one-time step per
doc, not something that runs on every chat request. Re-uploading the
same filename replaces its previous chunks. The agent's `search_sops`
tool then embeds just the user's question at chat time and does a
cosine-similarity search against that index.

## Project structure

```
app/
├── main.py              FastAPI entrypoint - mounts /health and /admin
├── config.py             env var loading
├── llm/
│   ├── client.py           Ollama chat connection (swap models/providers here only)
│   └── embeddings.py       Ollama embedding connection
├── agent/
│   ├── executor.py        agent definition (create_agent + system prompt + tools)
│   └── memory.py          conversation memory via LangGraph checkpointer;
│                            invoke_with_history (blocking full answer),
│                            stream_agent_events (tool_call/tool_result/
│                            reasoning/answer events, for SSE), and
│                            stream_with_history (answer text only,
│                            thin wrapper over stream_agent_events)
├── tools/
│   ├── pandas_tool.py      dataframe query tool
│   ├── sql_tool.py         SQL query tool (in-memory SQLite)
│   ├── inventory_tool.py   inventory gap check by date + SLOC (calls external REST API)
│   └── knowledge_tool.py   search_sops - runtime query side of the SOP knowledge base
├── knowledge/
│   ├── chunking.py         paragraph-aware text chunker
│   ├── vector_store.py     sqlite-vec backed chunk/embedding store
│   └── ingest.py           offline ingestion: chunk + embed + store an SOP doc
├── db/
│   └── session.py          SQLAlchemy engine/session + init_db()
├── models/
│   └── chat_session.py     ChatSession entity (SQLAlchemy)
├── services/
│   └── chat_session_service.py   session create/touch/list/get
├── schemas/
│   ├── chat.py             ChatRequest / ChatResponse
│   ├── chat_session.py     ChatSessionRead (entity -> API mapping)
│   ├── upload.py           UploadResponse
│   └── sop.py              SopIngestResponse
└── api/
    ├── health.py           unversioned GET /health
    ├── deps.py             API key auth dependency
    ├── exceptions.py       centralized error handler (no leaked tracebacks)
    └── v1/
        ├── router.py       aggregates endpoint routers under /admin
        └── endpoints/
            ├── upload.py    POST /admin/upload
            ├── sops.py      POST /admin/sops/upload
            └── chat.py      POST/GET /admin/chat, /admin/chat/sessions[/{id}]
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
- The SOP knowledge base (`app/knowledge/`) only supports `.txt`/`.md`
  ingestion for now. Add a PDF/DOCX extractor to `ingest.py` if SOPs live
  in those formats.
- No LangSmith tracing wired in - useful once you're debugging why the
  agent picked a specific tool call; set `LANGCHAIN_TRACING_V2=true` and
  `LANGCHAIN_API_KEY` env vars to enable it with zero code changes.
# toolboxAI
