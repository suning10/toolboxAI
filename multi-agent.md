# Split the single ReAct agent into a supervisor + 3-node multi-agent graph

**Status: implemented.** See `app/agent/executor.py` (graph construction) and `app/agent/memory.py` (streaming/history). Verification steps 1-3 below were run and passed against the real code; step 4 needs a live Ollama server to exercise manually.

## Context

The agent used to run as one flat `create_agent()` ReAct loop (`app/agent/executor.py`) with all 4 tools bound at once: `search_sops`, `check_inventory_gap`, `check_inventory_gap_for_sku`, `check_top_inventory_gap`. This splits it into three specialized nodes - a **retrieval agent** (SOP search), an **api call agent** (the 3 inventory-gap tools), and a **response agent** (formatting/summarization, no tools) - wired together by a simple supervisor that routes between them. This narrows each node's decision surface (relevant given the small local model's already-observed tool-selection flakiness) and cleanly separates "gather information" from "answer the user."

## Empirically verified LangGraph behavior (this is what makes the plan concrete, not guesswork)

Prototyped directly against this project's installed `langgraph==1.2.9`/`langchain==1.3.14` using fake chat models (no real Ollama needed), mirroring the same verification approach used for the original SSE streaming feature:

- **A compiled `create_agent()` graph can be added as a node in a parent `StateGraph`** via `graph.add_node("retrieval_agent", retrieval_agent_compiled_graph)` - confirmed working with the real `create_agent()` import, only the underlying chat model was faked.
- **By default (no `subgraphs=True`), a nested `create_agent()` node's internal activity streams as complete, non-chunked messages tagged with the *parent's* node name** - not token-by-token, and not tagged with the child's own internal `"model"`/`"tools"` node names. Concretely, for `retrieval_agent`/`api_agent`: one complete `AIMessage` with a fully-formed `.tool_calls` list (not `.tool_call_chunks`), then one complete `ToolMessage`, then one complete wrap-up `AIMessage` - all three tagged `langgraph_node == "retrieval_agent"` (or `"api_agent"`).
- **A plain top-level node (a normal Python function calling `model.invoke(...)`) still streams full token-by-token `AIMessageChunk`s** tagged with that node's own name - confirmed even though the node function calls `.invoke()`, not `.stream()`; LangGraph's `stream_mode="messages"` instruments the model call regardless.
- **The supervisor's own routing-decision LLM call is cleanly isolated** by `langgraph_node == "supervisor"` and is trivial to exclude from user-facing output.
- **Keeping the supervisor's raw routing output out of the shared `messages` state is a deliberate node-return choice** (`return {"next": decision}`, no `"messages"` key) - verified this keeps retrieval/api/response agents from ever seeing routing noise in their own context.

This means the existing SSE contract (`session` / `reasoning` / `tool_call` / `tool_result` / `token` / `done`) is fully preservable - it just needs re-pointing at the new node names, and the tool-call detection logic actually gets *simpler* (complete `.tool_calls` list per message, no more partial-chunk accumulation/dedup needed for the specialist nodes).

## Design

### Graph shape

```
START -> supervisor -> (retrieval_agent | api_agent | response_agent)   [conditional edge on state["next"]]
retrieval_agent -> supervisor
api_agent -> supervisor
response_agent -> END
```

- **State**: `class AgentGraphState(MessagesState): next: str` - plain `messages` (shared conversation, `add_messages` reducer) plus a routing field.
- **supervisor node**: a bare LLM call (no tools) using `.with_structured_output(Route)` where `Route` is a small Pydantic model with `next: Literal["retrieval_agent", "api_agent", "response_agent"]`. Returns `{"next": decision}` only - never touches `"messages"`, so its routing text never pollutes the conversation the other nodes see.
- **retrieval_agent node**: `create_agent(model=..., tools=[search_sops], system_prompt=RETRIEVAL_SYSTEM_PROMPT)` - added directly as a graph node (no checkpointer of its own; only the outer graph is checkpointed).
- **api_agent node**: `create_agent(model=..., tools=[check_inventory_gap, check_inventory_gap_for_sku, check_top_inventory_gap], system_prompt=API_SYSTEM_PROMPT)` - same pattern.
- **response_agent node**: a plain function - `model.invoke([SystemMessage(RESPONSE_SYSTEM_PROMPT)] + state["messages"])` - no tools bound at all. This is the only node whose output streams as the final `token` events and whose message becomes the true answer.
- Both `retrieval_agent` and `api_agent` edge back to `supervisor` (so the supervisor can chain retrieval -> api -> response if a question needs both, or go straight to `response_agent` for anything that doesn't need tools at all). Relying on LangGraph's default `recursion_limit` (25) as the loop safety net rather than adding custom loop-counting - keeps this "simple" per the request; can tighten later if the small model loops.

### System prompts (replacing the single flat `SYSTEM_PROMPT`)

Four focused prompts in `app/agent/executor.py`, replacing the old one-size-fits-all prompt:
- `SUPERVISOR_SYSTEM_PROMPT` - coarse routing only: SOP/procedure/"how do I" questions -> `retrieval_agent`; inventory/SCR/gap/SKU questions -> `api_agent`; anything else, or once enough info has been gathered -> `response_agent`.
- `RETRIEVAL_SYSTEM_PROMPT` - carries the old `search_sops` guidance (cite doc titles, say so if nothing relevant comes back).
- `API_SYSTEM_PROMPT` - carries the old "keep tool calls minimal, retry once on error" framing. The detailed inter-tool disambiguation (SKU vs SLOC vs top-gap) already lives in each tool's own docstring per `app/tools/inventory_tool.py` and stays there unchanged - `create_agent()`'s ReAct loop reads those directly.
- `RESPONSE_SYSTEM_PROMPT` - new. Explicitly told it has no tools and must synthesize a clear, formatted final answer from whatever the conversation already contains (user question + any retrieval/api results) - not call anything, not repeat raw tool output verbatim.

### Files changed

**`app/agent/executor.py`** - replaced the single-agent `build_agent()` with:
- The 4 system prompts above.
- `TOOLS_RETRIEVAL = [search_sops]`, `TOOLS_API = [check_inventory_gap, check_inventory_gap_for_sku, check_top_inventory_gap]` (splits the old flat `TOOLS` list).
- `build_graph(checkpointer=None)` - constructs the `StateGraph` described above and returns `graph.compile(checkpointer=checkpointer)`. A new shared builder so the construction logic isn't duplicated between `executor.py` and `memory.py` the way the old single `create_agent(...)` call already was (both files used to build their own copy) - the multi-agent version is enough more code that duplicating it isn't worth it.
- `build_agent()` is now a thin wrapper: `return build_graph()` (no checkpointer) - keeps the old external function name/contract for anything still calling it.

**`app/agent/memory.py`**:
- `build_agent_with_memory()` - calls `app.agent.executor.build_graph(checkpointer=SqliteSaver(conn))` instead of its own duplicated `create_agent(...)` call. Everything else here (the sqlite connection setup) stays the same.
- `stream_agent_events()` - event-classification filter rewritten using the verified behavior above:
  - `tool_call`: `isinstance(chunk, AIMessage) and chunk.tool_calls and metadata.get("langgraph_node") in {"retrieval_agent", "api_agent"}` -> iterate `chunk.tool_calls`, yield one `{"type": "tool_call", "tool": tc["name"]}` per entry (simpler than before - no more partial `tool_call_chunks` accumulation/dedup, since these arrive as one complete message).
  - `tool_result`: same as before, `isinstance(chunk, ToolMessage)`, just dropped the `langgraph_node == "tools"` equality check (that was `create_agent()`'s fixed internal name; here it's the parent's node name that shows up, and any `ToolMessage` is unambiguous on its own).
  - `answer`/`token`: only `AIMessageChunk`s tagged `metadata.get("langgraph_node") == "response_agent"` - the one behavioral tightening: retrieval/api agents' own wrap-up commentary (e.g. "Found relevant SOP info.") is no longer treated as user-facing answer text, matching the new architecture's intent that only the response agent produces the final answer.
  - `reasoning`: unchanged mechanism (`chunk.additional_kwargs.get("reasoning_content")`), just also gated to `response_agent` alongside the `answer` check.
- `get_message_history()` - collapses consecutive `"assistant"`-role entries within a turn down to just the last one, so a specialist's wrap-up message never shows up as a separate turn in `GET /chat/sessions/{id}/messages` - only the response agent's true final answer does (it's always the last assistant message before the next human message or end of list, since `response_agent` is the guaranteed terminal node). Implemented as a simple "replace last entry if it's also assistant" step while building the history list - no new tagging/state needed, and it's a no-op for the old already-single-assistant-message-per-turn shape (old tests kept passing unchanged).

**No changes needed**: `app/api/v1/endpoints/chat.py` (same `_agent.invoke()`/`.stream()`/`.get_state()` contract), `app/llm/client.py`, `app/tools/*`, `app/db/*`, `app/knowledge/*`, `app/schemas/*`.

### Testing

- **`tests/test_chat_stream.py`** - fully rewritten fixtures for the new multi-node graph: a fake supervisor model (structured-output stand-in returning routing decisions in sequence), fake retrieval/api specialist models (two-call tool-then-wrapup fakes), a fake response model with a real `_stream` implementation. Covers: routing straight to `response_agent` (no tools needed), routing through `retrieval_agent`/`api_agent` then `response_agent`, tool-call/tool-result event shape from a specialist, reasoning gated to `response_agent` only, and specialist wrap-up text never becoming a `token`/`answer` event. 7 tests, all passing.
- **`tests/test_chat_history.py`** - old `touch_session`/title tests unaffected. Added one new case building a two-node graph that produces two consecutive assistant messages in one turn and asserting `get_message_history` keeps only the last. 6 tests, all passing.
- Unaffected: `tests/test_tools.py`.

## Verification

1. `PYTHONPATH=. uv run pytest tests/ -v` - rewritten/new tests pass (16/16 in the affected files); the pre-existing unrelated failures elsewhere (embedding dimension mismatches, a hardcoded test path, a literal `assertEqual(True, False)` stub, one real-network test) are untouched by this change. **Done.**
2. Swapped a fake multi-agent graph into `chat_module._agent` and hit `/admin/ai/chat/stream` via `TestClient` - confirmed the SSE sequence is `session` -> `tool_call` -> `tool_result` -> `token`* -> `done`, matching the existing wire contract exactly. **Done.**
3. Hit `/admin/ai/chat/sessions/{id}/messages` after a tool-using turn - confirmed only the response agent's final answer appears ("The gap is 5 units."), not the api_agent's internal wrap-up commentary. **Done.**
4. Manual sanity check against real Ollama: ask a pure SOP question (should route supervisor -> retrieval_agent -> supervisor -> response_agent), an inventory question (-> api_agent), and a simple conversational message (should skip straight to response_agent with no tool calls at all). **Not yet run - needs a live Ollama server.**
