For an AI Agent Engineer role, here's the tightened set — leads with agent architecture and tool ecosystems, since that's what the title signals:

- Architected a tool-calling LLM agent (LangChain/LangGraph) with multi-turn conversational memory, a provider-agnostic model layer (Ollama/Anthropic/OpenAI), and a curated tool surface spanning REST APIs, SQL, and a custom RAG pipeline
- Integrated the Model Context Protocol (MCP) to extend agent capabilities with an external MySQL tool server, resolving a live SDK/adapter version incompatibility and bridging MCP's async-only tool interface into the agent's synchronous execution path
- Built a retrieval-augmented generation pipeline from scratch — document chunking, embedding generation, and vector search (sqlite-vec) — exposed to the agent as a self-serve knowledge tool
- Engineered agent reliability and safety guardrails: read-only enforcement on agent-issued SQL to prevent unintended writes, and diagnosed small-model tool-calling failure modes (repeated/looping tool invocations) to inform model and prompt design decisions
- Implemented real-time agent observability for end users via SSE streaming — surfacing tool-call, tool-result, and model-reasoning events as distinct live event types, not just the final answer

That's 5 — if you need to cut to 3-4 for space, I'd keep the first two (agent architecture + MCP, since MCP is the single hottest keyword for this role right now) and the reliability/safety one, since it shows engineering maturity beyond just "wired up an API."