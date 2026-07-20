"""
Runtime side of the SOP knowledge base: embeds the user's query (cheap
- one short string) and does a similarity search against the
pre-built vector index (see app/knowledge/ingest.py for how that index
gets built).
"""
from langchain.tools import tool

from app.config import SOP_SEARCH_TOP_K
from app.knowledge import vector_store as vs
from app.llm.embeddings import get_embeddings


@tool
def search_sops(query: str) -> str:
    """Search the office SOP (standard operating procedure) knowledge
    base for text relevant to the user's question, e.g. "how do I
    request a new badge" or "what's the process for reporting a safety
    incident". Returns the most relevant SOP excerpts with their
    source document titles - read them and answer in your own words,
    citing the document title.
    """
    conn = vs.init_store()
    query_embedding = get_embeddings().embed_query(query)
    results = vs.search(conn, query_embedding, top_k=SOP_SEARCH_TOP_K)

    if not results:
        return "No SOP documents have been ingested yet."

    formatted = [
        f"[{r['doc_title']}]\n{r['content']}" for r in results
    ]
    return "\n\n---\n\n".join(formatted)
