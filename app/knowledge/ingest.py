"""
Offline ingestion: chunk an SOP document, embed each chunk, and store
it in the vector index. Run this once per doc when it's added or
updated - not on every app start or every chat request (see
app/tools/knowledge_tool.py for the cheap runtime query path).
"""
import os

from langchain_ollama import OllamaEmbeddings

from app.config import SOP_CHUNK_OVERLAP, SOP_CHUNK_SIZE
from app.knowledge import vector_store as vs
from app.knowledge.chunking import chunk_text
from app.llm.embeddings import get_embeddings

ALLOWED_EXTENSIONS = (".txt", ".md")


def ingest_sop_file(filepath: str, embeddings: OllamaEmbeddings | None = None) -> str:
    """Reads filepath, chunks it, embeds every chunk, and (re)stores it
    in the vector index keyed by filepath - re-ingesting the same path
    replaces its previous chunks rather than duplicating them.
    """
    if not filepath.lower().endswith(ALLOWED_EXTENSIONS):
        raise ValueError(f"Unsupported SOP file type: {filepath}. Use {ALLOWED_EXTENSIONS}")

    with open(filepath, encoding="utf-8") as f:
        text = f.read()

    doc_title = os.path.splitext(os.path.basename(filepath))[0]
    chunks = chunk_text(text, chunk_size=SOP_CHUNK_SIZE, overlap=SOP_CHUNK_OVERLAP)
    if not chunks:
        return f"No content found in {filepath}, nothing ingested."

    embeddings = embeddings or get_embeddings()
    vectors = embeddings.embed_documents(chunks)

    conn = vs.init_store()
    vs.delete_document(conn, filepath)
    inserted = vs.insert_chunks(conn, doc_title, filepath, chunks, vectors)
    return f"Ingested '{doc_title}' from {filepath}: {inserted} chunks"
