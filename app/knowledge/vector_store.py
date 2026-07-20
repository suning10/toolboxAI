"""
sqlite-vec backed store for SOP chunk embeddings. Raw sqlite3 (not
SQLAlchemy) because vec0 is a virtual table type provided by the
sqlite-vec extension - SQLAlchemy has no concept of it.

Two tables, joined by rowid:
  sop_chunks         doc_title, source_path, chunk_index, content
  sop_chunk_vectors  vec0 virtual table holding the embedding, one
                      row per sop_chunks row (same rowid)
"""
import os
import sqlite3
import struct

import sqlite_vec

from app.config import EMBEDDING_DIMENSIONS, SOP_VECTOR_DB_PATH


def _connect() -> sqlite3.Connection:
    db_dir = os.path.dirname(SOP_VECTOR_DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(SOP_VECTOR_DB_PATH, check_same_thread=False)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def _to_blob(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def init_store(conn: sqlite3.Connection | None = None) -> sqlite3.Connection:
    """Create the chunk + vector tables if they don't exist yet. Safe to
    call on every ingest/search - it's a no-op once the tables exist.
    """
    conn = conn or _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sop_chunks (
            id INTEGER PRIMARY KEY,
            doc_title TEXT NOT NULL,
            source_path TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL
        )
        """
    )
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS sop_chunk_vectors USING vec0(
            embedding float[{EMBEDDING_DIMENSIONS}] distance_metric=cosine
        )
        """
    )
    conn.commit()
    return conn


def delete_document(conn: sqlite3.Connection, source_path: str) -> None:
    """Remove any previously ingested chunks for this file, so
    re-ingesting an updated SOP doesn't leave stale chunks behind."""
    rows = conn.execute(
        "SELECT id FROM sop_chunks WHERE source_path = ?", (source_path,)
    ).fetchall()
    ids = [r[0] for r in rows]
    if not ids:
        return
    conn.executemany("DELETE FROM sop_chunks WHERE id = ?", [(i,) for i in ids])
    conn.executemany("DELETE FROM sop_chunk_vectors WHERE rowid = ?", [(i,) for i in ids])
    conn.commit()


def insert_chunks(
    conn: sqlite3.Connection,
    doc_title: str,
    source_path: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> int:
    inserted = 0
    for i, (content, embedding) in enumerate(zip(chunks, embeddings)):
        cursor = conn.execute(
            "INSERT INTO sop_chunks (doc_title, source_path, chunk_index, content) VALUES (?, ?, ?, ?)",
            (doc_title, source_path, i, content),
        )
        row_id = cursor.lastrowid
        conn.execute(
            "INSERT INTO sop_chunk_vectors (rowid, embedding) VALUES (?, ?)",
            (row_id, _to_blob(embedding)),
        )
        inserted += 1
    conn.commit()
    return inserted


def search(conn: sqlite3.Connection, query_embedding: list[float], top_k: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT sop_chunks.doc_title, sop_chunks.source_path, sop_chunks.content, sop_chunk_vectors.distance
        FROM sop_chunk_vectors
        JOIN sop_chunks ON sop_chunks.id = sop_chunk_vectors.rowid
        WHERE sop_chunk_vectors.embedding MATCH ? AND k = ?
        ORDER BY sop_chunk_vectors.distance
        """,
        (_to_blob(query_embedding), top_k),
    ).fetchall()
    return [
        {"doc_title": r[0], "source_path": r[1], "content": r[2], "distance": r[3]}
        for r in rows
    ]
