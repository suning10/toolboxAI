"""
Test the SOP knowledge base (chunking, ingest, vector search, and the
search_sops agent tool) with a fake embeddings client - no Ollama
embedding model required, same rationale as test_tools.py.

Run: pytest tests/test_knowledge.py -v
"""
import pytest

from app.knowledge import ingest, vector_store
from app.knowledge.chunking import chunk_text
from app.llm.embeddings import get_embeddings
from app.tools import knowledge_tool

DIM = 768


class FakeEmbeddings:
    """Maps known keywords to fixed basis vectors so search results are
    deterministic and don't depend on a real embedding model."""

    VECTORS = {
        "badge": [1.0, 0.0, 0.0, 0.0],
        "fire": [0.0, 1.0, 0.0, 0.0],
    }

    def _vector_for(self, text: str) -> list[float]:
        for keyword, vector in self.VECTORS.items():
            if keyword in text.lower():
                return vector
        return [0.0, 0.0, 1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector_for(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector_for(text)


@pytest.fixture(autouse=True)
def isolated_vector_db(tmp_path, monkeypatch):
    db_path = str(tmp_path / "sop_vectors.sqlite")
    monkeypatch.setattr(vector_store, "SOP_VECTOR_DB_PATH", db_path)
    monkeypatch.setattr(vector_store, "EMBEDDING_DIMENSIONS", DIM)
    monkeypatch.setattr(knowledge_tool, "get_embeddings", lambda: FakeEmbeddings())
    return db_path


def test_chunk_text_keeps_short_text_as_one_chunk():
    chunks = chunk_text("A short SOP paragraph.", chunk_size=800, overlap=150)
    assert chunks == ["A short SOP paragraph."]


def test_chunk_text_splits_long_text():
    long_para = "word " * 500
    chunks = chunk_text(long_para, chunk_size=200, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_ingest_and_search_sops(tmp_path):
    badge_file = tmp_path / "badge_request.md"
    badge_file.write_text("How to request a badge: submit a ticket to Facilities.")
    fire_file = tmp_path / "fire_safety.md"
    fire_file.write_text("In case of fire, evacuate via the nearest stairwell.")

    ingest.ingest_sop_file(str(badge_file), embeddings=FakeEmbeddings())
    ingest.ingest_sop_file(str(fire_file), embeddings=FakeEmbeddings())

    result = knowledge_tool.search_sops.invoke({"query": "badge process"})
    assert "badge_request" in result
    assert "submit a ticket to Facilities" in result


def test_search_sops_no_documents_ingested():
    result = knowledge_tool.search_sops.invoke({"query": "anything"})
    assert "No SOP documents" in result


def test_reingest_replaces_previous_chunks(tmp_path):
    badge_file = tmp_path / "badge_request.md"
    badge_file.write_text("How to request a badge: old process.")
    ingest.ingest_sop_file(str(badge_file), embeddings=FakeEmbeddings())

    badge_file.write_text("How to request a badge: new process v2.")
    ingest.ingest_sop_file(str(badge_file), embeddings=FakeEmbeddings())

    result = knowledge_tool.search_sops.invoke({"query": "badge"})
    assert "new process v2" in result
    assert "old process" not in result

def test_ingest_and_search_sops():
    tmp_path = "tests/ingest/scr.md"
    # badge_file = tmp_path / "badge_request.md"
    # badge_file.write_text("How to request a badge: submit a ticket to Facilities.")
    # fire_file = tmp_path / "fire_safety.md"
    # fire_file.write_text("In case of fire, evacuate via the nearest stairwell.")
    embeddings = get_embeddings()
    print(embeddings)
    ingest.ingest_sop_file(str(tmp_path), embeddings)
    # ingest.ingest_sop_file(str(fire_file), embeddings=FakeEmbeddings())

    result = knowledge_tool.search_sops.invoke({"query": "how to run scr"})
    print(result)
