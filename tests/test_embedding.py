import pytest
from pathlib import Path
from app.knowledge import ingest, vector_store
from app.knowledge.chunking import chunk_text
from app.llm.embeddings import get_embeddings
from app.tools import knowledge_tool

def test_ingest():
    tmp_path = Path("tests/ingest")
    embeddings = get_embeddings()
    # print(embeddings)
    # ingest.ingest_sop_file(str(tmp_path), embeddings)
    # result = knowledge_tool.search_sops.invoke({"query": "how to run scr"})
    # print(result)
    #
    for item in tmp_path.iterdir():
        ingest.ingest_sop_file(str(item), embeddings)
        result = knowledge_tool.search_sops.invoke({"query": "how to do inventory adjustment"})
        print(result)


