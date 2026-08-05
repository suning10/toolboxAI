from pathlib import Path

from app.config import SOP_CHUNK_SIZE, SOP_CHUNK_OVERLAP
from app.knowledge import vector_store as vs
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import Docx2txtLoader

from app.knowledge.chunking import chunk_text, chunk_recursive
from app.llm.embeddings import get_embeddings


def test_sop_load_recursive():
    loader = DirectoryLoader(
        r"C:\Users\l.qin3\PycharmProjects\toolboxAI\tests\ingest\word",
        glob= "*.docx",
        loader_cls= Docx2txtLoader,
        recursive= True,
        silent_errors=True
    )

    docs = loader.load()
    print(len(docs))

    for doc in docs:
        source_path = doc.metadata['source']
        doc_title = Path(source_path).stem
        chunks = chunk_recursive(doc.page_content, SOP_CHUNK_SIZE,SOP_CHUNK_OVERLAP)
        if not chunks:
            print( f"No content found , nothing ingested.")

        embeddings = get_embeddings()
        vectors = embeddings.embed_documents(chunks)

        conn = vs.init_store()
        vs.delete_document(conn, source_path)
        inserted = vs.insert_chunks(conn, doc_title, source_path, chunks, vectors)
        s= f"Ingested '{doc_title}' from {source_path}: {inserted} chunks"
        print(s)