from pathlib import Path

from rag_template.config.settings import Settings
from rag_template.ingestion.chunker import chunk_documents
from rag_template.ingestion.loader import load_documents
from rag_template.providers.base import EmbeddingProvider
from rag_template.retrieval.vector_store import LocalVectorStore


def build_index(
    *,
    data_dir: Path,
    out_path: Path,
    settings: Settings,
    embeddings: EmbeddingProvider,
) -> int:
    docs = load_documents(data_dir)
    chunked_docs = chunk_documents(docs, settings.chunk_size, settings.chunk_overlap)

    if not chunked_docs:
        raise ValueError(f"No valid documents found in: {data_dir}")

    chunks = [(doc.id, doc.text, doc.metadata) for doc in chunked_docs]
    vectors = embeddings.embed_texts([doc.text for doc in chunked_docs])

    store = LocalVectorStore()
    store.build(chunks, vectors)
    store.save(out_path)
    return len(chunked_docs)
