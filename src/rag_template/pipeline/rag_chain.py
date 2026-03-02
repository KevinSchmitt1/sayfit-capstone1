from pathlib import Path

from rag_template.config.settings import Settings
from rag_template.generation.prompting import format_context
from rag_template.providers.base import ChatProvider, EmbeddingProvider
from rag_template.retrieval.vector_store import LocalVectorStore


def answer_query(
    *,
    query: str,
    index_path: Path,
    settings: Settings,
    embeddings: EmbeddingProvider,
    chat: ChatProvider,
) -> str:
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    store = LocalVectorStore.load(index_path)
    query_vector = embeddings.embed_query(query)
    retrieved = store.search(query_vector, top_k=settings.top_k)

    if not retrieved:
        return "No relevant context found in the index."

    context = format_context(retrieved)
    return chat.answer(query=query, context=context)
