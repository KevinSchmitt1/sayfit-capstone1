from rag_template.core.types import RetrievedChunk


def format_context(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        parts.append(f"[{i}] Source: {source}\n{chunk.text}")
    return "\n\n".join(parts)
