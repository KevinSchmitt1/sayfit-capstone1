from rag_template.core.types import Document


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    step = chunk_size - chunk_overlap
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size]
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    chunked_docs: list[Document] = []
    for doc in docs:
        chunks = chunk_text(doc.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, chunk in enumerate(chunks):
            chunked_docs.append(
                Document(
                    id=f"{doc.id}::chunk_{i}",
                    text=chunk,
                    metadata={**doc.metadata, "parent_id": doc.id, "chunk_index": str(i)},
                )
            )
    return chunked_docs
