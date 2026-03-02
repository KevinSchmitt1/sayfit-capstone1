from rag_template.ingestion.chunker import chunk_text


def test_chunk_text_basic() -> None:
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=300, chunk_overlap=50)
    assert len(chunks) >= 3
    assert all(len(c) <= 300 for c in chunks)
