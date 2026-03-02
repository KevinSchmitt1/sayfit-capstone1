from pathlib import Path

from rag_template.core.types import Document

SUPPORTED_SUFFIXES = {".txt", ".md"}


def load_documents(data_dir: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        docs.append(
            Document(
                id=str(path.relative_to(data_dir)),
                text=text,
                metadata={"source": str(path)},
            )
        )
    return docs
