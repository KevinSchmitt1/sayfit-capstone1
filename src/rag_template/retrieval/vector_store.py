import json
from pathlib import Path

import numpy as np

from rag_template.core.types import RetrievedChunk


class LocalVectorStore:
    def __init__(self) -> None:
        self.chunk_ids: list[str] = []
        self.texts: list[str] = []
        self.metadata: list[dict[str, str]] = []
        self.matrix: np.ndarray | None = None

    def build(self, chunks: list[tuple[str, str, dict[str, str]]], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        self.chunk_ids = [item[0] for item in chunks]
        self.texts = [item[1] for item in chunks]
        self.metadata = [item[2] for item in chunks]

        matrix = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.matrix = matrix / np.clip(norms, 1e-10, None)

    def search(self, query_embedding: list[float], top_k: int = 4) -> list[RetrievedChunk]:
        if self.matrix is None or len(self.chunk_ids) == 0:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q = q / np.clip(np.linalg.norm(q), 1e-10, None)
        scores = self.matrix @ q

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(
                chunk_id=self.chunk_ids[idx],
                text=self.texts[idx],
                score=float(scores[idx]),
                metadata=self.metadata[idx],
            )
            for idx in top_indices
        ]

    def save(self, path: Path) -> None:
        if self.matrix is None:
            raise ValueError("Vector store is empty")
        payload = {
            "chunk_ids": self.chunk_ids,
            "texts": self.texts,
            "metadata": self.metadata,
            "embeddings": self.matrix.tolist(),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LocalVectorStore":
        payload = json.loads(path.read_text(encoding="utf-8"))
        store = cls()
        store.chunk_ids = payload["chunk_ids"]
        store.texts = payload["texts"]
        store.metadata = payload["metadata"]
        store.matrix = np.array(payload["embeddings"], dtype=np.float32)
        return store
