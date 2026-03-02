from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class ChatProvider(Protocol):
    def answer(self, *, query: str, context: str) -> str:
        ...
