from openai import OpenAI

from rag_template.providers.base import ChatProvider, EmbeddingProvider


class OpenAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=[text])
        return response.data[0].embedding


class OpenAIChatProvider(ChatProvider):
    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def answer(self, *, query: str, context: str) -> str:
        prompt = (
            "You are a helpful assistant. Use the provided context to answer. "
            "If context is insufficient, say what is missing.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )
        response = self.client.responses.create(model=self.model, input=prompt)
        return response.output_text.strip()
