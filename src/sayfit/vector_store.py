"""
FAISS-backed vector store for food retrieval.

Embeds food names using sentence-transformers and stores them in a FAISS index.
Metadata (macros, source, etc.) is kept in a side-car parquet file.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


class FoodVectorStore:
    """Wraps a FAISS flat-IP index + metadata DataFrame."""

    INDEX_FILE = "food.faiss"
    META_FILE = "food_meta.csv"
    MODEL_NAME_FILE = "model_name.txt"

    def __init__(
        self,
        index: faiss.Index | None = None,
        meta: pd.DataFrame | None = None,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.index = index
        self.meta = meta if meta is not None else pd.DataFrame()
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    @classmethod
    def build_from_dataframe(
        cls,
        df: pd.DataFrame,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 512,
    ) -> "FoodVectorStore":
        """Create a new store from the unified food index DataFrame."""
        model = SentenceTransformer(model_name)
        texts = df["text_for_embedding"].tolist()

        print(f"[vector_store] Encoding {len(texts):,} food entries …")
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner-product = cosine on unit vectors
        index.add(embeddings)

        store = cls(index=index, meta=df.reset_index(drop=True), model_name=model_name)
        store._model = model
        print(f"[vector_store] FAISS index built: {index.ntotal:,} vectors, dim={dim}")
        return store

    # ------------------------------------------------------------------
    # Persist / load
    # ------------------------------------------------------------------
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / self.INDEX_FILE))
        self.meta.to_csv(directory / self.META_FILE, index=False)
        (directory / self.MODEL_NAME_FILE).write_text(self.model_name)
        print(f"[vector_store] Saved to {directory}")

    @classmethod
    def load(cls, directory: Path) -> "FoodVectorStore":
        index = faiss.read_index(str(directory / cls.INDEX_FILE))
        meta = pd.read_csv(directory / cls.META_FILE, dtype={"doc_id": str})
        model_name = (directory / cls.MODEL_NAME_FILE).read_text().strip()
        store = cls(index=index, meta=meta, model_name=model_name)
        print(f"[vector_store] Loaded {index.ntotal:,} vectors from {directory}")
        return store

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def search(self, query: str, top_k: int = 8) -> list[dict]:
        """Return the top-K candidate food rows for a text query."""
        model = self._get_model()
        q_vec = model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(q_vec, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row = self.meta.iloc[idx]
            brand_val = row.get("brand", "")
            if pd.isna(brand_val):
                brand_val = ""
            results.append({
                "doc_id": str(row["doc_id"]),
                "source": row["source"],
                "name": row["name"],
                "brand": str(brand_val),
                "kcal_100g": float(row["kcal_100g"]),
                "protein_100g": float(row["protein_100g"]),
                "carbs_100g": float(row["carbs_100g"]),
                "fat_100g": float(row["fat_100g"]),
                "score": float(score),
            })
        return results
