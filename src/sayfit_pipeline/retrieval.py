from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from .models import FoodCandidate

TOKEN_RE = re.compile(r"[a-z0-9]+")
CONTEXT_TERMS = {
    "drink",
    "bread",
    "cookie",
    "cake",
    "mix",
    "bar",
    "cereal",
    "cracker",
    "tea",
    "juice",
    "soda",
    "flavor",
    "flavored",
}


class HashingRetriever:
    def __init__(self, food_index: pd.DataFrame, dim: int = 2048, top_k: int = 8) -> None:
        self.food_index = food_index.reset_index(drop=True)
        self.dim = dim
        self.top_k = top_k
        self._names_lower = self.food_index["item_name"].fillna("").astype(str).str.lower()
        self._brands_lower = self.food_index["brand"].fillna("").astype(str).str.lower()

    def _tokenize(self, text: str) -> list[str]:
        tokens: list[str] = []
        for token in TOKEN_RE.findall((text or "").lower()):
            if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
                token = token[:-1]
            tokens.append(token)
        return tokens

    def _embed(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in self._tokenize(text):
            idx = hash(token) % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def _lexical_prefilter(self, query: str, limit: int = 600) -> pd.DataFrame:
        tokens = [t for t in self._tokenize(query) if len(t) >= 2]
        if not tokens:
            return self.food_index.head(limit)

        mask = np.zeros(len(self.food_index), dtype=bool)
        overlap = np.zeros(len(self.food_index), dtype=np.int16)

        for token in tokens:
            patt = rf"(?<![a-z0-9]){re.escape(token)}s?(?![a-z0-9])"
            in_name = self._names_lower.str.contains(patt, regex=True).to_numpy()
            in_brand = self._brands_lower.str.contains(patt, regex=True).to_numpy()
            present = in_name | in_brand
            mask |= present
            overlap += present.astype(np.int16)

        idx = np.where(mask)[0]
        if len(idx) == 0:
            return self.food_index.head(limit)

        # Prioritize rows with higher lexical overlap before vector scoring.
        top_idx = idx[np.argsort(overlap[idx])[::-1][:limit]]
        return self.food_index.iloc[top_idx].copy()

    def retrieve(self, query: str, top_k: int | None = None) -> list[FoodCandidate]:
        k = top_k or self.top_k
        q_tokens = {t for t in self._tokenize(query) if len(t) >= 2}
        q_l = (query or "").lower().strip()
        qvec = self._embed(query)
        if not np.any(qvec):
            return []

        subset = self._lexical_prefilter(query, limit=600)
        texts = subset["text_for_embedding"].tolist()
        if not texts:
            return []
        matrix = np.vstack([self._embed(t) for t in texts])
        sims = matrix @ qvec

        lexical_adj = np.zeros(len(subset), dtype=np.float32)
        names = subset["item_name"].fillna("").astype(str).str.lower()
        for i, name in enumerate(names):
            t = set(self._tokenize(name))
            if q_tokens:
                lexical_adj[i] += 0.20 * (len(q_tokens & t) / len(q_tokens))
            if q_l and (name == q_l or name.startswith(f"{q_l},")):
                lexical_adj[i] += 0.35
            if q_tokens and q_tokens == t:
                lexical_adj[i] += 0.30
            elif q_tokens and q_tokens.issubset(t):
                lexical_adj[i] += 0.12
            if len(q_tokens) <= 2 and any(term in t for term in CONTEXT_TERMS):
                lexical_adj[i] -= 0.20
            if len(q_tokens) == 1 and len(t) >= 3 and not q_tokens.issubset(t):
                lexical_adj[i] -= 0.25

        combined = sims + lexical_adj
        idxs = np.argpartition(combined, -min(k, len(combined)))[-min(k, len(combined)) :]
        idxs = idxs[np.argsort(combined[idxs])[::-1]]

        out: list[FoodCandidate] = []
        for idx in idxs:
            row = subset.iloc[int(idx)]
            score = float(sims[int(idx)])
            if math.isnan(score):
                continue
            out.append(
                FoodCandidate(
                    source=str(row["source"]),
                    item_id=str(row["item_id"]),
                    item_name=str(row["item_name"]),
                    brand=(str(row["brand"]) if str(row["brand"]).strip() else None),
                    kcal_100g=float(row["kcal_100g"]),
                    protein_100g=float(row["protein_100g"]),
                    carbs_100g=float(row["carbs_100g"]),
                    fat_100g=float(row["fat_100g"]),
                    score=score,
                    portion_description=(
                        str(row["portion_description"]) if str(row["portion_description"]).strip() else None
                    ),
                    gram_weight=(
                        float(row["gram_weight"]) if pd.notna(row["gram_weight"]) else None
                    ),
                )
            )
        return out
