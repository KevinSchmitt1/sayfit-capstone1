"""
Step 3 – Selection / Grounding.

Choose the best match from retrieved candidates.
Supports deterministic (highest similarity, prefer USDA) and LLM rerank modes.
"""

from __future__ import annotations

import json
from openai import OpenAI

from sayfit.models import FoodCandidate


# ---------- Deterministic selection ----------

def select_best_deterministic(candidates: list[FoodCandidate]) -> FoodCandidate:
    """
    Pick the best candidate by highest cosine similarity.
    Tie-break: prefer USDA (generic) over OpenFoodFacts (branded).
    """
    if not candidates:
        raise ValueError("No candidates to select from")

    def sort_key(c: FoodCandidate) -> tuple:
        source_priority = 0 if c.source == "usda" else 1
        return (-c.score, source_priority)

    return sorted(candidates, key=sort_key)[0]


# ---------- LLM rerank selection ----------

RERANK_SYSTEM_PROMPT = """\
You are a food-matching assistant. The user searched for a food item.
Below is a list of candidate matches from a nutrition database.

Pick the SINGLE best match. Rules:
- Prefer the candidate that IS the food, not a product MADE FROM it.
  Example: "grapes" → pick "GRAPES" or "SEEDLESS GRAPES", NOT "GRAPE TOMATOES" or "GRAPE JELLY".
  Example: "red wine" → pick actual wine, NOT "RED WINE VINEGAR".
- Prefer generic/plain versions over flavored/processed unless the user was specific.
- Prefer USDA (generic) over branded entries when quality is similar.
- You MUST choose from the list – do NOT invent new entries.

Return JSON: {"index": <0-based index of chosen candidate>}
"""


def select_best_llm(
    query_name: str,
    candidates: list[FoodCandidate],
    api_key: str,
    model: str = "gpt-5.2",
) -> FoodCandidate:
    """Use an LLM to pick the best candidate from the list."""
    if not candidates:
        raise ValueError("No candidates to select from")

    candidate_text = "\n".join(
        f"[{i}] {c.name} (source={c.source}, brand={c.brand or 'n/a'}, "
        f"kcal={c.kcal_100g}, protein={c.protein_100g}g, "
        f"carbs={c.carbs_100g}g, fat={c.fat_100g}g, score={c.score:.3f})"
        for i, c in enumerate(candidates)
    )

    user_msg = f'User searched for: "{query_name}"\n\nCandidates:\n{candidate_text}'

    prompt = f"{RERANK_SYSTEM_PROMPT}\n\n{user_msg}\n\nReturn ONLY valid JSON."

    client = OpenAI(api_key=api_key)
    resp = client.responses.create(
        model=model,
        input=prompt,
    )

    raw = resp.output_text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    data = json.loads(raw)
    idx = int(data.get("index", 0))
    idx = max(0, min(idx, len(candidates) - 1))
    return candidates[idx]


# ---------- Unified interface ----------

def select_best(
    query_name: str,
    candidates: list[FoodCandidate],
    *,
    use_llm: bool = False,
    api_key: str = "",
    model: str = "gpt-5.2",
) -> FoodCandidate:
    """Select the best candidate – deterministic by default, LLM if requested."""
    if use_llm and api_key:
        return select_best_llm(query_name, candidates, api_key, model)
    return select_best_deterministic(candidates)
