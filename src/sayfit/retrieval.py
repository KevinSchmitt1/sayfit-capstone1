"""
Step 2 – RAG Retrieval.

For each extracted food item, retrieve top-K candidates from the FAISS vector store.
Includes exact-match boosting and simple stemming for better accuracy.
"""

from __future__ import annotations

import re

from sayfit.models import FoodCandidate, FoodItem
from sayfit.vector_store import FoodVectorStore

# Boost applied when the candidate name contains the query as an exact word
EXACT_WORD_BOOST = 0.15
# Smaller boost when the query appears as a substring (but not a whole word)
SUBSTRING_BOOST = 0.05


def _simple_variants(word: str) -> list[str]:
    """
    Generate simple singular/plural variants of a word.
    'grapes' → ['grapes', 'grape']
    'grape'  → ['grape', 'grapes']
    'egg'    → ['egg', 'eggs']
    """
    w = word.lower().strip()
    variants = [w]
    if w.endswith("es") and len(w) > 3:
        variants.append(w[:-2])  # grapes → grap? No — need smarter logic
    if w.endswith("s") and not w.endswith("ss"):
        variants.append(w[:-1])  # grapes → grape, eggs → egg
    if not w.endswith("s"):
        variants.append(w + "s")  # grape → grapes, egg → eggs
    # Deduplicate while preserving order
    seen = set()
    return [v for v in variants if not (v in seen or seen.add(v))]


def _name_boost(query: str, candidate_name: str) -> float:
    """
    Return a score boost based on how well the candidate name matches the query.

    Checks the query and its singular/plural variants:
    - Query or variant appears as whole word(s) in candidate → full boost
    - Query is a substring → small boost
    - No overlap → 0
    """
    q = query.lower().strip()
    c = candidate_name.lower().strip()

    # Exact full match
    if q == c:
        return EXACT_WORD_BOOST

    # Check query + its singular/plural variants as whole words
    for variant in _simple_variants(q):
        pattern = r'\b' + re.escape(variant) + r'\b'
        if re.search(pattern, c):
            return EXACT_WORD_BOOST

    # For multi-word queries, also check if ALL words appear as whole words
    words = q.split()
    if len(words) > 1:
        if all(re.search(r'\b' + re.escape(w) + r'\b', c) for w in words):
            return EXACT_WORD_BOOST

    # Substring match (weaker)
    if q in c:
        return SUBSTRING_BOOST

    return 0.0


def retrieve_candidates(
    item: FoodItem,
    store: FoodVectorStore,
    top_k: int = 8,
) -> list[FoodCandidate]:
    """Search the vector store for the best food matches with exact-match boosting."""
    # Build multiple query variants for better recall
    queries = _build_query_variants(item.name)

    # Fetch candidates from all query variants
    fetch_k = top_k * 2
    seen_ids: set[str] = set()
    all_results: list[dict] = []

    for q in queries:
        results = store.search(q, top_k=fetch_k)
        for r in results:
            rid = r["doc_id"]
            if rid not in seen_ids:
                seen_ids.add(rid)
                all_results.append(r)

    # Apply exact-match boosting against the ORIGINAL query
    for r in all_results:
        r["boosted_score"] = r["score"] + _name_boost(item.name, r["name"])

    # Sort by boosted score, then take top_k
    all_results.sort(key=lambda r: r["boosted_score"], reverse=True)
    all_results = all_results[:top_k]

    return [
        FoodCandidate(
            doc_id=r["doc_id"],
            source=r["source"],
            name=r["name"],
            brand=r.get("brand"),
            kcal_100g=r["kcal_100g"],
            protein_100g=r["protein_100g"],
            carbs_100g=r["carbs_100g"],
            fat_100g=r["fat_100g"],
            score=r["boosted_score"],
        )
        for r in all_results
    ]


def _build_query_variants(name: str) -> list[str]:
    """
    Generate search query variants for better recall.
    'grape'    → ['grape', 'grapes']
    'red wine' → ['red wine', 'wine red', 'wine table red']
    'egg'      → ['egg', 'eggs']
    """
    q = name.lower().strip()
    variants = [q]

    # Add singular/plural variants
    for v in _simple_variants(q):
        if v not in variants:
            variants.append(v)

    # For multi-word: also add reversed order
    words = q.split()
    if len(words) >= 2:
        reversed_q = " ".join(reversed(words))
        if reversed_q not in variants:
            variants.append(reversed_q)

    # Add USDA-style search hints for common categories
    # These match the actual USDA naming conventions in the database
    _USDA_HINTS = {
        "wine": "Alcoholic beverage, wine, table, all",
        "red wine": "Alcoholic Beverage, wine, table, red",
        "white wine": "Alcoholic beverage, wine, table, white",
        "beer": "Alcoholic beverage, beer, regular",
        "vodka": "Alcoholic beverage, distilled, vodka",
        "whiskey": "Alcoholic beverage, distilled, whiskey",
        "rum": "Alcoholic beverage, distilled, rum",
        "grapes": "grapes raw seedless",
        "grape": "grapes raw seedless",
    }
    if q in _USDA_HINTS:
        variants.append(_USDA_HINTS[q])

    return variants
