from __future__ import annotations

import re

from .models import FoodCandidate

BRANDED_HINTS = {
    "bar",
    "cereal",
    "kellogg",
    "cheerios",
    "oreo",
    "coke",
    "pepsi",
    "mcdonald",
    "subway",
    "starbucks",
    "protein shake",
}

JUNK_TERMS = {
    "candy",
    "gumball",
    "cookie",
    "cookies",
    "frosting",
    "syrup",
    "soda",
    "dessert",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")
FRUIT_TERMS = {"apple", "banana", "orange", "strawberry", "blueberry", "grape", "pear"}
VEG_TERMS = {"broccoli", "spinach", "carrot", "cucumber", "tomato", "lettuce", "pepper"}
BEVERAGE_TERMS = {"tea", "juice", "drink", "soda", "water", "beverage", "quencher", "thirst", "ade"}
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
    "alfredo",
}


def _looks_branded(query: str) -> bool:
    q = query.lower()
    return any(hint in q for hint in BRANDED_HINTS)

def _tokens(text: str) -> set[str]:
    out: set[str] = set()
    for token in TOKEN_RE.findall((text or "").lower()):
        if len(token) < 2:
            continue
        if token.endswith("s") and len(token) > 3 and not token.endswith("ss"):
            token = token[:-1]
        out.add(token)
    return out


def choose_best_candidate(query: str, candidates: list[FoodCandidate]) -> FoodCandidate | None:
    if not candidates:
        return None

    branded_query = _looks_branded(query)
    q_tokens = _tokens(query)
    simple_query = (len(q_tokens) <= 2) and (not branded_query)
    query_l = query.lower().strip()

    best = None
    best_score = float("-inf")
    for c in candidates:
        adjusted = c.score
        c_tokens = _tokens(c.item_name)
        if q_tokens:
            inter = q_tokens & c_tokens
            recall = len(inter) / len(q_tokens)
            precision = len(inter) / max(len(c_tokens), 1)
            adjusted += 0.35 * recall
            adjusted += 0.25 * precision
            adjusted -= 0.15 * (1.0 - precision)
        if c.source == "usda" and not branded_query:
            adjusted += 0.03
        if c.source == "off" and branded_query:
            adjusted += 0.04
        if c.brand:
            adjusted += 0.01
        if simple_query and len(c_tokens) > 5:
            adjusted -= 0.25
        if simple_query and c.brand:
            adjusted -= 0.15
        if simple_query and q_tokens and q_tokens == c_tokens:
            adjusted += 0.45
        elif simple_query and q_tokens and q_tokens.issubset(c_tokens) and len(c_tokens) >= 3:
            adjusted -= 0.25
        name_l = c.item_name.lower().strip()
        if simple_query and (name_l == query_l or name_l.startswith(f"{query_l},")):
            adjusted += 0.45
        if simple_query and any(term in c_tokens for term in CONTEXT_TERMS):
            adjusted -= 0.35
        if any(term in c_tokens for term in JUNK_TERMS) and not any(term in q_tokens for term in JUNK_TERMS):
            adjusted -= 0.60
        if q_tokens & FRUIT_TERMS and c.kcal_100g > 95:
            adjusted -= 0.55
        if q_tokens & FRUIT_TERMS and c.carbs_100g > 20:
            adjusted -= 0.55
        if q_tokens & VEG_TERMS and c.kcal_100g > 90:
            adjusted -= 0.40
        if any(term in c_tokens for term in BEVERAGE_TERMS) and not any(
            term in q_tokens for term in BEVERAGE_TERMS | {"milk"}
        ):
            adjusted -= 0.35
        if adjusted > best_score:
            best = c
            best_score = adjusted

    return best


def confidence_band(score: float) -> str:
    if score >= 0.52:
        return "high"
    if score >= 0.38:
        return "medium"
    return "low"
