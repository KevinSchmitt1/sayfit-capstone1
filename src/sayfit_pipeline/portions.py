from __future__ import annotations

from dataclasses import dataclass

from .models import ExtractedItem, FoodCandidate


UNIT_TO_GRAMS = {
    "g": 1.0,
    "gram": 1.0,
    "grams": 1.0,
    "kg": 1000.0,
    "mg": 0.001,
    "ml": 1.0,
    "l": 1000.0,
    "piece": None,
    "pieces": None,
    "pcs": None,
    "egg": None,
    "eggs": None,
    "serving": None,
    "servings": None,
    "slice": None,
    "slices": None,
    "cup": 240.0,
    "cups": 240.0,
    "tbsp": 15.0,
    "tsp": 5.0,
}

DEFAULT_GRAMS_BY_FOOD = {
    "egg": 50.0,
    "banana": 118.0,
    "apple": 182.0,
    "pizza": 150.0,
    "pepperoni pizza": 150.0,
    "oatmeal": 40.0,
    "almond milk": 240.0,
    "rice": 158.0,
    "noodle": 180.0,
    "pasta": 180.0,
    "chicken breast": 120.0,
    "broccoli": 91.0,
}

UNKNOWN_PORTION_HINTS = {
    "handful": 30.0,
    "fist": 90.0,
    "pinch": 1.0,
}


@dataclass
class PortionResult:
    grams: float
    assumptions: list[str]


def _food_default_grams(name: str) -> tuple[float | None, str | None]:
    n = (name or "").lower()
    for key, grams in DEFAULT_GRAMS_BY_FOOD.items():
        if key in n:
            return grams, f"Used default portion for '{key}'"
    for key, grams in UNKNOWN_PORTION_HINTS.items():
        if key in n:
            return grams, f"Mapped informal portion '{key}' to {grams}g"
    return None, None


def normalize_to_grams(item: ExtractedItem, match: FoodCandidate | None) -> PortionResult:
    assumptions: list[str] = []

    if item.amount is not None and item.unit:
        unit = item.unit.lower().strip()
        if unit in UNIT_TO_GRAMS and UNIT_TO_GRAMS[unit] is not None:
            return PortionResult(grams=float(item.amount) * float(UNIT_TO_GRAMS[unit]), assumptions=assumptions)

        if unit in {"piece", "pieces", "pcs", "egg", "eggs", "slice", "slices", "serving", "servings"}:
            if match and match.gram_weight:
                assumptions.append("Used USDA/OFF portion gram_weight")
                return PortionResult(grams=float(item.amount) * float(match.gram_weight), assumptions=assumptions)

            grams, note = _food_default_grams(item.name)
            if grams is not None:
                assumptions.append(note or "Used default portion")
                return PortionResult(grams=float(item.amount) * grams, assumptions=assumptions)

            assumptions.append("Unknown portion size; fallback 100g each")
            return PortionResult(grams=float(item.amount) * 100.0, assumptions=assumptions)

    if match and match.gram_weight:
        assumptions.append("Amount/unit missing; used candidate gram_weight")
        return PortionResult(grams=float(match.gram_weight), assumptions=assumptions)

    grams, note = _food_default_grams((match.item_name if match else item.name) or item.name)
    if grams is not None:
        assumptions.append(note or "Used default portion")
        return PortionResult(grams=grams, assumptions=assumptions)

    assumptions.append("No amount/unit; fallback 100g")
    return PortionResult(grams=100.0, assumptions=assumptions)
