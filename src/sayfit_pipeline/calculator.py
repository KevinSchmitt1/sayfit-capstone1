from __future__ import annotations

from .models import FoodCandidate


def nutrition_for_grams(candidate: FoodCandidate, grams: float) -> dict[str, float]:
    factor = grams / 100.0
    kcal = round(candidate.kcal_100g * factor, 2)
    protein = round(candidate.protein_100g * factor, 2)
    carbs = round(candidate.carbs_100g * factor, 2)
    fat = round(candidate.fat_100g * factor, 2)
    return {
        "calories": kcal,
        "protein": protein,
        "carbs": carbs,
        "fat": fat,
    }
