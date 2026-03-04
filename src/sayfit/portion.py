"""
Step 4 – Portion Normalisation & Nutrition Math.

Converts units/amounts to grams, then computes macros from per-100g values.
Includes a built-in portion-defaults table for common foods.
"""

from __future__ import annotations

from sayfit.models import FoodCandidate, FoodItem, MatchedItem, NutritionInfo


# ---------------------------------------------------------------------------
# Portion defaults: estimated grams for common items and units.
# Used when the user says "1 egg" or "a pizza" without specifying grams.
# ---------------------------------------------------------------------------

# Maps (food keyword → grams per 1 piece/serving)
FOOD_PORTION_DEFAULTS: dict[str, float] = {
    # Proteins
    "egg": 50.0,
    "boiled egg": 50.0,
    "hard boiled egg": 50.0,
    "chicken breast": 150.0,
    "grilled chicken": 150.0,
    "salmon": 150.0,
    "baked salmon": 150.0,
    "steak": 200.0,
    "tuna": 100.0,
    "shrimp": 85.0,
    "tofu": 125.0,
    # Dairy
    "yogurt": 150.0,
    "greek yogurt": 170.0,
    "cheese slice": 28.0,
    "milk": 250.0,  # 1 glass
    # Grains / Bread
    "bread": 30.0,  # 1 slice
    "toast": 30.0,
    "whole grain bread": 35.0,
    "bagel": 100.0,
    "tortilla": 45.0,
    "rice": 180.0,  # 1 cooked serving
    "pasta": 180.0,
    "oatmeal": 40.0,  # dry
    "noodles": 180.0,
    "pancake": 75.0,
    # Fruits
    "apple": 180.0,
    "banana": 120.0,
    "orange": 150.0,
    "pear": 175.0,
    "avocado": 150.0,
    "blueberries": 75.0,  # 1 handful
    "strawberries": 100.0,
    "grapes": 80.0,
    "mango": 200.0,
    "watermelon": 280.0,
    "peach": 150.0,
    # Vegetables
    "broccoli": 85.0,
    "spinach": 30.0,
    "carrot": 60.0,
    "potato": 150.0,
    "roasted potatoes": 150.0,
    "sweet potato": 150.0,
    "tomato": 120.0,
    "cucumber": 100.0,
    "green beans": 100.0,
    "corn": 90.0,
    "lettuce": 50.0,
    # Meals / complex
    "pizza": 150.0,       # 1 serving / slice
    "pepperoni pizza": 150.0,
    "burger": 200.0,
    "sandwich": 150.0,
    "sushi roll": 200.0,
    "taco": 100.0,
    "burrito": 250.0,
    # Snacks & nuts
    "almond": 28.0,       # 1 handful
    "almonds": 28.0,
    "roasted almonds": 28.0,
    "walnuts": 28.0,
    "peanuts": 28.0,
    "peanut butter": 32.0,  # 2 tbsp
    "granola bar": 40.0,
    "protein bar": 60.0,
    "chips": 28.0,
    "chocolate": 40.0,
    # Drinks / liquids
    "almond milk": 250.0,
    "orange juice": 250.0,
    "protein shake": 350.0,
    "protein powder": 30.0,  # 1 scoop
    "coffee": 240.0,
    "smoothie": 350.0,
    # Condiments
    "honey": 21.0,        # 1 tbsp
    "tomato sauce": 60.0,
    "olive oil": 14.0,    # 1 tbsp
    "butter": 14.0,
    "parmesan": 10.0,
    "ketchup": 17.0,
    "mayonnaise": 15.0,
}

# Unit conversion helpers
UNIT_ALIASES: dict[str, str] = {
    "gram": "g", "grams": "g",
    "kilogram": "kg", "kilograms": "kg",
    "milliliter": "ml", "milliliters": "ml", "millilitre": "ml",
    "liter": "l", "liters": "l", "litre": "l",
    "pieces": "piece", "pcs": "piece", "pc": "piece",
    "slices": "slice",
    "servings": "serving",
    "cups": "cup",
    "tablespoon": "tbsp", "tablespoons": "tbsp",
    "teaspoon": "tsp", "teaspoons": "tsp",
    "handful": "piece",
}

# Approximate grams per unit (when not food-specific)
UNIT_TO_GRAMS: dict[str, float] = {
    "g": 1.0,
    "kg": 1000.0,
    "ml": 1.0,      # rough assumption: density ≈ 1
    "l": 1000.0,
    "cup": 240.0,
    "tbsp": 15.0,
    "tsp": 5.0,
}

# Default portion when nothing else matches
DEFAULT_PORTION_GRAMS = 100.0


def _normalise_unit(unit: str | None) -> str | None:
    """Canonicalize a unit string."""
    if unit is None:
        return None
    u = unit.strip().lower()
    return UNIT_ALIASES.get(u, u)


def _lookup_portion_default(food_name: str) -> float | None:
    """Check the portion defaults table for a food name."""
    name_lower = food_name.lower()
    # Exact match
    if name_lower in FOOD_PORTION_DEFAULTS:
        return FOOD_PORTION_DEFAULTS[name_lower]
    # Partial match (e.g., "pepperoni pizza" contains "pizza")
    for key, grams in FOOD_PORTION_DEFAULTS.items():
        if key in name_lower or name_lower in key:
            return grams
    return None


def estimate_grams(item: FoodItem) -> float:
    """
    Convert an item's amount + unit into grams.

    Priority:
    1. If amount and unit are provided and unit is weight → direct conversion.
    2. If amount and unit == piece/slice/serving → lookup portion defaults.
    3. If amount is provided but no unit → assume grams.
    4. If nothing → lookup portion default or fall back to 100g.
    """
    unit = _normalise_unit(item.unit)
    amount = item.amount

    # Case 1: explicit weight unit
    if amount is not None and unit in UNIT_TO_GRAMS:
        return amount * UNIT_TO_GRAMS[unit]

    # Case 2: count-based units (piece, slice, serving)
    if amount is not None and unit in ("piece", "slice", "serving"):
        per_piece = _lookup_portion_default(item.name) or DEFAULT_PORTION_GRAMS
        return amount * per_piece

    # Case 3: amount without unit → assume grams
    if amount is not None and unit is None:
        return amount

    # Case 4: no amount → portion default or 100g
    per_piece = _lookup_portion_default(item.name) or DEFAULT_PORTION_GRAMS
    if amount is not None:
        return amount * per_piece
    return per_piece


def compute_nutrition(grams: float, candidate: FoodCandidate) -> NutritionInfo:
    """Compute macros for a given portion from per-100g values."""
    factor = grams / 100.0
    return NutritionInfo(
        calories=round(candidate.kcal_100g * factor, 2),
        protein=round(candidate.protein_100g * factor, 2),
        carbs=round(candidate.carbs_100g * factor, 2),
        fat=round(candidate.fat_100g * factor, 2),
    )


def normalise_and_compute(
    item: FoodItem,
    candidate: FoodCandidate,
) -> MatchedItem:
    """Full step 4: portion normalisation + nutrition math for one item."""
    grams = estimate_grams(item)
    nutrition = compute_nutrition(grams, candidate)

    return MatchedItem(
        name=item.name,
        original_amount=item.amount,
        original_unit=item.unit,
        matched_name=candidate.name,
        matched_source=candidate.source,
        amount_grams=round(grams, 1),
        nutrition=nutrition,
    )
