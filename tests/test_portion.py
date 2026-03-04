"""Tests for the portion normalization module."""

import pytest

from sayfit.models import FoodCandidate, FoodItem, NutritionInfo
from sayfit.portion import (
    DEFAULT_PORTION_GRAMS,
    compute_nutrition,
    estimate_grams,
    normalise_and_compute,
)


# ---------- Fixtures ----------

def _candidate(kcal=200, protein=10, carbs=25, fat=8) -> FoodCandidate:
    return FoodCandidate(
        doc_id="test-1",
        source="usda",
        name="test food",
        kcal_100g=kcal,
        protein_100g=protein,
        carbs_100g=carbs,
        fat_100g=fat,
        score=0.95,
    )


# ---------- estimate_grams ----------

class TestEstimateGrams:
    def test_explicit_grams(self):
        item = FoodItem(name="rice", amount=200, unit="g")
        assert estimate_grams(item) == 200.0

    def test_explicit_kg(self):
        item = FoodItem(name="rice", amount=1.5, unit="kg")
        assert estimate_grams(item) == 1500.0

    def test_explicit_ml(self):
        item = FoodItem(name="almond milk", amount=200, unit="ml")
        assert estimate_grams(item) == 200.0

    def test_pieces_with_default(self):
        item = FoodItem(name="egg", amount=3, unit="piece")
        assert estimate_grams(item) == 150.0  # 3 × 50g

    def test_slices_with_default(self):
        item = FoodItem(name="bread", amount=2, unit="slice")
        assert estimate_grams(item) == 60.0  # 2 × 30g

    def test_serving(self):
        item = FoodItem(name="pizza", amount=1, unit="serving")
        assert estimate_grams(item) == 150.0

    def test_amount_no_unit_assumes_grams(self):
        item = FoodItem(name="chicken", amount=250, unit=None)
        assert estimate_grams(item) == 250.0

    def test_no_amount_no_unit_uses_default(self):
        item = FoodItem(name="banana", amount=None, unit=None)
        assert estimate_grams(item) == 120.0  # banana default

    def test_unknown_food_no_amount_fallback(self):
        item = FoodItem(name="xyzfood", amount=None, unit=None)
        assert estimate_grams(item) == DEFAULT_PORTION_GRAMS

    def test_cups(self):
        item = FoodItem(name="rice", amount=1, unit="cup")
        assert estimate_grams(item) == 240.0

    def test_tablespoons(self):
        item = FoodItem(name="honey", amount=2, unit="tbsp")
        assert estimate_grams(item) == 30.0

    def test_unit_alias_grams(self):
        item = FoodItem(name="oatmeal", amount=100, unit="grams")
        assert estimate_grams(item) == 100.0


# ---------- compute_nutrition ----------

class TestComputeNutrition:
    def test_100g(self):
        c = _candidate(kcal=200, protein=10, carbs=25, fat=8)
        n = compute_nutrition(100, c)
        assert n.calories == 200.0
        assert n.protein == 10.0
        assert n.carbs == 25.0
        assert n.fat == 8.0

    def test_50g(self):
        c = _candidate(kcal=200, protein=10, carbs=25, fat=8)
        n = compute_nutrition(50, c)
        assert n.calories == 100.0
        assert n.protein == 5.0

    def test_250g(self):
        c = _candidate(kcal=100, protein=5, carbs=20, fat=3)
        n = compute_nutrition(250, c)
        assert n.calories == 250.0


# ---------- normalise_and_compute ----------

class TestNormaliseAndCompute:
    def test_full_flow(self):
        item = FoodItem(name="oatmeal", amount=100, unit="g")
        candidate = _candidate(kcal=389, protein=16.9, carbs=66, fat=6.9)
        result = normalise_and_compute(item, candidate)
        assert result.name == "oatmeal"
        assert result.matched_name == "test food"
        assert result.amount_grams == 100.0
        assert result.nutrition.calories == 389.0

    def test_pieces(self):
        item = FoodItem(name="egg", amount=2, unit="piece")
        candidate = _candidate(kcal=155, protein=13, carbs=1.1, fat=11)
        result = normalise_and_compute(item, candidate)
        assert result.amount_grams == 100.0  # 2 × 50g
        assert result.nutrition.calories == 155.0
