"""Tests for models module."""

from sayfit.models import FoodItem, ParsedMeal, NutritionInfo, MatchedItem, MealResult


class TestFoodItem:
    def test_with_all_fields(self):
        item = FoodItem(name="egg", amount=3, unit="piece")
        assert item.name == "egg"
        assert item.amount == 3
        assert item.unit == "piece"

    def test_with_missing_fields(self):
        item = FoodItem(name="oatmeal")
        assert item.name == "oatmeal"
        assert item.amount is None
        assert item.unit is None


class TestParsedMeal:
    def test_creation(self):
        meal = ParsedMeal(
            items=[FoodItem(name="apple", amount=1, unit="piece")],
            timestamp="2026-03-04T08:00:00",
        )
        assert len(meal.items) == 1
        assert meal.timestamp == "2026-03-04T08:00:00"


class TestMealResult:
    def test_serialization(self):
        result = MealResult(
            items=[
                MatchedItem(
                    name="egg",
                    original_amount=3,
                    original_unit="piece",
                    matched_name="EGG, WHOLE",
                    matched_source="usda",
                    amount_grams=150,
                    nutrition=NutritionInfo(calories=232.5, protein=19.5, carbs=1.65, fat=16.5),
                )
            ],
            timestamp="2026-03-04T08:00:00",
            totals=NutritionInfo(calories=232.5, protein=19.5, carbs=1.65, fat=16.5),
        )
        d = result.model_dump()
        assert d["totals"]["calories"] == 232.5
        assert len(d["items"]) == 1
