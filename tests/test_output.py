"""Tests for output module."""

from sayfit.models import MatchedItem, NutritionInfo
from sayfit.output import build_meal_result


def _make_item(name="oatmeal", cals=389, prot=16.9, carbs=66, fat=6.9, grams=100):
    return MatchedItem(
        name=name,
        original_amount=grams,
        original_unit="g",
        matched_name=name.upper(),
        matched_source="usda",
        amount_grams=grams,
        nutrition=NutritionInfo(calories=cals, protein=prot, carbs=carbs, fat=fat),
    )


class TestBuildMealResult:
    def test_totals(self):
        items = [
            _make_item("oatmeal", cals=389, prot=16.9, carbs=66, fat=6.9, grams=100),
            _make_item("apple", cals=78, prot=0.45, carbs=20, fat=0.3, grams=150),
        ]
        result = build_meal_result(items, timestamp="2026-03-04T08:00:00")
        assert result.totals.calories == 467.0
        assert result.totals.protein == 17.35
        assert result.timestamp == "2026-03-04T08:00:00"
        assert len(result.items) == 2

    def test_empty_items(self):
        result = build_meal_result([])
        assert result.totals.calories == 0
        assert len(result.items) == 0
