from __future__ import annotations


def generate_coaching(totals: dict[str, float], goal: str | None = None) -> list[str]:
    calories = totals.get("calories", 0.0)
    protein = totals.get("protein", 0.0)
    carbs = totals.get("carbs", 0.0)
    fat = totals.get("fat", 0.0)

    tips: list[str] = []

    if calories < 400:
        tips.append("Total calories look low for a full meal day. Consider adding a balanced meal.")
    elif calories > 2800:
        tips.append("Calories are high; consider reducing calorie-dense portions for your next meal.")

    if protein < 60:
        tips.append("Protein is relatively low. Add lean protein sources like yogurt, eggs, chicken, tofu, or legumes.")

    if carbs > protein * 4 and carbs > 250:
        tips.append("Carbs dominate intake. Pair carb-heavy items with protein and fiber-rich foods.")

    if fat > 100:
        tips.append("Fat intake is high. Favor grilled/baked options and reduce added oils.")

    if goal:
        tips.append(f"Goal context noted: {goal}")

    if not tips:
        tips.append("Your macro balance looks reasonable. Keep meal timing and hydration consistent.")

    return tips
