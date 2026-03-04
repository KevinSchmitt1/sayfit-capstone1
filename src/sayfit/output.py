"""
Step 5 – Output formatting.

Produces a rich table in the terminal and a MealResult JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from sayfit.models import MatchedItem, MealResult, NutritionInfo


def build_meal_result(
    items: list[MatchedItem],
    timestamp: str | None = None,
) -> MealResult:
    """Aggregate matched items into a MealResult with totals."""
    totals = NutritionInfo(calories=0, protein=0, carbs=0, fat=0)
    for item in items:
        totals.calories += item.nutrition.calories
        totals.protein += item.nutrition.protein
        totals.carbs += item.nutrition.carbs
        totals.fat += item.nutrition.fat

    # Round totals
    totals.calories = round(totals.calories, 2)
    totals.protein = round(totals.protein, 2)
    totals.carbs = round(totals.carbs, 2)
    totals.fat = round(totals.fat, 2)

    return MealResult(items=items, timestamp=timestamp, totals=totals)


def print_meal_table(result: MealResult, console: Console | None = None) -> None:
    """Pretty-print the meal result as a rich table."""
    if console is None:
        console = Console()

    table = Table(
        title="🍽  Meal Nutrition Breakdown",
        show_lines=True,
        header_style="bold cyan",
    )
    table.add_column("Food", style="bold")
    table.add_column("Matched", style="dim")
    table.add_column("Source", style="dim")
    table.add_column("Grams", justify="right")
    table.add_column("Calories", justify="right", style="yellow")
    table.add_column("Protein (g)", justify="right", style="green")
    table.add_column("Carbs (g)", justify="right", style="blue")
    table.add_column("Fat (g)", justify="right", style="red")

    for item in result.items:
        table.add_row(
            item.name,
            item.matched_name,
            item.matched_source,
            f"{item.amount_grams:.0f}",
            f"{item.nutrition.calories:.1f}",
            f"{item.nutrition.protein:.1f}",
            f"{item.nutrition.carbs:.1f}",
            f"{item.nutrition.fat:.1f}",
        )

    # Totals row
    table.add_row(
        "[bold]TOTAL[/bold]", "", "", "",
        f"[bold yellow]{result.totals.calories:.1f}[/bold yellow]",
        f"[bold green]{result.totals.protein:.1f}[/bold green]",
        f"[bold blue]{result.totals.carbs:.1f}[/bold blue]",
        f"[bold red]{result.totals.fat:.1f}[/bold red]",
    )

    console.print()
    if result.timestamp:
        console.print(f"  ⏰ Timestamp: {result.timestamp}", style="dim")
    console.print(table)
    console.print()


def save_result_json(result: MealResult, path: Path) -> None:
    """Write MealResult to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[output] Saved → {path}")
