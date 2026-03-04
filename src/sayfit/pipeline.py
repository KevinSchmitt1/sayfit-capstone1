"""
End-to-end pipeline: text → structured items → retrieval → selection → nutrition → output.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from sayfit.config import Settings
from sayfit.extraction import extract_items
from sayfit.models import MatchedItem, MealResult, ParsedMeal
from sayfit.output import build_meal_result, print_meal_table, save_result_json
from sayfit.portion import normalise_and_compute
from sayfit.retrieval import retrieve_candidates
from sayfit.selection import select_best
from sayfit.vector_store import FoodVectorStore


class SayFitPipeline:
    """Wires all five steps together."""

    def __init__(
        self,
        settings: Settings,
        store: FoodVectorStore,
        *,
        use_llm_rerank: bool = True,
        console: Console | None = None,
    ) -> None:
        self.settings = settings
        self.store = store
        self.use_llm_rerank = use_llm_rerank
        self.console = console or Console()

    # ------------------------------------------------------------------
    def run(
        self,
        text: str,
        *,
        timestamp: str | None = None,
        save_path: Path | None = None,
    ) -> MealResult:
        """Execute the full pipeline on a single text input."""
        s = self.settings

        # Step 1: LLM extraction
        self.console.print("[bold cyan]Step 1:[/] Extracting food items …")
        parsed = extract_items(
            text,
            api_key=s.openai_api_key,
            model=s.openai_model,
            timestamp=timestamp,
        )
        self.console.print(f"  → Found {len(parsed.items)} item(s): "
                           f"{[i.name for i in parsed.items]}")

        # Steps 2–4 for each item
        matched_items: list[MatchedItem] = []
        for item in parsed.items:
            # Step 2: Retrieve candidates
            candidates = retrieve_candidates(item, self.store, top_k=s.top_k)
            if not candidates:
                self.console.print(f"  ⚠  No candidates found for '{item.name}' – skipping",
                                   style="yellow")
                continue

            # Step 3: Select best
            best = select_best(
                item.name,
                candidates,
                use_llm=self.use_llm_rerank,
                api_key=s.openai_api_key,
                model=s.openai_model,
            )
            self.console.print(
                f"  ✔ [green]{item.name}[/green] → "
                f"{best.name} ({best.source}, score={best.score:.3f})"
            )

            # Step 4: Portion normalisation + math
            matched = normalise_and_compute(item, best)
            matched_items.append(matched)

        # Step 5: Output
        result = build_meal_result(matched_items, timestamp=parsed.timestamp)
        print_meal_table(result, self.console)

        if save_path:
            save_result_json(result, save_path)

        return result
