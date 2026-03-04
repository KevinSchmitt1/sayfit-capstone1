from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .calculator import nutrition_for_grams
from .coaching import generate_coaching
from .data_loader import load_food_index
from .extractor import ItemExtractor
from .models import MatchedItem, PipelineResult
from .portions import normalize_to_grams
from .retrieval import HashingRetriever
from .selection import choose_best_candidate, confidence_band

DEFAULT_PROFILES_100G = {
    "apple": {"calories": 52.0, "protein": 0.3, "carbs": 14.0, "fat": 0.2},
    "banana": {"calories": 89.0, "protein": 1.1, "carbs": 22.8, "fat": 0.3},
    "egg": {"calories": 143.0, "protein": 12.6, "carbs": 0.7, "fat": 9.5},
    "broccoli": {"calories": 35.0, "protein": 2.4, "carbs": 7.2, "fat": 0.4},
    "rice": {"calories": 130.0, "protein": 2.7, "carbs": 28.0, "fat": 0.3},
}

FRUIT_KEYS = {"apple", "banana"}


class NutritionPipeline:
    def __init__(self, data_dir: Path, top_k: int = 8) -> None:
        self.data_dir = data_dir
        self.food_index = load_food_index(data_dir)
        self.extractor = ItemExtractor()
        self.retriever = HashingRetriever(self.food_index, top_k=top_k)

    def run(self, payload: dict[str, Any], goal: str | None = None) -> PipelineResult:
        text = str(payload.get("text", "")).strip()
        timestamp = payload.get("timestamp")
        extracted = self.extractor.extract(text)

        results: list[MatchedItem] = []
        follow_ups: list[str] = []

        totals = {
            "calories": 0.0,
            "protein": 0.0,
            "carbs": 0.0,
            "fat": 0.0,
        }

        for item in extracted:
            candidates = self.retriever.retrieve(item.name)
            match = choose_best_candidate(item.name, candidates)

            if match is None:
                follow_ups.append(f"Could not match '{item.name}'. Please clarify food type and amount.")
                continue

            conf = confidence_band(match.score)
            if conf == "low":
                follow_ups.append(
                    f"Low confidence for '{item.name}' matched to '{match.item_name}'. "
                    "Please confirm brand/type and approximate grams."
                )

            portion = normalize_to_grams(item, match)
            nutrition = nutrition_for_grams(match, portion.grams)
            default_key = self._default_profile_key(item.name)
            if default_key:
                default_nutrition = self._default_nutrition(default_key, portion.grams)
                if self._implausible_match(default_key, match):
                    nutrition = default_nutrition
                    portion.assumptions.append(
                        f"Used default nutrition profile for '{default_key}' due implausible DB match"
                    )
                    follow_ups.append(
                        f"'{item.name}' match looked noisy in DB. Using default profile; confirm exact product if branded."
                    )

            for key in totals:
                totals[key] += nutrition[key]

            results.append(
                MatchedItem(
                    name=item.name,
                    amount=item.amount,
                    unit=item.unit,
                    matched_name=match.item_name,
                    source=match.source,
                    item_id=match.item_id,
                    match_score=round(match.score, 4),
                    confidence=conf,
                    grams=round(portion.grams, 2),
                    portion_description=match.portion_description,
                    assumptions=portion.assumptions,
                    nutrition=nutrition,
                )
            )

        rounded_totals = {k: round(v, 2) for k, v in totals.items()}
        coaching = generate_coaching(rounded_totals, goal=goal)

        return PipelineResult(
            items=results,
            totals=rounded_totals,
            timestamp=timestamp,
            follow_ups=follow_ups,
            coaching=coaching,
        )

    @staticmethod
    def _default_profile_key(name: str) -> str | None:
        n = (name or "").lower()
        for key in DEFAULT_PROFILES_100G:
            if key in n:
                return key
        return None

    @staticmethod
    def _default_nutrition(key: str, grams: float) -> dict[str, float]:
        base = DEFAULT_PROFILES_100G[key]
        factor = grams / 100.0
        return {k: round(v * factor, 2) for k, v in base.items()}

    @staticmethod
    def _implausible_match(key: str, match) -> bool:
        if key in FRUIT_KEYS:
            return match.kcal_100g > 95 or match.carbs_100g > 20
        return False

    @staticmethod
    def result_to_dict(result: PipelineResult) -> dict[str, Any]:
        return {
            "items": [
                {
                    "name": i.name,
                    "amount": i.amount,
                    "unit": i.unit,
                    "matched_name": i.matched_name,
                    "source": i.source,
                    "item_id": i.item_id,
                    "match_score": i.match_score,
                    "confidence": i.confidence,
                    "grams": i.grams,
                    "portion_description": i.portion_description,
                    "assumptions": i.assumptions,
                    "nutrition": i.nutrition,
                }
                for i in result.items
            ],
            "timestamp": result.timestamp,
            "totals": result.totals,
            "follow_ups": result.follow_ups,
            "coaching": result.coaching,
        }


def run_pipeline_file(input_path: Path, output_path: Path, data_dir: Path, goal: str | None = None) -> dict[str, Any]:
    pipeline = NutritionPipeline(data_dir=data_dir, top_k=8)
    payload = json.loads(input_path.read_text())
    result = pipeline.run(payload, goal=goal)
    out = pipeline.result_to_dict(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, indent=2))
    return out
