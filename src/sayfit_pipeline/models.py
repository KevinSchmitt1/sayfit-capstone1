from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExtractedItem:
    name: str
    amount: Optional[float] = None
    unit: Optional[str] = None


@dataclass
class FoodCandidate:
    source: str
    item_id: str
    item_name: str
    brand: Optional[str]
    kcal_100g: float
    protein_100g: float
    carbs_100g: float
    fat_100g: float
    score: float
    portion_description: Optional[str] = None
    gram_weight: Optional[float] = None


@dataclass
class MatchedItem:
    name: str
    amount: Optional[float]
    unit: Optional[str]
    matched_name: str
    source: str
    item_id: str
    match_score: float
    confidence: str
    grams: float
    portion_description: Optional[str]
    assumptions: list[str] = field(default_factory=list)
    nutrition: dict[str, float] = field(default_factory=dict)


@dataclass
class PipelineResult:
    items: list[MatchedItem]
    totals: dict[str, float]
    timestamp: Optional[str]
    follow_ups: list[str] = field(default_factory=list)
    coaching: list[str] = field(default_factory=list)
