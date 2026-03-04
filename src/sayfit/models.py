"""Pydantic models shared across the pipeline."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ---------- Step 1: Extraction ----------

class FoodItem(BaseModel):
    """A single food item extracted from natural language."""
    name: str = Field(description="Food name as spoken by the user")
    amount: Optional[float] = Field(default=None, description="Numeric amount (may be missing)")
    unit: Optional[str] = Field(default=None, description="Unit: g, kg, ml, l, piece, slice, serving, etc.")


class ParsedMeal(BaseModel):
    """The structured output of step 1 (LLM extraction)."""
    items: list[FoodItem]
    timestamp: Optional[str] = None


# ---------- Step 2–3: Retrieval & Selection ----------

class FoodCandidate(BaseModel):
    """A candidate food row retrieved from the vector store."""
    doc_id: str
    source: str  # "usda" or "openfoodfacts"
    name: str
    brand: Optional[str] = None
    kcal_100g: float
    protein_100g: float
    carbs_100g: float
    fat_100g: float
    score: float = 0.0  # cosine similarity


class MatchedItem(BaseModel):
    """A food item after matching and portion normalisation."""
    name: str
    original_amount: Optional[float] = None
    original_unit: Optional[str] = None
    matched_name: str
    matched_source: str
    amount_grams: float
    nutrition: NutritionInfo


class NutritionInfo(BaseModel):
    """Macros for a specific portion."""
    calories: float
    protein: float
    carbs: float
    fat: float


# Rebuild MatchedItem so forward-ref to NutritionInfo resolves
MatchedItem.model_rebuild()


# ---------- Step 5: Final output ----------

class MealResult(BaseModel):
    """The complete pipeline output for one meal."""
    items: list[MatchedItem]
    timestamp: Optional[str] = None
    totals: NutritionInfo
