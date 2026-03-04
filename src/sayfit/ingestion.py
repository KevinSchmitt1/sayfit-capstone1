"""
Data ingestion – load USDA + OFF CSVs into a unified food index.

Creates a pandas DataFrame with columns:
  doc_id, source, name, brand, kcal_100g, protein_100g, carbs_100g, fat_100g,
  text_for_embedding
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_usda(csv_path: Path) -> pd.DataFrame:
    """Load the cleaned USDA CSV and normalise columns."""
    df = pd.read_csv(csv_path, dtype={"item_id": str}, low_memory=False)
    df = df.rename(columns={"item_id": "doc_id", "item_name": "name"})
    df["source"] = "usda"
    # Fill missing brand
    df["brand"] = df.get("brand", pd.Series(dtype=str)).fillna("")
    # Build embedding text
    df["text_for_embedding"] = df["name"].str.lower()
    return df[
        ["doc_id", "source", "name", "brand", "kcal_100g", "protein_100g",
         "carbs_100g", "fat_100g", "text_for_embedding"]
    ].dropna(subset=["kcal_100g"])


def load_off(csv_path: Path) -> pd.DataFrame:
    """Load the cleaned OpenFoodFacts CSV and normalise columns."""
    df = pd.read_csv(csv_path, dtype={"item_id": str})
    df = df.rename(columns={"item_id": "doc_id", "item_name": "name"})
    df["source"] = "openfoodfacts"
    df["brand"] = df.get("brand", pd.Series(dtype=str)).fillna("")
    # Richer embedding text: name | brand
    parts = df["name"].str.lower().fillna("")
    brand_part = df["brand"].str.lower().fillna("")
    df["text_for_embedding"] = (parts + " | " + brand_part).str.strip(" |")
    return df[
        ["doc_id", "source", "name", "brand", "kcal_100g", "protein_100g",
         "carbs_100g", "fat_100g", "text_for_embedding"]
    ].dropna(subset=["kcal_100g"])


def build_food_index(usda_path: Path, off_path: Path) -> pd.DataFrame:
    """Merge both data sources into one unified food index."""
    usda = load_usda(usda_path)
    off = load_off(off_path)
    combined = pd.concat([usda, off], ignore_index=True)
    # Drop rows where name is missing
    combined = combined.dropna(subset=["name"])
    combined = combined.reset_index(drop=True)
    print(f"[ingest] USDA rows: {len(usda):,}  |  OFF rows: {len(off):,}  |  total: {len(combined):,}")
    return combined
