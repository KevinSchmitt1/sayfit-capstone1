from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd


REQUIRED_FILES = ("off_nutrition_clean.csv", "usda_nutrition_clean.csv")


def ensure_processed_csvs(data_dir: Path) -> tuple[Path, Path]:
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    off_csv = processed_dir / REQUIRED_FILES[0]
    usda_csv = processed_dir / REQUIRED_FILES[1]

    if off_csv.exists() and usda_csv.exists():
        return off_csv, usda_csv

    zip_path = data_dir / "food_dbs.zip"
    if not zip_path.exists():
        missing = [p.name for p in (off_csv, usda_csv) if not p.exists()]
        raise FileNotFoundError(
            f"Missing processed csvs {missing} and no zip archive at {zip_path}"
        )

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
        for filename in REQUIRED_FILES:
            if filename not in names:
                raise FileNotFoundError(f"{filename} not found inside {zip_path}")
            zf.extract(filename, path=processed_dir)

    return off_csv, usda_csv


def _coalesce_string(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return df[col].fillna("").astype(str).str.strip()
    return pd.Series([""] * len(df), index=df.index)


def _coalesce_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([pd.NA] * len(df), index=df.index)


def load_food_index(data_dir: Path) -> pd.DataFrame:
    off_csv, usda_csv = ensure_processed_csvs(data_dir)
    off_df = pd.read_csv(off_csv, low_memory=False)
    usda_df = pd.read_csv(usda_csv, low_memory=False)

    off_norm = pd.DataFrame(
        {
            "source": "off",
            "item_id": _coalesce_string(off_df, "item_id"),
            "item_name": _coalesce_string(off_df, "item_name"),
            "brand": _coalesce_string(off_df, "brand"),
            "kcal_100g": _coalesce_numeric(off_df, "kcal_100g"),
            "protein_100g": _coalesce_numeric(off_df, "protein_100g"),
            "carbs_100g": _coalesce_numeric(off_df, "carbs_100g"),
            "fat_100g": _coalesce_numeric(off_df, "fat_100g"),
            "portion_description": _coalesce_string(off_df, "portion_description"),
            "gram_weight": _coalesce_numeric(off_df, "gram_weight"),
        }
    )

    usda_norm = pd.DataFrame(
        {
            "source": "usda",
            "item_id": _coalesce_string(usda_df, "item_id"),
            "item_name": _coalesce_string(usda_df, "item_name"),
            "brand": _coalesce_string(usda_df, "brand"),
            "kcal_100g": _coalesce_numeric(usda_df, "kcal_100g"),
            "protein_100g": _coalesce_numeric(usda_df, "protein_100g"),
            "carbs_100g": _coalesce_numeric(usda_df, "carbs_100g"),
            "fat_100g": _coalesce_numeric(usda_df, "fat_100g"),
            "portion_description": _coalesce_string(usda_df, "portion_description"),
            "gram_weight": _coalesce_numeric(usda_df, "gram_weight"),
        }
    )

    merged = pd.concat([usda_norm, off_norm], ignore_index=True)
    merged = merged.dropna(subset=["kcal_100g", "protein_100g", "carbs_100g", "fat_100g"])
    merged = merged[merged["item_name"] != ""]
    merged = merged.drop_duplicates(subset=["source", "item_id"], keep="first").reset_index(drop=True)

    merged["text_for_embedding"] = (
        merged["item_name"].str.lower()
        + " | "
        + merged["brand"].str.lower()
        + " | "
        + merged["source"].str.lower()
    )
    return merged
