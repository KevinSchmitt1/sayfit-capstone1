"""Configuration loaded from .env / environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- LLM (OpenAI) ---
    openai_api_key: str = ""
    openai_model: str = "gpt-5.2"

    # --- Embedding ---
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Retrieval ---
    top_k: int = 15

    # --- Data paths ---
    usda_csv: Path = Path("data/processed/usda_nutrition_clean.csv")
    off_csv: Path = Path("data/processed/off_nutrition_clean.csv")
    faiss_index_dir: Path = Path("data/vector_store")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
