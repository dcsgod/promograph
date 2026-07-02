"""Central configuration, loaded from environment / .env.

Everything downstream imports `settings` from here so paths and connection
details live in exactly one place.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = two levels up from this file (backend/config.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- local data warehouse ---
    tpo_duckdb_path: str = "data/tpo.duckdb"
    tpo_embeddings_path: str = "data/embeddings.parquet"
    tpo_lgbm_model_path: str = "data/lgbm_lift.txt"

    # --- Graph store backend: "duckdb" (no Docker needed) or "neo4j" ---
    tpo_graph_backend: str = "duckdb"

    # --- Neo4j ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "tpo_password"

    # --- LLM (Ollama, OpenAI-compatible) ---
    openai_base_url: str = "http://localhost:11434/v1"
    openai_api_key: str = "ollama"
    tpo_llm_model: str = "qwen2.5:3b"

    # --- Genie backend ---
    tpo_genie_backend: str = "mock"  # "mock" | "databricks"

    # --- Databricks (only for real Genie) ---
    databricks_host: str = ""
    databricks_token: str = ""
    databricks_warehouse_id: str = ""
    databricks_catalog: str = "tpo"
    databricks_schema: str = "dunnhumby"
    genie_space_sales: str = ""
    genie_space_promo: str = ""
    genie_space_inventory: str = ""

    # --- resolved absolute paths ---
    @property
    def duckdb_path(self) -> Path:
        return self._abs(self.tpo_duckdb_path)

    @property
    def embeddings_path(self) -> Path:
        return self._abs(self.tpo_embeddings_path)

    @property
    def lgbm_model_path(self) -> Path:
        return self._abs(self.tpo_lgbm_model_path)

    @staticmethod
    def _abs(p: str) -> Path:
        path = Path(p)
        return path if path.is_absolute() else (REPO_ROOT / path)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
