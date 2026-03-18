"""
config.py — Centralised settings using Pydantic BaseSettings.

All values can be overridden via environment variables or a .env file.
Example .env:

    DATASET_ROOT=Data/synthetic_dataset
    FAISS_CASES_DIR=Data/rag_artifacts/vectordb_faiss_minilm_384
    FAISS_GUIDELINES_DIR=Data/rag_artifacts/vectordb_guidelines_minilm_384
    GUIDELINES_DIR=Data/synthetic_dataset/guidelines
    RF_PATH=Data/rag_artifacts/models/risk_rf.joblib
    DEFAULT_RESULTS_JSONL=results/results.jsonl
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    dataset_root: str = "Data/synthetic_dataset"
    faiss_cases_dir: str = "Data/rag_artifacts/vectordb_faiss_minilm_384"
    faiss_guidelines_dir: str = "Data/rag_artifacts/vectordb_guidelines_minilm_384"
    guidelines_dir: str = "Data/synthetic_dataset/guidelines"
    rf_path: str = "Data/rag_artifacts/models/risk_rf.joblib"
    default_results_jsonl: str = "results/results.jsonl"


@lru_cache
def get_settings() -> Settings:
    return Settings()
