"""
AST-GuardCache Configuration Module.

Loads environment variables for OpenRouter LLM configuration and application settings.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


class Settings:
    """Application-wide settings sourced from environment variables."""

    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = os.getenv(
        "OPENROUTER_MODEL", "qwen/qwen-2.5-coder-32b-instruct"
    )

    APP_ENV: str = os.getenv("APP_ENV", "development")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Retriever settings
    RETRIEVER_TOP_K: int = 4
    RRF_K: int = 60  # Reciprocal Rank Fusion constant

    # Cache settings
    CACHE_SIMILARITY_THRESHOLD: float = 0.88

    # Verifier settings
    MAX_AST_RETRIES: int = 2

    # Embedding model
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Database
    SQLITE_DB_PATH: str = ":memory:"


settings = Settings()
