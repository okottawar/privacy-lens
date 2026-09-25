"""Application configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    nvidia_api_key: str | None
    nvidia_base_url: str
    embedding_provider: str
    embedding_model: str
    embedding_batch_size: int
    embedding_concurrency: int
    chat_model: str


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return immutable process-wide settings.

    The model/provider are explicit configuration instead of implementation
    details, which makes provider migrations and local testing safer.
    """
    return Settings(
        nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
        nvidia_base_url=os.getenv(
            "NVIDIA_BASE_URL",
            "https://integrate.api.nvidia.com/v1",
        ),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "nvidia").lower(),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL",
            "nvidia/llama-3.2-nv-embedqa-1b-v2",
        ),
        embedding_batch_size=max(
            1, int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
        ),
        embedding_concurrency=max(
            1, int(os.getenv("EMBEDDING_CONCURRENCY", "4"))
        ),
        chat_model=os.getenv(
            "NVIDIA_CHAT_MODEL",
            "meta/llama-3.1-70b-instruct",
        ),
    )
