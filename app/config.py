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
    retrieval_dense_weight: float
    retrieval_lexical_weight: float
    allowed_origins: list[str]
    reasoning_concurrency: int
    reasoning_timeout_seconds: float
    reasoning_evidence_chunks: int
    reasoning_chunk_chars: int


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
            "nvidia/nemotron-3-embed-1b",
        ),
        embedding_batch_size=max(
            1, int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
        ),
        embedding_concurrency=max(
            1, int(os.getenv("EMBEDDING_CONCURRENCY", "4"))
        ),
        chat_model=os.getenv(
            "NVIDIA_CHAT_MODEL",
            "nvidia/nemotron-3.5-lightning-30b-a3b",
        ),
        retrieval_dense_weight=min(
            1.0, max(0.0, float(os.getenv("RETRIEVAL_DENSE_WEIGHT", "0.75")))
        ),
        retrieval_lexical_weight=min(
            1.0, max(0.0, float(os.getenv("RETRIEVAL_LEXICAL_WEIGHT", "0.25")))
        ),
        allowed_origins=[
            origin.strip()
            for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
            if origin.strip()
        ],
        reasoning_concurrency=max(
            1, int(os.getenv("REASONING_CONCURRENCY", "2"))
        ),
        reasoning_timeout_seconds=max(
            5.0, float(os.getenv("REASONING_TIMEOUT_SECONDS", "30"))
        ),
        reasoning_evidence_chunks=max(
            1, int(os.getenv("REASONING_EVIDENCE_CHUNKS", "4"))
        ),
        reasoning_chunk_chars=max(
            300, int(os.getenv("REASONING_CHUNK_CHARS", "900"))
        ),
    )
