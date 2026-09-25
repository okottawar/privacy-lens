"""
Embedding pipeline + FAISS vector index.

The index is provider-agnostic; the active embedding backend/model is selected
through environment configuration.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache
import faiss
import numpy as np
from openai import AsyncOpenAI

from app.config import get_settings
from app.embedding_provider import EmbeddingModelUnavailableError
from app.reranking import rerank


class NvidiaEmbeddingProvider:
    """NVIDIA NIM embedding provider using the OpenAI-compatible API."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.nvidia_api_key:
            raise RuntimeError("NVIDIA_API_KEY environment variable is not set.")

        self.settings = settings
        self._client = AsyncOpenAI(
            api_key=settings.nvidia_api_key,
            base_url=settings.nvidia_base_url,
        )
        self._semaphore = asyncio.Semaphore(settings.embedding_concurrency)

    async def embed_texts(
        self,
        texts: list[str],
        input_type: str = "passage",
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype="float32")

        batches = [
            texts[i : i + self.settings.embedding_batch_size]
            for i in range(0, len(texts), self.settings.embedding_batch_size)
        ]

        async def embed_batch(batch: list[str]) -> list[list[float]]:
            async with self._semaphore:
                try:
                    response = await self._client.embeddings.create(
                        input=batch,
                        model=self.settings.embedding_model,
                        encoding_format="float",
                        extra_body={
                            "input_type": input_type,
                            "truncate": "END",
                        },
                    )
                except Exception as exc:
                    if getattr(exc, "status_code", None) == 410:
                        raise EmbeddingModelUnavailableError(
                            f"The configured embedding model "
                            f"'{self.settings.embedding_model}' is unavailable."
                        ) from exc
                    raise
                return [item.embedding for item in response.data]

        results = await asyncio.gather(*(embed_batch(batch) for batch in batches))
        vectors = [vector for batch in results for vector in batch]
        return np.asarray(vectors, dtype="float32")


@lru_cache(maxsize=1)
def get_embedding_provider() -> NvidiaEmbeddingProvider:
    settings = get_settings()
    if settings.embedding_provider != "nvidia":
        raise ValueError(
            f"Unsupported EMBEDDING_PROVIDER='{settings.embedding_provider}'. "
            "Supported provider: 'nvidia'."
        )
    return NvidiaEmbeddingProvider()


async def embed_texts(
    texts: list[str],
    input_type: str = "passage",
) -> np.ndarray:
    """Embed text through the configured provider."""
    provider = get_embedding_provider()
    return await provider.embed_texts(texts, input_type=input_type)


class EmbeddingIndex:
    """Build an in-memory normalized FAISS cosine-similarity index."""

    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self.index: faiss.IndexFlatIP | None = None
        self.dim: int | None = None
        self._token_sets: list[set[str]] = []

    @classmethod
    async def create(cls, chunks: list[dict]) -> "EmbeddingIndex":
        if not chunks:
            raise ValueError("Cannot build an embedding index from zero chunks.")

        instance = cls(chunks)
        vectors = await embed_texts(
            [chunk["content"] for chunk in chunks],
            input_type="passage",
        )

        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError(
                "Embedding provider returned an invalid vector matrix: "
                f"shape={vectors.shape}, chunks={len(chunks)}"
            )

        faiss.normalize_L2(vectors)
        instance.dim = int(vectors.shape[1])
        instance.index = faiss.IndexFlatIP(instance.dim)
        instance.index.add(vectors)
        instance._token_sets = [_tokenize(chunk["content"]) for chunk in chunks]
        return instance

    async def search(self, query: str, k: int = 6) -> list[dict]:
        if self.index is None:
            raise RuntimeError("Embedding index has not been initialized.")

        q_vec = await embed_texts([query], input_type="query")
        faiss.normalize_L2(q_vec)

        k = min(max(1, k), len(self.chunks))
        candidate_k = min(max(k * 8, 32), len(self.chunks))
        dense_scores, indices = self.index.search(q_vec, candidate_k)

        candidates = []
        for dense_score, index in zip(dense_scores[0], indices[0]):
            if index == -1:
                continue
            chunk = self.chunks[int(index)]
            candidates.append({
                "section": chunk["section"],
                "content": chunk["content"],
                "chunk_id": chunk["chunk_id"],
                "similarity": float(dense_score),
            })

        return rerank(query, candidates, k)
