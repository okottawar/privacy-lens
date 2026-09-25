"""Provider contract and normalized errors for embedding services."""
from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    async def embed_texts(
        self,
        texts: list[str],
        input_type: str = "passage",
    ) -> np.ndarray:
        """Embed a list of texts and return float32 vectors."""


class EmbeddingProviderError(RuntimeError):
    """Base class for embedding provider failures."""


class EmbeddingModelUnavailableError(EmbeddingProviderError):
    """The configured model is no longer available from the provider."""
