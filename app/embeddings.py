"""
Embedding Pipeline + FAISS Vector Database
Uses NVIDIA NIM embedding endpoint (OpenAI-compatible) via the `openai` client.
"""
import os
import numpy as np
import faiss
from openai import OpenAI

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
EMBED_MODEL = os.environ.get("NVIDIA_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not NVIDIA_API_KEY:
            raise RuntimeError("NVIDIA_API_KEY environment variable is not set.")
        _client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)
    return _client


def embed_texts(texts: list[str], input_type: str = "passage") -> np.ndarray:
    """
    input_type: "passage" for documents being indexed, "query" for search queries.
    NVIDIA NIM E5 embedding models require this field via extra_body.
    """
    client = get_client()
    # NIM embedding endpoints accept batches; keep batches modest for reliability.
    all_vecs = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(
            input=batch,
            model=EMBED_MODEL,
            encoding_format="float",
            extra_body={"input_type": input_type, "truncate": "END"},
        )
        vecs = [d.embedding for d in resp.data]
        all_vecs.extend(vecs)
    return np.array(all_vecs, dtype="float32")


class EmbeddingIndex:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        texts = [c["content"] for c in chunks]
        vectors = embed_texts(texts, input_type="passage")

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(vectors)
        self.dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vectors)

    def search(self, query: str, k: int = 6) -> list[dict]:
        q_vec = embed_texts([query], input_type="query")
        faiss.normalize_L2(q_vec)
        k = min(k, len(self.chunks))
        scores, idxs = self.index.search(q_vec, k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({
                "section": chunk["section"],
                "content": chunk["content"],
                "chunk_id": chunk["chunk_id"],
                "similarity": float(score),
            })
        return results
