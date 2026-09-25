"""Deterministic second-stage reranking for retrieved policy chunks."""
from __future__ import annotations

from app.config import get_settings

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "that", "the", "their", "this",
    "to", "user", "users", "we", "what", "with",
}


def tokenize(text: str) -> set[str]:
    tokens = set()
    for token in text.lower().replace("/", " ").replace("-", " ").split():
        token = "".join(char for char in token if char.isalnum())
        if len(token) >= 3 and token not in _STOPWORDS:
            tokens.add(token)
    return tokens


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def rerank(query: str, candidates: list[dict], k: int) -> list[dict]:
    """Rerank dense candidates with lexical overlap and preserve source metadata."""
    settings = get_settings()
    dense_weight = settings.retrieval_dense_weight
    lexical_weight = settings.retrieval_lexical_weight
    total = dense_weight + lexical_weight
    if total <= 0:
        dense_weight, lexical_weight, total = 1.0, 0.0, 1.0
    dense_weight /= total
    lexical_weight /= total

    query_tokens = tokenize(query)
    ranked = []
    for candidate in candidates:
        lexical_score = jaccard(query_tokens, tokenize(candidate["content"]))
        dense_score = float(candidate.get("similarity", 0.0))
        retrieval_score = dense_weight * dense_score + lexical_weight * lexical_score
        ranked.append({
            **candidate,
            "lexical_score": lexical_score,
            "retrieval_score": retrieval_score,
        })

    ranked.sort(key=lambda item: item["retrieval_score"], reverse=True)
    return ranked[:k]
