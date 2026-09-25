from app.reranking import jaccard, rerank, tokenize


def test_tokenize_removes_common_stopwords():
    tokens = tokenize("users can delete their personal data")
    assert "users" not in tokens
    assert "delete" in tokens
    assert "personal" in tokens


def test_jaccard_is_zero_for_disjoint_sets():
    assert jaccard({"delete"}, {"cookie"}) == 0.0


def test_rerank_promotes_exact_term_overlap():
    candidates = [
        {
            "chunk_id": "a",
            "section": "General",
            "content": "We operate the service globally.",
            "similarity": 0.90,
        },
        {
            "chunk_id": "b",
            "section": "Deletion",
            "content": "Users may delete personal data from their account.",
            "similarity": 0.80,
        },
    ]

    result = rerank("delete personal data", candidates, 2)

    assert result[0]["chunk_id"] == "b"
    assert result[0]["retrieval_score"] > result[1]["retrieval_score"]
