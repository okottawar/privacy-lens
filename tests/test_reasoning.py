from app.reasoning import _clamp_score, _parse_json_response, resolve_evidence


def test_clamp_score_handles_out_of_range_values():
    assert _clamp_score(-5) == 0
    assert _clamp_score(15) == 10
    assert _clamp_score("6.4") == 6


def test_parse_json_response_accepts_markdown_fence():
    raw = '''```json
{"risk_score": 7, "summary": "Concern", "evidence": ["quote"]}
```'''

    parsed = _parse_json_response(raw)

    assert parsed["risk_score"] == 7
    assert parsed["summary"] == "Concern"


def test_resolve_evidence_only_returns_retrieved_chunk_ids():
    chunks = [
        {"chunk_id": "chunk_1", "section": "Deletion", "content": "Users may delete data."},
        {"chunk_id": "chunk_2", "section": "Cookies", "content": "We use cookies."},
    ]

    ids, cited = resolve_evidence(["chunk_2", "missing", "chunk_2"], chunks)

    assert ids == ["chunk_2"]
    assert cited == [chunks[1]]
