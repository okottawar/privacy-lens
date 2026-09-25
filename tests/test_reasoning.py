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


def test_batch_analysis_makes_one_llm_request(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    import app.reasoning as reasoning

    calls = {"count": 0}

    payload = {
        "findings": [
            {
                "risk_category": category["name"],
                "risk_score": 5,
                "confidence": 0.5,
                "disclosure_status": "partial",
                "summary": "Test summary",
                "explanation": "Test explanation.",
                "key_findings": [],
                "red_flags": [],
                "positive_indicators": [],
                "evidence": ["Test evidence"],
                "evidence_chunk_ids": [f"chunk_{i}"],
            }
            for i, category in enumerate(reasoning.RISK_CATEGORIES)
        ]
    }

    class FakeCompletions:
        async def create(self, **kwargs):
            calls["count"] += 1
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=__import__("json").dumps(payload))
                    )
                ]
            )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))
    monkeypatch.setattr(reasoning, "get_client", lambda: fake_client)

    evidence = {
        category["name"]: [
            {"chunk_id": f"chunk_{i}", "section": category["name"], "content": "policy evidence"}
        ]
        for i, category in enumerate(reasoning.RISK_CATEGORIES)
    }

    results = asyncio.run(reasoning.analyze_categories(evidence))

    assert calls["count"] == 1
    assert len(results) == 7
    assert [result["risk_category"] for result in results] == [
        category["name"] for category in reasoning.RISK_CATEGORIES
    ]
