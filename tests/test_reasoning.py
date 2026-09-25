from app.reasoning import _clamp_score, _parse_json_response


def test_clamp_score_handles_out_of_range_values():
    assert _clamp_score(-5) == 0
    assert _clamp_score(15) == 10
    assert _clamp_score("6.4") == 6


def test_parse_json_response_accepts_markdown_fence():
    raw = """\
```json
{"risk_score": 7, "summary": "Concern", "evidence": ["quote"]}
```
"""
    parsed = _parse_json_response(raw)

    assert parsed["risk_score"] == 7
    assert parsed["summary"] == "Concern"
