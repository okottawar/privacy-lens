from app.scoring import compute_overall


def test_compute_overall_uses_weighted_average():
    findings = [
        {"risk_category": "Data Collection", "risk_score": 2},
        {"risk_category": "Third-Party Sharing", "risk_score": 8},
    ]

    result = compute_overall(findings)

    # (2*1.0 + 8*1.3) / 2.3 ~= 5.39 -> display score 5
    assert result["score"] == 5
    assert result["label"] == "Moderate Risk"


def test_compute_overall_empty_is_indeterminate():
    assert compute_overall([]) == {"score": 5, "label": "Indeterminate"}
