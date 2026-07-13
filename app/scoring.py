"""
Risk Scoring Engine — deterministic aggregation, NOT LLM-generated.
The LLM scores each category; this module combines them.
"""

# Weight riskier-by-nature categories slightly higher
CATEGORY_WEIGHTS = {
    "Data Collection": 1.0,
    "Third-Party Sharing": 1.3,
    "Retention": 1.0,
    "Deletion Rights": 1.1,
    "Tracking / Cookies": 1.0,
    "Transparency": 0.8,
    "Consent Mechanisms": 1.1,
}


def compute_overall(findings: list[dict]) -> dict:
    if not findings:
        return {"score": 5, "label": "Indeterminate"}

    total_weight = 0.0
    weighted_sum = 0.0
    for f in findings:
        w = CATEGORY_WEIGHTS.get(f["risk_category"], 1.0)
        weighted_sum += f["risk_score"] * w
        total_weight += w

    score = weighted_sum / total_weight if total_weight else 5.0
    score = round(score, 1)
    # Round to nearest int for display consistent with per-category scale, but keep 1 decimal feel
    display_score = int(round(score))

    label = _label_for(display_score)
    return {"score": display_score, "label": label}


def _label_for(score: int) -> str:
    if score <= 2:
        return "Excellent Privacy Practices"
    if score <= 4:
        return "Good — Minor Concerns"
    if score <= 6:
        return "Moderate Risk"
    if score <= 8:
        return "High Risk"
    return "Severe Risk"
