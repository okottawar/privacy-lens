from pydantic import ValidationError
import pytest

from app.schemas import FindingOutput


def test_finding_output_applies_safe_defaults():
    result = FindingOutput(risk_score=6)

    assert result.confidence == 0.5
    assert result.disclosure_status == "unclear"
    assert result.key_findings == []


def test_finding_output_rejects_invalid_score():
    with pytest.raises(ValidationError):
        FindingOutput(risk_score=11)


def test_finding_output_rejects_invalid_disclosure_status():
    with pytest.raises(ValidationError):
        FindingOutput(risk_score=5, disclosure_status="unknown")
