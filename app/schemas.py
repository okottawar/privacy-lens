"""Typed API/intermediate schemas for model-generated analysis."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


DisclosureStatus = Literal["explicit", "partial", "not_found", "unclear"]


class FindingOutput(BaseModel):
    risk_score: int = Field(ge=0, le=10)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    disclosure_status: DisclosureStatus = "unclear"
    summary: str = ""
    explanation: str = ""
    key_findings: list[str] = Field(default_factory=list, max_length=4)
    red_flags: list[str] = Field(default_factory=list, max_length=4)
    positive_indicators: list[str] = Field(default_factory=list, max_length=4)
    evidence: list[str] = Field(default_factory=list, max_length=3)
