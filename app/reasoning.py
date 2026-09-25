"""Evidence-grounded privacy policy reasoning with a single batched LLM call."""
from __future__ import annotations

import asyncio
import json
import logging
from functools import lru_cache

from openai import APITimeoutError, AsyncOpenAI

from app.config import get_settings
from app.schemas import BatchAnalysisOutput

logger = logging.getLogger("privacelens.reasoning")


@lru_cache(maxsize=1)
def get_client() -> AsyncOpenAI:
    settings = get_settings()
    if not settings.nvidia_api_key:
        raise RuntimeError("NVIDIA_API_KEY environment variable is not set.")
    return AsyncOpenAI(
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url,
        timeout=settings.reasoning_timeout_seconds,
    )


RISK_CATEGORIES = [
    {"name": "Data Collection", "query": "what personal data information is collected from users"},
    {"name": "Third-Party Sharing", "query": "sharing data with third parties advertisers partners affiliates"},
    {"name": "Retention", "query": "how long is data retained data retention deletion period"},
    {"name": "Deletion Rights", "query": "user rights to delete access correct or export their data"},
    {"name": "Tracking / Cookies", "query": "cookies tracking technologies pixels analytics behavioral advertising"},
    {"name": "Transparency", "query": "policy changes notification transparency about data practices"},
    {"name": "Consent Mechanisms", "query": "user consent opt-in opt-out mechanisms for data processing"},
]

SYSTEM_PROMPT = """You are a privacy policy risk analyst. Analyze exactly seven categories using only the supplied policy evidence.

Return ONLY one JSON object with this shape:
{
  "findings": [
    {
      "risk_category": "Data Collection",
      "risk_score": 0,
      "confidence": 0.0,
      "disclosure_status": "explicit",
      "summary": "one sentence",
      "explanation": "2-4 sentences",
      "key_findings": [],
      "red_flags": [],
      "positive_indicators": [],
      "evidence": [],
      "evidence_chunk_ids": []
    }
  ]
}

Rules:
- Return exactly one finding per category.
- Use only evidence supplied for that category.
- evidence_chunk_ids must be chunk IDs shown in that category evidence.
- Keep evidence quotes under 200 characters and lists concise.
- If evidence is insufficient: risk_score=5, confidence<=0.25, disclosure_status="not_found" or "unclear".
- Do not use external knowledge or make legal conclusions.
"""


def _fallback_finding(category: dict, reason: str, status: str = "unclear") -> dict:
    return {
        "risk_category": category["name"],
        "risk_score": 5,
        "confidence": 0.0,
        "disclosure_status": status,
        "summary": reason,
        "explanation": reason,
        "key_findings": [],
        "red_flags": [],
        "positive_indicators": [],
        "evidence": [],
        "evidence_chunk_ids": [],
        "evidence_chunks": [],
    }


async def analyze_categories(category_evidence: dict[str, list[dict]]) -> list[dict]:
    """Analyze all seven categories with exactly one bounded LLM request."""
    settings = get_settings()
    evidence_blocks = []
    for category in RISK_CATEGORIES:
        selected = category_evidence.get(category["name"], [])[: settings.reasoning_evidence_chunks]
        if not selected:
            evidence_blocks.append(f"=== {category["name"]} ===\nNO RELEVANT EVIDENCE RETRIEVED")
            continue
        body = "\n\n".join(
            f"[Chunk ID: {chunk["chunk_id"]}] [Section: {chunk["section"]}]\n"
            f"{chunk["content"][:settings.reasoning_chunk_chars]}"
            for chunk in selected
        )
        evidence_blocks.append(
            f"=== {category["name"]} ===\n"
            f"Category query: {category["query"]}\n{body}"
        )

    user_prompt = "Analyze all seven privacy-risk categories from these evidence bundles only.\n\n" + "\n\n".join(evidence_blocks)
    started = asyncio.get_running_loop().time()
    try:
        response = await asyncio.wait_for(
            get_client().chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=2600,
                response_format={"type": "json_object"},
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            ),
            timeout=settings.reasoning_timeout_seconds,
        )
        raw = (response.choices[0].message.content or "").strip()
        parsed = BatchAnalysisOutput.model_validate(_parse_json_response(raw))
    except (asyncio.TimeoutError, APITimeoutError):
        logger.warning("reasoning.batch.timeout timeout_seconds=%s", settings.reasoning_timeout_seconds)
        return [_fallback_finding(c, "Batch reasoning timed out; treated as indeterminate.") for c in RISK_CATEGORIES]
    except Exception as exc:
        logger.exception("reasoning.batch.failed")
        return [_fallback_finding(c, f"Batch reasoning failed; treated as indeterminate: {exc}") for c in RISK_CATEGORIES]

    by_name = {item.risk_category: item for item in parsed.findings}
    results = []
    for category in RISK_CATEGORIES:
        finding = by_name.get(category["name"])
        evidence = category_evidence.get(category["name"], [])
        if finding is None:
            results.append(_fallback_finding(category, "The model did not return a finding for this category."))
            continue
        cited_ids, cited_chunks = resolve_evidence(finding.evidence_chunk_ids, evidence)
        results.append({
            **finding.model_dump(),
            "evidence_chunk_ids": cited_ids,
            "evidence_chunks": cited_chunks,
        })
    logger.info("reasoning.batch.complete duration_ms=%d findings=%d", int((asyncio.get_running_loop().time() - started) * 1000), len(results))
    return results


async def analyze_category(category: dict, retrieved_chunks: list[dict]) -> dict:
    """Compatibility adapter; production path uses analyze_categories()."""
    return (await analyze_categories({category["name"]: retrieved_chunks}))[0 if True else 0]


def resolve_evidence(requested_ids: list[str], retrieved_chunks: list[dict]) -> tuple[list[str], list[dict]]:
    available = {chunk["chunk_id"]: chunk for chunk in retrieved_chunks}
    cited_ids = [chunk_id for chunk_id in dict.fromkeys(requested_ids) if chunk_id in available]
    cited_chunks = [
        {"chunk_id": chunk_id, "section": available[chunk_id]["section"], "content": available[chunk_id]["content"]}
        for chunk_id in cited_ids
    ]
    return cited_ids, cited_chunks


def _parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model response.")
    return json.loads(raw[start : end + 1])


def _clamp_score(score) -> int:
    try:
        score = int(round(float(score)))
    except (TypeError, ValueError):
        score = 5
    return max(0, min(10, score))
