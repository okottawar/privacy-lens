"""
LLM Reasoning Pipeline
The LLM reasons only over retrieved evidence chunks and returns structured JSON.
Uses NVIDIA NIM chat completion endpoint.
"""
import json
import logging
from functools import lru_cache

from openai import AsyncOpenAI

from app.config import get_settings
from app.schemas import FindingOutput

logger = logging.getLogger("privacylens.reasoning")


@lru_cache(maxsize=1)
def get_client() -> AsyncOpenAI:
    settings = get_settings()
    if not settings.nvidia_api_key:
        raise RuntimeError("NVIDIA_API_KEY environment variable is not set.")
    return AsyncOpenAI(api_key=settings.nvidia_api_key, base_url=settings.nvidia_base_url)


RISK_CATEGORIES = [
    {
        "name": "Data Collection",
        "query": "what personal data information is collected from users",
    },
    {
        "name": "Third-Party Sharing",
        "query": "sharing data with third parties advertisers partners affiliates",
    },
    {
        "name": "Retention",
        "query": "how long is data retained data retention deletion period",
    },
    {
        "name": "Deletion Rights",
        "query": "user rights to delete access correct or export their data",
    },
    {
        "name": "Tracking / Cookies",
        "query": "cookies tracking technologies pixels analytics behavioral advertising",
    },
    {
        "name": "Transparency",
        "query": "policy changes notification transparency about data practices",
    },
    {
        "name": "Consent Mechanisms",
        "query": "user consent opt-in opt-out mechanisms for data processing",
    },
]

SYSTEM_PROMPT = """You are a privacy policy risk analyst. You must reason ONLY over the evidence \
chunks provided to you — do not invent facts not present in the evidence. If the evidence does not \
address the category, say so explicitly and score conservatively (5) for "unknown/undisclosed".

Respond with ONLY a single JSON object, no markdown fences, no preamble, matching this exact schema:
{
  "risk_score": <integer 0-10, 0=no risk/excellent, 10=severe risk>,
  "confidence": <number 0.0-1.0 representing confidence in the assessment>,
  "disclosure_status": "<explicit|partial|not_found|unclear>",
  "summary": "<one sentence summary>",
  "explanation": "<2-4 sentence explanation grounded in the evidence>",
  "key_findings": ["<short finding>", ...up to 4],
  "red_flags": ["<short red flag phrase>", ...0-4, empty list if none],
  "positive_indicators": ["<short positive phrase>", ...0-4, empty list if none],
  "evidence": ["<short verbatim-ish snippet under 200 chars>", ...up to 3],
  "evidence_chunk_ids": ["<chunk_id from the supplied evidence>", ...up to 3]
}

Scoring guidance:
- High risk (7-10): vague/broad sharing language, indefinite retention, no deletion rights, dark-pattern consent.
- Medium risk (4-6): some risk factors present but partially mitigated, or evidence is ambiguous/incomplete.
- Low risk (0-3): explicit limits, clear deletion/retention periods, opt-out/opt-in support, minimal collection.
"""


async def analyze_category(category: dict, retrieved_chunks: list[dict]) -> dict:
    if not retrieved_chunks:
        return {
            "risk_category": category["name"],
            "risk_score": 5,
            "confidence": 0.0,
            "disclosure_status": "not_found",
            "summary": "No relevant evidence retrieved for this category.",
            "explanation": "The policy did not contain content that matched this category well enough to assess.",
            "key_findings": [],
            "red_flags": ["No disclosure found for this category"],
            "positive_indicators": [],
            "evidence": [],
            "evidence_chunks": [],
        }

    evidence_text = "\n\n".join(
        f"[Section: {c['section']}]\n{c['content'][:1200]}" for c in retrieved_chunks
    )

    user_prompt = f"""Category to analyze: {category['name']}

Evidence retrieved from the privacy policy (top {len(retrieved_chunks)} relevant chunks):

{evidence_text}

Analyze the "{category['name']}" risk category based strictly on this evidence. In evidence_chunk_ids, select only chunk IDs supplied in the evidence above that directly support the finding. Return the JSON object."""

    client = get_client()
    raw_content = None
    try:
        resp = await client.chat.completions.create(
            model=get_settings().chat_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        raw_content = resp.choices[0].message.content.strip()
        parsed = FindingOutput.model_validate(_parse_json_response(raw_content)).model_dump()
    except Exception as e:
        logger.warning(f"LLM call/parse failed for {category['name']}: {e}. Raw: {raw_content!r}")
        parsed = {
            "risk_score": 5,
            "confidence": 0.0,
            "disclosure_status": "unclear",
            "summary": "Model response could not be parsed; treated as indeterminate.",
            "explanation": "The reasoning step failed to return valid structured output.",
            "key_findings": [],
            "red_flags": [],
            "positive_indicators": [],
            "evidence": [],
            "evidence_chunk_ids": [],
        }

    validated = FindingOutput.model_validate({
        **parsed,
        "risk_score": _clamp_score(parsed.get("risk_score", 5)),
    })

    cited_ids, cited_chunks = resolve_evidence(validated.evidence_chunk_ids, retrieved_chunks)

    if validated.evidence_chunk_ids and not cited_ids:
        validated = validated.model_copy(
            update={
                "confidence": 0.0,
                "disclosure_status": "unclear",
                "evidence": [],
                "evidence_chunk_ids": [],
            }
        )

    return {
        "risk_category": category["name"],
        "risk_score": validated.risk_score,
        "confidence": validated.confidence,
        "disclosure_status": validated.disclosure_status,
        "summary": validated.summary,
        "explanation": validated.explanation,
        "key_findings": validated.key_findings,
        "red_flags": validated.red_flags,
        "positive_indicators": validated.positive_indicators,
        "evidence": validated.evidence,
        "evidence_chunk_ids": cited_ids,
        "evidence_chunks": cited_chunks,
    }


def resolve_evidence(
    requested_ids: list[str],
    retrieved_chunks: list[dict],
) -> tuple[list[str], list[dict]]:
    """Resolve model-selected IDs against retrieved chunks from the current request."""
    available = {chunk["chunk_id"]: chunk for chunk in retrieved_chunks}
    cited_ids = [
        chunk_id
        for chunk_id in dict.fromkeys(requested_ids)
        if chunk_id in available
    ]
    cited_chunks = [
        {
            "chunk_id": chunk_id,
            "section": available[chunk_id]["section"],
            "content": available[chunk_id]["content"],
        }
        for chunk_id in cited_ids
    ]
    return cited_ids, cited_chunks

def _parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    # strip markdown fences if the model added them anyway
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    # find first { ... last }
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model response.")
    return json.loads(raw[start:end + 1])


def _clamp_score(score) -> int:
    try:
        score = int(round(float(score)))
    except (TypeError, ValueError):
        score = 5
    return max(0, min(10, score))
