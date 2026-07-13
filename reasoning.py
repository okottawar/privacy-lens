"""
LLM Reasoning Pipeline
The LLM reasons only over retrieved evidence chunks and returns structured JSON.
Uses NVIDIA NIM chat completion endpoint.
"""
import os
import json
import logging
from openai import OpenAI

logger = logging.getLogger("privacylens.reasoning")

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
NVIDIA_BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
CHAT_MODEL = os.environ.get("NVIDIA_CHAT_MODEL", "meta/llama-3.1-70b-instruct")

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not NVIDIA_API_KEY:
            raise RuntimeError("NVIDIA_API_KEY environment variable is not set.")
        _client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)
    return _client


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
  "summary": "<one sentence summary>",
  "explanation": "<2-4 sentence explanation grounded in the evidence>",
  "key_findings": ["<short finding>", ...up to 4],
  "red_flags": ["<short red flag phrase>", ...0-4, empty list if none],
  "positive_indicators": ["<short positive phrase>", ...0-4, empty list if none],
  "evidence": ["<short verbatim-ish snippet under 200 chars>", ...up to 3]
}

Scoring guidance:
- High risk (7-10): vague/broad sharing language, indefinite retention, no deletion rights, dark-pattern consent.
- Medium risk (4-6): some risk factors present but partially mitigated, or evidence is ambiguous/incomplete.
- Low risk (0-3): explicit limits, clear deletion/retention periods, opt-out/opt-in support, minimal collection.
"""


def analyze_category(category: dict, retrieved_chunks: list[dict]) -> dict:
    if not retrieved_chunks:
        return {
            "risk_category": category["name"],
            "risk_score": 5,
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

Analyze the "{category['name']}" risk category based strictly on this evidence. Return the JSON object."""

    client = get_client()
    raw_content = None
    try:
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        raw_content = resp.choices[0].message.content.strip()
        parsed = _parse_json_response(raw_content)
    except Exception as e:
        logger.warning(f"LLM call/parse failed for {category['name']}: {e}. Raw: {raw_content!r}")
        parsed = {
            "risk_score": 5,
            "summary": "Model response could not be parsed; treated as indeterminate.",
            "explanation": "The reasoning step failed to return valid structured output.",
            "key_findings": [],
            "red_flags": [],
            "positive_indicators": [],
            "evidence": [],
        }

    risk_score = _clamp_score(parsed.get("risk_score", 5))

    return {
        "risk_category": category["name"],
        "risk_score": risk_score,
        "summary": parsed.get("summary", ""),
        "explanation": parsed.get("explanation", ""),
        "key_findings": parsed.get("key_findings", []) or [],
        "red_flags": parsed.get("red_flags", []) or [],
        "positive_indicators": parsed.get("positive_indicators", []) or [],
        "evidence": parsed.get("evidence", []) or [],
        "evidence_chunks": [
            {"section": c["section"], "content": c["content"]} for c in retrieved_chunks
        ],
    }


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
