"""
PrivacyLens backend — RAG pipeline for privacy policy risk analysis.
Fetch -> Parse/Clean -> Chunk -> Embed (NVIDIA NIM) -> FAISS -> Retrieve -> LLM (NVIDIA NIM) -> Score -> Report
"""
import asyncio
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.retrieval import fetch_and_parse, chunk_sections
from app.embeddings import EmbeddingIndex
from app.reasoning import analyze_category, RISK_CATEGORIES
from app.scoring import compute_overall

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("privacylens")

app = FastAPI(title="PrivacyLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # public demo — tighten if you deploy for real users
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    url: str
    policy_text: str | None = None


@app.get("/")
def root():
    return {"status": "ok", "service": "PrivacyLens API"}


@app.get("/health")
def health():
    return {"status": "healthy"}


async def _run_category(category: dict, index: EmbeddingIndex) -> dict:
    try:
        retrieved = await index.search(category["query"], k=6)
        return await analyze_category(category, retrieved)
    except Exception as e:
        logger.exception(f"Category analysis failed: {category['name']}")
        return {
            "risk_category": category["name"],
            "risk_score": 5,
            "summary": "Analysis failed for this category; treated as indeterminate.",
            "explanation": f"Error during reasoning: {e}",
            "key_findings": [],
            "red_flags": [],
            "positive_indicators": [],
            "evidence": [],
            "evidence_chunks": [],
        }


@app.post("/api/v1/analyze")
async def analyze(req: AnalyzeRequest):
    # 1. Retrieval / ingestion ------------------------------------------------
    try:
        if req.policy_text and req.policy_text.strip():
            sections = fetch_and_parse(text_override=req.policy_text)
        else:
            if not req.url or not req.url.strip():
                raise HTTPException(status_code=400, detail="Provide a url or policy_text.")
            sections = fetch_and_parse(url=req.url)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Fetch/parse failed")
        raise HTTPException(status_code=422, detail=f"Could not fetch or parse the policy: {e}")

    if not sections:
        raise HTTPException(status_code=422, detail="No readable content found at that URL.")

    # 2. Chunking ---------------------------------------------------------------
    chunks = chunk_sections(sections)
    if not chunks:
        raise HTTPException(status_code=422, detail="Document parsed but produced no usable chunks.")

    # 3. Embedding + FAISS index -------------------------------------------------
    try:
        index = await EmbeddingIndex.create(chunks)
    except Exception as e:
        logger.exception("Embedding/index build failed")
        raise HTTPException(status_code=502, detail=f"Embedding service error: {e}")

    # 4. Retrieval + LLM reasoning per risk category — run concurrently ----------
    findings = await asyncio.gather(*(_run_category(c, index) for c in RISK_CATEGORIES))
    findings = list(findings)

    # 5. Deterministic overall scoring -------------------------------------------
    overall = compute_overall(findings)

    executive_summary = build_executive_summary(overall, findings)
    executive_summary_parts = build_executive_summary_parts(overall, findings)

    return {
        "url": req.url if req.url else "pasted-text",
        "overall": overall,
        "executive_summary": executive_summary,
        "executive_summary_parts": executive_summary_parts,
        "total_chunks": len(chunks),
        "total_sections": len(sections),
        "findings": findings,
    }


def build_executive_summary(overall: dict, findings: list) -> str:
    high = [f["risk_category"] for f in findings if f["risk_score"] >= 7]
    low = [f["risk_category"] for f in findings if f["risk_score"] <= 3]
    parts = [f"Overall privacy risk is rated {overall['score']}/10 ({overall['label']})."]
    if high:
        parts.append(f"Highest-risk areas: {', '.join(high)}.")
    if low:
        parts.append(f"Stronger areas: {', '.join(low)}.")
    parts.append(
        "This assessment is generated from retrieved policy text and heuristic scoring; "
        "it is not legal advice."
    )
    return " ".join(parts)


def build_executive_summary_parts(overall: dict, findings: list) -> dict:
    high = [f["risk_category"] for f in findings if f["risk_score"] >= 7]
    low = [f["risk_category"] for f in findings if f["risk_score"] <= 3]
    return {
        "headline": f"Overall privacy risk is rated {overall['score']}/10 ({overall['label']}).",
        "highest_risk_areas": high,
        "stronger_areas": low,
        "disclaimer": (
            "This assessment is generated from retrieved policy text and heuristic scoring; "
            "it is not legal advice."
        ),
    }
