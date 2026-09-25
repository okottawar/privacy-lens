"""
PrivacyLens backend — RAG pipeline for privacy policy risk analysis.
Fetch -> Parse/Clean -> Chunk -> Embed (NVIDIA NIM) -> FAISS -> Retrieve -> LLM (NVIDIA NIM) -> Score -> Report
"""
import asyncio
import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.retrieval import fetch_and_parse, chunk_sections
from app.config import get_settings
from app.embeddings import EmbeddingIndex
from app.embedding_provider import EmbeddingModelUnavailableError
from app.reasoning import analyze_category, RISK_CATEGORIES
from app.scoring import compute_overall

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("privacylens")
_reasoning_semaphore = asyncio.Semaphore(get_settings().reasoning_concurrency)

app = FastAPI(title="PrivacyLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().allowed_origins,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
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
    started = time.perf_counter()
    try:
        async with _reasoning_semaphore:
            retrieved = await index.search(category["query"], k=6)
            result = await analyze_category(category, retrieved)
        logger.info(
            "category.complete category=%s duration_ms=%d status=%s",
            category["name"],
            int((time.perf_counter() - started) * 1000),
            result.get("disclosure_status", "unknown"),
        )
        return result
    except Exception as e:
        logger.exception(f"Category analysis failed: {category['name']}")
        logger.exception("category.failed category=%s", category["name"])
        return {
            "risk_category": category["name"],
            "risk_score": 5,
            "confidence": 0.0,
            "disclosure_status": "unclear",
            "summary": "Analysis failed for this category; treated as indeterminate.",
            "explanation": f"Error during reasoning: {e}",
            "key_findings": [],
            "red_flags": [],
            "positive_indicators": [],
            "evidence": [],
            "evidence_chunk_ids": [],
            "evidence_chunks": [],
        }


@app.post("/api/v1/analyze")
async def analyze(req: AnalyzeRequest):
    request_started = time.perf_counter()
    logger.info("analysis.start url=%s", req.url)
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

    logger.info("analysis.fetch_parse_complete duration_ms=%d sections=%d", int((time.perf_counter() - request_started) * 1000), len(sections))

    # 2. Chunking ---------------------------------------------------------------
    chunks = chunk_sections(sections)
    logger.info("analysis.chunk_complete duration_ms=%d chunks=%d", int((time.perf_counter() - request_started) * 1000), len(chunks))
    if not chunks:
        raise HTTPException(status_code=422, detail="Document parsed but produced no usable chunks.")

    # 3. Embedding + FAISS index -------------------------------------------------
    try:
        index = await EmbeddingIndex.create(chunks)
        logger.info("analysis.embedding_complete duration_ms=%d", int((time.perf_counter() - request_started) * 1000))
    except EmbeddingModelUnavailableError as e:
        logger.exception("Configured embedding model is unavailable")
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        logger.exception("Embedding/index build failed")
        raise HTTPException(status_code=502, detail=f"Embedding service error: {e}") from e

    # 4. Retrieval + LLM reasoning per risk category with bounded concurrency ---
    findings = await asyncio.gather(*(_run_category(c, index) for c in RISK_CATEGORIES))
    logger.info("analysis.reasoning_complete duration_ms=%d", int((time.perf_counter() - request_started) * 1000))
    findings = list(findings)

    # 5. Deterministic overall scoring -------------------------------------------
    overall = compute_overall(findings)

    executive_summary = build_executive_summary(overall, findings)
    executive_summary_parts = build_executive_summary_parts(overall, findings)
    logger.info("analysis.complete duration_ms=%d score=%s", int((time.perf_counter() - request_started) * 1000), overall.get("score"))

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
