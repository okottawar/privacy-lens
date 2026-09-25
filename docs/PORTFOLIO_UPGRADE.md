# PrivacyLens Portfolio Upgrade Plan

## Phase 1 — Foundation
- Replace retired NVIDIA E5 v5 embedding configuration.
- Centralize environment-driven model/provider settings.
- Normalize provider errors.
- Add focused unit tests and CI.
- Remove stale technology claims from the UI.

## Phase 2 — Trustworthy analysis
- [x] Introduce structured finding schemas.
- [x] Attach findings to retrieved section/chunk evidence.
- [x] Separate severity, confidence, and disclosure status.
- [x] Improve invalid/partial model-response handling.

## Phase 3 — Retrieval quality
- [x] Add lexical retrieval alongside dense retrieval.
- [x] Add an explicit second-stage reranking boundary.
- [x] Preserve document hierarchy and evidence metadata.
- [ ] Measure retrieval recall on a human-reviewed benchmark.

## Phase 4 — Transparent scoring
- Make category signals explicit and reproducible.
- Keep deterministic aggregation separate from model interpretation.
- Document scoring assumptions and uncertainty.

## Phase 5 — Evaluation
- [x] Add reusable Recall@K, Precision@K, and Reciprocal Rank metrics.
- [ ] Build a manually reviewed privacy-policy benchmark.
- [ ] Track retrieval hit rate, evidence relevance, citation accuracy,
  category/severity accuracy, JSON validity, latency, and failure rate.
- [ ] Compare baseline vs improved retrieval strategies.

## Phase 6 — Production hardening
- Tighten CORS.
- Add request IDs and structured logs.
- Add health/readiness endpoints.
- Add API schema validation and clearer failure responses.
- Improve dependency/version management.

## Phase 7 — Product experience
- Evidence-first report UI.
- Explain-why interactions.
- Policy comparison and version diffing.
- Export/shareable reports.
- Optional batch analysis.

## Current implementation status
Phases 1–3 are implemented on this branch. The remaining work is primarily
benchmarking, production hardening, and product-level policy comparison.

## Definition of done
PrivacyLens should demonstrate an auditable pipeline from policy ingestion to
retrieval, evidence-backed findings, transparent scoring, evaluation results,
automated tests, and a polished deployed report experience.
