# PrivacyLens

**Evidence-grounded privacy policy analysis.**

PrivacyLens ingests a privacy policy, retrieves clauses relevant to seven privacy-risk categories, asks an LLM to reason only over retrieved evidence, and combines category findings into a deterministic overall report.

> PrivacyLens is an engineering/demo project, not legal advice.

## Architecture

```text
Policy URL / pasted text
        |
        v
Fetch + HTML/plaintext parsing
        |
        v
Section-aware chunking
        |
        v
Configurable embedding provider
        |
        v
FAISS cosine-similarity index
        |
        v
Hybrid dense + lexical retrieval
        |
        v
Evidence retrieval per risk category
        |
        v
NVIDIA NIM LLM reasoning
        |
        v
Deterministic weighted scoring
        |
        v
Evidence-grounded report
```

The current implementation uses NVIDIA NIM for embeddings and chat reasoning. Retrieval is explicitly two-stage: FAISS produces dense candidates, then a deterministic reranker combines dense similarity with lexical overlap. All seven privacy categories are then analyzed in one structured LLM request to avoid seven independent hosted inference calls. The default embedding model is `nvidia/nemotron-3-embed-1b`.

## Why this project

Privacy policies are long, inconsistent, and difficult to compare quickly. A useful analyzer should do more than generate a generic summary: it should surface relevant clauses, preserve section context, expose supporting evidence, and make its scoring logic inspectable.

## Risk categories

- Data Collection
- Third-Party Sharing
- Retention
- Deletion Rights
- Tracking / Cookies
- Transparency
- Consent Mechanisms

## API

`POST /api/v1/analyze`

URL input:

```json
{ "url": "https://example.com/privacy" }
```

Pasted policy input:

```json
{
  "url": "pasted-text",
  "policy_text": "..."
}
```

## Configuration

Required:

- `NVIDIA_API_KEY`

Optional:

- `NVIDIA_BASE_URL` — default: `https://integrate.api.nvidia.com/v1`
- `EMBEDDING_PROVIDER` — default: `nvidia`
- `EMBEDDING_MODEL` — default: `nvidia/nemotron-3-embed-1b`
- `EMBEDDING_BATCH_SIZE` — default: `32`
- `EMBEDDING_CONCURRENCY` — default: `4`
- `NVIDIA_CHAT_MODEL` — default: `nvidia/nemotron-3.5-lightning-30b-a3b`
- `REASONING_TIMEOUT_SECONDS` — default: `60`
- `REASONING_EVIDENCE_CHUNKS` — default: `4`
- `REASONING_CHUNK_CHARS` — default: `900`
- `RETRIEVAL_DENSE_WEIGHT` — default: `0.75`
- `RETRIEVAL_LEXICAL_WEIGHT` — default: `0.25`
- `ALLOWED_ORIGINS` — default: `*` for the public demo; set explicit origins in production

The provider/model are deliberately configuration-driven so a model retirement does not require changing application code. The chat model uses NVIDIA's current `openai/gpt-oss-20b` Free Endpoint.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export NVIDIA_API_KEY="..."
uvicorn app.main:app --reload
```

## Deployment

The repository includes a Dockerfile and Render Blueprint configuration. Set `NVIDIA_API_KEY` in Render and deploy the web service.

## Upgrade status

Completed on `upgrade/portfolio-foundation`:

1. Current NVIDIA embedding model migration
2. Provider/configuration boundary
3. Structured findings with confidence/disclosure status
4. Authoritative evidence chunk citations
5. Two-stage hybrid retrieval + reranking
6. Evaluation metric harness
7. Automated tests and GitHub Actions CI
8. Configurable CORS
9. Single-request batched LLM reasoning with bounded timeout and timing logs

Next major milestones:

1. Curated human-reviewed evaluation dataset
2. Citation-accuracy and severity-accuracy benchmark
3. Policy version comparison and change detection
4. Exportable/shareable reports

## Project status

The foundation and retrieval-quality phases are implemented on `upgrade/portfolio-foundation`. The branch is intentionally kept as a draft PR while CI and the next evaluation/benchmark phase are completed.
