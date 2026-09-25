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

The current implementation uses NVIDIA NIM for both embeddings and chat reasoning, with the embedding model configured through `EMBEDDING_MODEL`. Retrieval combines dense similarity with a lightweight lexical overlap signal so exact policy terms remain discoverable. The default is `nvidia/llama-3.2-nv-embedqa-1b-v2`.

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
- `EMBEDDING_MODEL` — default: `nvidia/llama-3.2-nv-embedqa-1b-v2`
- `EMBEDDING_BATCH_SIZE` — default: `32`
- `EMBEDDING_CONCURRENCY` — default: `4`
- `NVIDIA_CHAT_MODEL` — default: `meta/llama-3.1-70b-instruct`

The provider/model are deliberately configuration-driven so a model retirement does not require changing application code.

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

## Roadmap

The next portfolio-focused milestones are:

1. Evidence citations and structured findings
2. Hybrid retrieval and reranking
3. Evaluation dataset and retrieval/analysis metrics
4. Automated tests and CI
5. Policy version comparison and change detection
6. Exportable, shareable reports

## Project status

This branch is the first step of the portfolio upgrade: it removes the retired NVIDIA E5 v5 dependency, centralizes configuration, and introduces an explicit embedding-provider boundary.
