# PrivacyLens

PrivacyLens analyzes a privacy policy using a simple evidence-grounded RAG pipeline.

## Architecture

```text
Policy URL / pasted text
        |
        v
Fetch + Parse + Clean
        |
        v
Section-aware Chunking
        |
        v
NVIDIA Embeddings
        |
        v
FAISS Vector Index
        |
        v
Retrieve top evidence per category
        |
        v
NVIDIA NIM reasoning
        |
        v
Deterministic risk scoring
        |
        v
Risk Analysis Report
```

The backend intentionally keeps this pipeline simple: one document is indexed once, each privacy category retrieves its own evidence, and the LLM reasons over that evidence. No batching layer, provider abstraction, reranker, or background job is required for the core flow.

## Risk categories

- Data Collection
- Third-Party Sharing
- Retention
- Deletion Rights
- Tracking / Cookies
- Transparency
- Consent Mechanisms

## Configuration

Set:

- `NVIDIA_API_KEY`
- `NVIDIA_EMBED_MODEL` — default: `nvidia/nemotron-3-embed-1b`
- `NVIDIA_CHAT_MODEL` — default: `nvidia/nemotron-3.5-lightning-30b-a3b`
- `NVIDIA_BASE_URL` — default: `https://integrate.api.nvidia.com/v1`

The embedding model uses NVIDIA's query/passage modes, as required by the NeMo Retriever API. citeturn686749search0turn686749search2

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export NVIDIA_API_KEY="..."
uvicorn app.main:app --reload
```

## Deployment

Render uses the included `render.yaml`. The frontend is a separate static file and is intentionally not modified by this reset.

## Status

This is the clean baseline for the next portfolio iteration. Keep the core request path stable before adding further optimization or evaluation layers.
