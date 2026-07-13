# PrivacyLens Backend

## Deploy on Render (free tier)

1. Push this folder to a GitHub repo.
2. Go to https://dashboard.render.com → **New** → **Blueprint** → connect the repo (it will detect `render.yaml` automatically).
   - Or: **New** → **Web Service** → connect repo → Runtime: **Docker** → Plan: **Free**.
3. When prompted, set the `NVIDIA_API_KEY` secret (get it from build.nvidia.com).
4. Deploy. Render builds the Dockerfile and gives you a URL like `https://privacylens-backend.onrender.com`.
5. Update `BACKEND_URL` in your frontend `index.html` to that URL.

Note: Render's free web services spin down after ~15 min of inactivity and take ~30-50s to
wake on the next request — the frontend's pipeline-status UI will just show "Fetching..." a
bit longer on a cold start. Fine for a demo/portfolio project.

FastAPI + LangChain-style RAG backend for privacy policy risk analysis.
Uses NVIDIA NIM (`NVIDIA_API_KEY`) for embeddings and LLM reasoning, FAISS for
vector search, and BeautifulSoup for retrieval/parsing.

## Required Space secret

Set this in **Settings → Repository secrets**:

- `NVIDIA_API_KEY` — your NVIDIA NIM API key from build.nvidia.com

## Optional variables

- `NVIDIA_EMBED_MODEL` (default: `nvidia/nv-embedqa-e5-v5`)
- `NVIDIA_CHAT_MODEL` (default: `meta/llama-3.1-70b-instruct`)
- `NVIDIA_BASE_URL` (default: `https://integrate.api.nvidia.com/v1`)

## Endpoint

`POST /api/v1/analyze`

```json
{ "url": "https://example.com/privacy" }
```
or
```json
{ "url": "pasted-text", "policy_text": "..." }
```
