FROM python:3.11-slim

WORKDIR /code

# System deps for faiss / lxml
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app

# Render sets $PORT at runtime; default to 7860 for local/HF Spaces use
EXPOSE 7860
ENV PORT=7860

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
