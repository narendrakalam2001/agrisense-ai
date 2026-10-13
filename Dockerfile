# AgriSense AI — FastAPI backend image (Render / any Docker host)
#
#   docker build -t agrisense-ai-api .
#   docker run -p 8000:8000 --env-file .env agrisense-ai-api
#
# The trained model (models/*.h5) is baked into the image, so it must be
# committed to the repo (see scripts/slim_model.py to keep it small).

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

COPY src/ ./src/
COPY services/ ./services/
COPY serving/ ./serving/
COPY scripts/ ./scripts/
COPY data/knowledge_base/ ./data/knowledge_base/
COPY models/ ./models/

# Build the ChromaDB store at BUILD time: downloads the embedding model once and
# makes container start-up fast and independent of Hugging Face availability.
RUN python scripts/ingest_knowledge_base.py

EXPOSE 8000

# Render injects $PORT (default 10000); locally it falls back to 8000.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import os, urllib.request as u; u.urlopen('http://localhost:%s/health' % os.environ.get('PORT', '8000'))" || exit 1

CMD ["sh", "-c", "uvicorn serving.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
