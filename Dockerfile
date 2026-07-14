# Stage 1: build React frontend
FROM node:20-alpine AS frontend

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Stage 2: make the current Windows corpus portable for Linux.
FROM python:3.11-slim AS corpus

WORKDIR /seed

COPY scripts/prepare_docker_corpus.py ./
COPY data/ ./data/
RUN python prepare_docker_corpus.py /seed/data --target-data-dir /app/data

# Stage 3: Python runtime
FROM python:3.11-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 imagecb \
    && useradd --uid 10001 --gid imagecb --create-home imagecb

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY imagecb/ ./imagecb/

COPY --from=frontend /app/frontend/dist ./imagecb/web/frontend_dist/
COPY --from=corpus --chown=imagecb:imagecb /seed/data/ ./data/

# Smoke-test corpus: a few small images auto-ingested on boot when the index
# is empty (see _lifespan in imagecb/api/server.py). Safe to remove once a real
# S3-backed corpus pipeline is in place.
COPY corpus/ ./corpus/

ENV TESSERACT_CMD=/usr/bin/tesseract
ENV BOOTSTRAP_CORPUS_DIR=/app/corpus/smoke

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail http://127.0.0.1:8080/api/health || exit 1

USER imagecb

CMD ["python", "-m", "imagecb.cli", "serve-web", "--host", "0.0.0.0", "--port", "8080"]
