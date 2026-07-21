ARG APP_BUILD_ID=development

# Stage 1: build React frontend
FROM node:20-alpine AS frontend
ARG APP_BUILD_ID
ENV VITE_APP_BUILD_ID=${APP_BUILD_ID}

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Stage 2: Python runtime
FROM python:3.11-slim AS runtime
ARG APP_BUILD_ID
ENV APP_BUILD_ID=${APP_BUILD_ID}

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

# Smoke-test corpus for standalone image runs. Production Compose disables
# bootstrap because private S3 is the durable corpus source.
# Runtime indexes/blobs come from the Compose bind-mount (./data) or an empty volume.
COPY corpus/ ./corpus/

RUN mkdir -p /app/data \
    && chown imagecb:imagecb /app/data

ENV TESSERACT_CMD=/usr/bin/tesseract
ENV BOOTSTRAP_CORPUS_DIR=/app/corpus/smoke

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail http://127.0.0.1:8080/api/health || exit 1

USER imagecb

CMD ["python", "-m", "imagecb.cli", "serve-web", "--host", "0.0.0.0", "--port", "8080"]
