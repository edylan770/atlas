# Stage 1: build React frontend
FROM node:20-alpine AS frontend

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Stage 2: Python runtime
FROM python:3.11-slim AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY imagecb/ ./imagecb/

COPY --from=frontend /app/frontend/dist ./imagecb/web/frontend_dist/

ENV TESSERACT_CMD=/usr/bin/tesseract

EXPOSE 8080

CMD ["python", "-m", "imagecb.cli", "serve-web", "--host", "0.0.0.0", "--port", "8080"]
