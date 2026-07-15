# Imagecb

Conversational multimodal retrieval over an image corpus that mixes
standalone image files with images embedded inside PowerPoint (`.pptx`)
and PDF documents.

Every image is:

- Embedded into a joint vision-language space with **Amazon Titan
  Multimodal Embeddings** on Bedrock (`amazon.titan-embed-image-v1`).
- Captioned by a **VLM** into a structured JSON document. Default is
  **AWS Bedrock** Claude Haiku 4.5
  (`us.anthropic.claude-haiku-4-5-20251001-v1:0`); OpenAI and Anthropic
  are also supported as drop-in providers.
- Indexed in **ChromaDB** (dense image vectors plus a caption-text
  vector lane), **rank-bm25** (sparse over caption + OCR + slide
  context), and **SQLite** (provenance + filters).

User turns are parsed by an LLM into a structured `QuerySpec`
(semantic phrase, time window, source/author filters, must-have /
must-avoid keywords, refinement flag). The system fuses three lanes —
visual dense, caption-text dense, and sparse BM25 — with Reciprocal
Rank Fusion, reranks with **Cohere Rerank 3.5**
on Bedrock (`cohere.rerank-v3-5:0`), and renders results with
full provenance (e.g. `Slide 7 of Q3_Review.pptx, modified 2026-05-08`).

## Architecture at a glance

```
ingest:  files -> extractor (pptx/pdf/image) -> OCR + VLM caption
                                              -> Titan image emb + caption-text emb
                                              -> SQLite + Chroma + BM25

query:   text + history -> LLM QuerySpec
                        -> metadata filter
                        -> visual dense + caption-text dense + sparse (BM25)
                        -> RRF fusion
                        -> Cohere Rerank 3.5 rerank
                        -> ranked images + provenance
```

**Match % on result cards** is a calibrated display value derived from
each hit's score. Chat search uses Cohere rerank relevance; similar-image
search uses normalized RRF fusion across the visual and text lanes (rank,
filter, and display all use the same fused score). The min-match % slider
uses this same calibrated scale (not raw rerank scores). **100%** appears
for near-excellent rerank scores (≥ 0.93), dense cosine (≥ 0.92), or fusion
scores at 1.0.

## Setup

### 1. Python deps

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

No local ML model downloads are required — embeddings, captioning,
query parsing, and reranking all run through AWS Bedrock APIs.

### 2. Tesseract OCR (Windows)

1. Install Tesseract from
   https://github.com/UB-Mannheim/tesseract/wiki (the UB-Mannheim
   build is the standard Windows installer).
2. Confirm the install path, typically
   `C:\Program Files\Tesseract-OCR\tesseract.exe`.
3. Set `TESSERACT_CMD` in your `.env` to that path.

If you don't want OCR right now, leave `TESSERACT_CMD` empty - ingest
will continue and OCR text will simply be blank.

### 3. AWS Bedrock (default)

The default config drives **all** model calls through AWS Bedrock:

| Role | Model | API |
|------|-------|-----|
| Image + text embeddings | `amazon.titan-embed-image-v1` | `invoke_model` |
| Caption-text embeddings | `amazon.titan-embed-text-v2:0` | `invoke_model` |
| VLM captioning | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | `converse` |
| Query parsing | same | `converse` |
| Reranking | `cohere.rerank-v3-5:0` | `invoke_model` |

1. In the AWS Bedrock console for `us-east-1`, enable model access for
   all five models listed above.
2. Copy `.env.example` to `.env`:

   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

3. Provide credentials in one of two ways:

   - **Bedrock API key (easiest):** set `AWS_BEARER_TOKEN_BEDROCK=...`
     in your `.env`. boto3 picks it up automatically for all
     `bedrock-runtime` calls (embeddings, captioning, query parsing, and
     reranking). These tokens are short-lived; refresh from the Bedrock
     console when they expire.
   - **Standard AWS credentials:** any of the usual sources works
     (`aws configure`, `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`,
     SSO, instance profile, etc.). Make sure your principal has
     `bedrock:InvokeModel`, `bedrock:Converse`, and rerank permissions
     for the model IDs you set.

4. Leave `AWS_REGION=us-east-1` unless you specifically want a
   different Bedrock region (you'll then also need a matching inference
   profile prefix, e.g. `eu.anthropic...` for EU regions).

Ingest makes **three Bedrock calls per image** (Titan image embed +
Claude caption + caption-text embed). Cost scales per image and per token. For a fast dry-run that
skips captioning, see `--skip-caption` under [Usage](#usage) (the Titan
embed call still runs — search requires it).

### 4. Optional: OpenAI or Anthropic providers

To bypass Bedrock and use a different cloud VLM, set in `.env`:

```
VLM_PROVIDER=openai      # or anthropic
LLM_PROVIDER=openai
VLM_MODEL=gpt-4o-mini    # or e.g. claude-3-5-sonnet-latest
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...    # or ANTHROPIC_API_KEY=...
```

The rest of the system is provider-agnostic; the only files that touch
the provider are
[imagecb/models/vlm.py](imagecb/models/vlm.py) and
[imagecb/models/llm.py](imagecb/models/llm.py).

## Docker

Run Imagecb in a container with the React UI, Tesseract OCR, and a snapshot
of the repository's current indexed corpus. No local Python or Node install
is required.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or
  Docker Engine + Compose v2)
- A `.env` file with Bedrock or other provider credentials (same as
  [Setup](#setup) — copy from `.env.example`)

### Start the web UI

```powershell
docker compose up --build
```

Open http://localhost:8080. The image builds `frontend/dist/`, converts
the corpus's Windows paths to container paths, and serves the UI via FastAPI.

The live search state under `/app/data` uses the `imagecb_data` named volume,
so SQLite, Chroma, BM25, hubness, and deck caches survive container
replacement. On EC2, place Docker's data root on EBS or replace the named
volume with an EBS bind mount if the state must also survive instance
replacement.

### Reset the local Docker search state

```powershell
docker compose down
docker volume rm atlas-image-search_imagecb_data
docker compose up --build
```

This deletes the local Docker search brain. It does not delete private S3
objects.

### Private S3 corpus storage on EC2

S3 stores original uploads and generated display PNGs. The search indexes
remain on the persistent `/app/data` volume because SQLite and Chroma must
not run directly against S3.

Configure the container:

```dotenv
BLOB_STORAGE_BACKEND=s3
S3_BUCKET=your-private-corpus-bucket
S3_PREFIX=imagecb
S3_REGION=us-east-1
BOOTSTRAP_CORPUS_DIR=
```

Attach an EC2 instance role granting `s3:GetObject` and `s3:PutObject` for
`arn:aws:s3:::your-private-corpus-bucket/imagecb/*`, plus
`s3:ListBucket` scoped to the `imagecb/*` prefix. Boto3 uses that role
automatically; do not add static AWS keys to `.env`.

Existing local blobs can be migrated safely:

```powershell
# Preview only
python -m imagecb.cli migrate-blobs-to-s3

# Upload objects and rewrite SQLite rows after each successful upload
python -m imagecb.cli migrate-blobs-to-s3 --apply
```

The command preserves local files for rollback. Legacy local paths remain
readable during cutover. To roll back new writes, set
`BLOB_STORAGE_BACKEND=local`; records already rewritten to `s3://` continue
to require S3 or must be restored from their retained local copies.

### Other CLI commands

Use the same pattern as ingest:

```powershell
docker compose run --rm imagecb python -m imagecb.cli status
docker compose run --rm imagecb python -m imagecb.cli repair-index
docker compose run --rm imagecb python -m imagecb.cli repair-captions --workers 4
```

### Troubleshooting

| Issue | What to try |
| ----- | ----------- |
| Cannot connect to http://localhost:8080 | Ensure the container is running; the server binds `0.0.0.0` inside the image |
| Bedrock / AWS errors | Refresh `AWS_BEARER_TOKEN_BEDROCK` or check `AWS_REGION` in `.env` |
| `ThrottlingException` during ingest | Lower `INGEST_WORKERS` in `.env` |
| Missing `.env` | `Copy-Item .env.example .env` and fill in credentials before `docker compose up` |

## Usage

### After upgrading: reset Chroma

Embedding dimensions changed (768-dim OpenCLIP → 1024-dim Titan).
Existing Chroma vectors are incompatible and must be rebuilt:

```powershell
Remove-Item -Recurse -Force .\data\chroma
python -m imagecb.cli ingest "C:\path\to\corpus"
```

SQLite metadata and BM25 can stay; only the Chroma directory needs
deleting.

The caption-text dense lane needs no manual migration: the next ingest
(or `python -m imagecb.cli repair-index`) backfills one text vector per
already-indexed image from its stored caption.

### Ingest a corpus

```powershell
python -m imagecb.cli ingest "C:\path\to\corpus"
```

The corpus path can be a single file or a directory; the directory is
walked recursively for `.pptx`, `.pdf`, and common image extensions
(`.png .jpg .jpeg .webp .bmp .gif .tif .tiff`). Re-running ingest is
idempotent: images are de-duplicated by content hash.

To do a fast dry-run without spending VLM tokens:

```powershell
python -m imagecb.cli ingest "C:\path\to\corpus" --skip-caption
```

(Search quality will be much worse without captions; useful only for
debugging the extraction stage.)

### Faster ingest

Ingest runs **two Bedrock calls per image** (caption + embed). By default
images are processed in parallel:

```powershell
python -m imagecb.cli ingest "C:\path\to\corpus" --force --workers 2 --batch-size 25
```

For ~50 images this typically lands around **5–10 minutes** instead of
~20+ minutes sequential, depending on Bedrock latency and rate limits.

| Flag | Effect |
|------|--------|
| `--workers 2` | Parallel images (default `4`; use `2` for large re-ingest) |
| `--batch-size 25` | Process files in batches of 25; rebuild BM25 once at end |
| `--no-defer-bm25` | Rebuild BM25 after every batch (slower for large corpora) |
| `--skip-ocr` | Skip Tesseract (faster; OCR text empty) |
| `--max-image-side 1024` | Smaller payload to the VLM (default) |
| `--force` | Re-process duplicates (refresh cache, captions, vectors) |

**Large corpus / re-ingest:** If ingest stalls after a handful of images,
lower concurrency and batch files:

```powershell
python -m imagecb.cli ingest "C:\path\to\corpus" --batch-size 25 --workers 2 --force
```

Set in `.env` for all ingest paths (CLI and API):

```
INGEST_WORKERS=2
BEDROCK_MAX_CONCURRENT=2
INGEST_BATCH_SIZE=25
```

Bedrock calls are gated by `BEDROCK_MAX_CONCURRENT` (default `2`) with
timeouts and adaptive retries so hung workers do not freeze the whole run.

If you see `ThrottlingException`, lower `--workers` to `2`. Refresh
`AWS_BEARER_TOKEN_BEDROCK` for your `AWS_REGION` before `--force` if
`status` still reports failed captions.

### Context-aware embeddings (after upgrade)

Ingest now embeds slide/PDF title and notes together with each image
(Titan `inputText` + `inputImage`). For an existing index, re-embed once
without re-extracting decks:

```powershell
python -m imagecb.cli reindex-embeddings --workers 4
```

### Index health and targeted repair

After every ingest (CLI, API, or Gradio), the index is automatically
checked for missing cached PNGs, failed captions, and missing Chroma
vectors. When issues exist, only affected source files are re-ingested —
not the whole corpus. Disable with `POST_INGEST_REPAIR_ENABLED=false`
in `.env`.

Check index health:

```powershell
python -m imagecb.cli status
python -m imagecb.cli status --verbose
```

Run repair manually:

```powershell
python -m imagecb.cli repair-index
python -m imagecb.cli repair-index --dry-run
python -m imagecb.cli repair-index --include-weak --workers 4
```

Rows where both the cached PNG and original source file are missing are
reported as **unrecoverable** (SQLite rows are not deleted automatically).
Parallel ingests are not supported.

### Repair failed captions

If `status` still reports failed captions after automatic repair:

```powershell
python -m imagecb.cli repair-captions --workers 4
```

This re-runs the VLM only on failed rows and rebuilds the BM25 index.
Requires cached PNGs; use `repair-index` if PNGs are missing.

### Launch the web UI (recommended)

**IT / no npm:** one Python command serves the **ATLAS** chat UI, admin
dashboard, and API on port **8080**. The repo ships a pre-built React
bundle at `imagecb/web/frontend_dist/` — you do not run `npm` locally.

```powershell
python -m imagecb.cli serve-web
```

Startup should print `UI bundle: ATLAS (React, imagecb/web/frontend_dist)`.
Then open:

- Chat: http://127.0.0.1:8080/ (hard refresh once if you still see the old UI: Ctrl+Shift+R)
- Deck suggest: http://127.0.0.1:8080/deck (also via the slide icon beside **Send** in the composer)
- Admin: http://127.0.0.1:8080/admin (enter `ADMIN_API_KEY` from `.env`)

If the page still shows the legacy **Imagecb** (light gray) UI, your clone’s
`frontend_dist` is out of date — `git pull` the latest branch that includes
an updated bundle (maintainers rebuild it; see below). Do not rely on local
`npm run build` if your machine cannot run npm.

If `frontend_dist` is missing entirely, `serve-web` falls back to the built-in
vanilla UI under `imagecb/web/static/` (chat only, no admin).

The Docker image builds the UI at image build time and does not require npm on
the host either.

**Frontend developers** (machines allowed to run npm) after changing
`frontend/`:

```powershell
cd frontend
npm ci
npm run build
cd ..
python scripts/sync_frontend_dist.py
```

CI verifies `frontend_dist` stays in sync when `frontend/` changes.

#### Upload from the UI

Click **Add to corpus**, choose one or more supported files (images,
`.pdf`, or `.pptx`), and ingest. In local mode files are retained under
`data/uploads/`. In S3 mode they are temporarily staged, stored under the
configured private `uploads/` prefix, and removed from local staging after
ingest. Generated PNGs are stored under the S3 `images/` prefix while the
search index remains local. Optional checkboxes mirror CLI flags: skip
captions, skip OCR, and force re-ingest. Large decks can take several
minutes (~2 Bedrock calls per extracted image).

#### Deck suggest (slide-aware image suggestions)

Open **Deck suggest** from the header (or http://127.0.0.1:8080/deck).
Upload a content-filled `.pptx` deck. The system:

1. Extracts each slide's title, body text, and speaker notes.
2. Batches slides through the same LLM as chat (`LLM_MODEL`) to produce
   grounded, caption-style search descriptions (or `no_image_needed` for
   tables, agendas, and text-only slides).
3. Runs each description through the normal hybrid search + Cohere rerank
   pipeline against the indexed corpus.
4. Shows a per-slide gallery with the generated description visible for
   debugging, plus **Accept** / **Dismiss** and **Force image** when the
   model skipped a slide.

Processed decks are cached on disk by content hash (`data/deck_cache/` by
default). Re-uploading an unchanged or lightly edited deck reuses per-slide
LLM and search results where slide text matches.

| Env var | Default | Purpose |
|---------|---------|---------|
| `DECK_CACHE_DIR` | `{DATA_DIR}/deck_cache` | Disk cache for LLM + search |
| `DECK_LLM_BATCH_SIZE` | `10` | Slides per LLM request |
| `DECK_MAX_SLIDES` | `200` | Maximum slides per upload |
| `DECK_MAX_CHARS_PER_SLIDE` | `6000` | Truncate text sent to the LLM |
| `DECK_MAX_UPLOAD_BYTES` | `52428800` (50 MB) | Max upload size |
| `DECK_CACHE_ENABLED` | `true` | Toggle disk cache |

API: `POST /api/deck/suggest` (multipart `.pptx`), `POST /api/deck/force` (JSON).

#### Search features in the web UI

- **Find similar** on any result card runs visual similarity search (no
  query LLM). Axis buttons (balanced / subject / style / layout) change both
  the text query focus and the visual-vs-text fusion weight.
- **Camera** in the composer uploads a reference image for similarity
  search.
- **Open source** downloads the original `.pptx`, `.pdf`, or image file;
  **Copy path** puts the on-disk path on the clipboard (use with slide/page
  chips for location).
- **How we interpreted your query** shows interpretation notes (refinement
  pool, carried-forward filters, must-have / exclude keywords).

### Legacy Gradio UI

```powershell
python -m imagecb.cli serve
```

Then open http://127.0.0.1:7860. This interface is deprecated in favor
of `serve-web`.

Try queries like:

- "Diagrams of system architecture from internal decks"
- "Screenshots of dashboards modified after May 2026"
- "Only the ones from Q3_Review.pptx"  (refinement)
- "Drop anything with bar charts"      (refinement, must-avoid)

### Inspect the index

```powershell
python -m imagecb.cli status
python -m imagecb.cli parse-query "screenshots of dashboards from last quarter"
```

### Search evaluation harness

Offline tool for measuring whether retrieval tuning actually helps. Not a
user-facing feature — run it locally against your ingested `./data` index
with Bedrock credentials.

1. **Label cases** in [`eval/golden.json`](eval/golden.json). Each text case
   maps a query to `relevant_ids`; each similar case maps a reference
   `image_id` to expected neighbors. Set `"template": false` once IDs are
   real. Starter entries are templates only.

2. **Discover image IDs** to label:

```powershell
python -m imagecb.cli eval-suggest "operational dashboards" --top-k 15
python -m imagecb.cli eval-suggest "operational dashboards" --mode chat
python -m imagecb.cli eval-suggest --similar <image_id> --axis style
```

3. **Run eval** (reports separate scoreboards for chat, retrieval-only, and
   similar search):

```powershell
python -m imagecb.cli eval-search
python -m imagecb.cli eval-search --mode retrieval --k 1,5,10
python -m imagecb.cli eval-search --case-id dashboard-screenshots --failures-only
python -m imagecb.cli eval-search --json-out runs/baseline.json
```

While drafting cases, `--skip-id-validation` skips the SQLite ID check.
Compare before/after tuning runs via `--json-out`.

## Configuration

Everything is driven by environment variables - see `.env.example` for
the full list. Highlights:

| Variable                    | Purpose                                                                              |
| --------------------------- | ------------------------------------------------------------------------------------ |
| `VLM_PROVIDER`              | `bedrock` (default), `openai`, or `anthropic`                                        |
| `LLM_PROVIDER`              | Same options; defaults to `bedrock`                                                  |
| `VLM_MODEL`                 | Model id for captioning (default `us.anthropic.claude-haiku-4-5-20251001-v1:0`)      |
| `LLM_MODEL`                 | Model id for query understanding (default `us.anthropic.claude-haiku-4-5-20251001-v1:0`) |
| `EMBEDDING_MODEL`           | Bedrock embedding model (default `amazon.titan-embed-image-v1`)                      |
| `EMBEDDING_DIM`             | Titan output dimension: 256, 384, or 1024 (default `1024`)                           |
| `RERANKER_MODEL`            | Bedrock rerank model (default `cohere.rerank-v3-5:0`)                                |
| `CAPTION_TEXT_LANE_ENABLED` | Text-to-text dense lane over caption documents (default `true`)                      |
| `TEXT_EMBEDDING_MODEL`      | Text embedding model for the caption-text lane (default `amazon.titan-embed-text-v2:0`) |
| `TEXT_EMBEDDING_DIM`        | Text embedding dimension (default `1024`)                                            |
| `AWS_REGION`                | AWS region for Bedrock (default `us-east-1`)                                         |
| `AWS_BEARER_TOKEN_BEDROCK`  | Short-lived Bedrock API key (optional; otherwise use standard AWS credentials)       |
| `DATA_DIR`                  | Root for all persisted state                                                         |
| `UPLOADS_DIR`               | Staging dir for UI uploads (default `<DATA_DIR>/uploads`)                            |
| `BLOB_STORAGE_BACKEND`      | Corpus blob backend: `local` (default) or `s3`                                       |
| `S3_BUCKET`                 | Private corpus bucket; required when the blob backend is `s3`                        |
| `S3_PREFIX`                 | Object key prefix for `uploads/` and `images/` (default `imagecb`)                   |
| `S3_REGION`                 | S3 client region; defaults to `AWS_REGION`                                           |
| `TESSERACT_CMD`             | Path to `tesseract.exe`                                                              |
| `INGEST_WORKERS`            | Parallel ingest workers (default `4`; use `2` for large re-ingest)                   |
| `INGEST_MAX_IMAGE_SIDE`     | Max longest edge for VLM caption input (default `1024`)                              |
| `INGEST_BATCH_UPSERT`       | Chroma vectors per batch upsert (default `16`)                                       |
| `INGEST_BATCH_SIZE`         | CLI file batch size (default `0` = no batching; use `25` for large corpora)          |
| `INGEST_IMAGE_TIMEOUT_SEC`  | Per-image ingest timeout in seconds (default `300`)                                  |
| `BEDROCK_MAX_CONCURRENT`    | Max concurrent Bedrock API calls (default `2`)                                       |
| `BEDROCK_READ_TIMEOUT`      | Bedrock read timeout in seconds (default `120`)                                      |
| `BEDROCK_CONNECT_TIMEOUT`   | Bedrock connect timeout in seconds (default `10`)                                    |
| `BEDROCK_MAX_RETRIES`       | Bedrock adaptive retry attempts (default `6`)                                        |

## Project layout

```
imagecb/
  config.py
  ingest.py
  app.py                 (Gradio UI)
  uploads.py             (UI upload staging)
  cli.py                 (Typer CLI)
  models/
    bedrock_client.py    (shared boto3 clients)
    embedder.py          (Titan Multimodal Embeddings)
    vlm.py               (structured-JSON captioner)
    llm.py               (query-understanding LLM)
    reranker.py          (Cohere Rerank 3.5)
    ocr.py               (Tesseract)
  extractors/
    pptx.py pdf.py image_file.py dispatch.py types.py
  storage/
    metadata_db.py       (SQLite + SQLAlchemy)
    vector_store.py      (Chroma)
    bm25_index.py        (rank_bm25, persisted)
  retrieval/
    query_parser.py      (text -> QuerySpec)
    hybrid.py            (filter + visual dense + caption-text dense + BM25 + RRF)
    rerank.py            (Bedrock rerank + provenance formatting)
    session.py           (multi-turn state and follow-up context)
  web/
    frontend_dist/       (pre-built React UI for serve-web — chat + admin)
    static/              (fallback vanilla chat UI)
```

## Admin layer (telemetry, quality, curation)

Set `ADMIN_API_KEY` in `.env` (see `.env.example`). This key protects:

- All `/api/admin/*` endpoints (analytics, corpus curation, audit log)
- `POST /api/ingest` (corpus uploads)

### Telemetry

Every chat and similar search records a **search event** (query, user id, served image ids, top raw score, result count). The UI sends **interaction events** (view, download, similar) linked by `search_event_id` from the stream metadata payload.

Optional header `X-User-Id` on chat requests labels telemetry until real SSO is added (default: `anonymous`).

### Admin UI

With `serve-web` running, open **http://127.0.0.1:8080/admin** and enter
`ADMIN_API_KEY`. The main chat UI also has an **Admin login** link in the
header and sidebar.

Optional: `npm run dev` in `frontend/` for hot reload at
**http://localhost:5173/admin** (requires API on 8080 for `/api` proxy).

For local dev you may set `VITE_ADMIN_API_KEY` in `frontend/.env.local`
(do not commit secrets).

### Soft delete

`POST /api/admin/images/{image_id}/soft-delete` removes the vector from Chroma and rebuilds BM25 while keeping SQLite metadata and cached PNGs. `POST .../restore` re-embeds and re-indexes.

## Notes / limitations (prototype)

- Single-user, single-process. Chat is open; admin and ingest require `ADMIN_API_KEY`.
- No live folder watcher - re-run `ingest` or use **Add to corpus** in the UI for new files.
- All model inference (embeddings, captioning, query parsing, reranking)
  runs through AWS Bedrock — no local GPU or HuggingFace downloads.
- Aimed at hundreds-to-low-thousands of images; for larger corpora swap
  Chroma for Qdrant/Milvus and move BM25 to OpenSearch or Tantivy.
