# ATLAS (Imagecb)

Conversational multimodal search over a mixed image corpus: standalone images plus figures embedded in PowerPoint (`.pptx`) and PDF files.

Each image is captioned by a VLM, embedded with Amazon Titan Multimodal Embeddings on Bedrock, and indexed in ChromaDB (visual + caption-text dense lanes), rank-bm25 (diagnostic sparse index), and SQLite (provenance and metadata filters). Users search in natural language, refine across turns in the same chat, find visually similar images, or get slide-aware suggestions for a deck.

**Runtime:** Python **3.11**, FastAPI on port **8080** (Docker and `serve-web`). Chat and search APIs are unauthenticated; corpus ingest and admin require `ADMIN_API_KEY`.

## Architecture

```
ingest:  files → extractor (pptx/pdf/image) → OCR + VLM caption
                                            → Titan image emb + caption-text emb
                                            → SQLite + Chroma + BM25 (+ hubness)

chat:    text + history → LLM QuerySpec
                        → metadata filter
                        → visual dense + caption-text dense
                        → 2-lane RRF fusion → ranked results (fusion score)

similar: reference image → Titan image NN
                        → optional caption facets → hybrid + Cohere rerank (text leg)
                        → axis-weighted RRF (visual + text)

deck:    .pptx slides → LLM search descriptions
                      → hybrid search + Cohere Rerank 3.5
```

**Chat ranking** fuses two dense lanes with Reciprocal Rank Fusion. BM25 is still retrieved for diagnostics (Pipeline Lab) but does **not** contribute to chat fusion (`sparse_weight=0.0`). Chat does **not** call Cohere rerank, hubness adjustment, or asset-type boost.

**Cohere Rerank 3.5** (with optional hubness + asset-type boost inside `rerank()`) is used by deck suggest and the similar-search text leg. Full pipelines: [How it works](#how-it-works). Feature catalog: [Features](#features).

### Corpus and index source of truth

SQLite active rows (`deleted_at IS NULL`) are the canonical corpus. API field `indexed_count` is a backward-compatible alias for that count; `/api/status` also exposes `total_records`.

Chroma image vectors, caption-text vectors, and BM25 documents are derived indexes. Counts (`chroma_vectors`, `text_vector_count`, `bm25_doc_count`) are health metrics, not corpus size.

**Match %** on result cards is a calibrated 0–100 display value:

| Path | Underlying score |
|------|------------------|
| Chat | Normalized 2-lane RRF fusion |
| Similar | Normalized RRF fusion (visual + text, axis weights) |
| Deck | Cohere rerank relevance |

The min-match % control uses the same scale. **100%** appears for near-excellent rerank (≥ 0.93), dense cosine (≥ 0.92), or fusion at 1.0.

### Multi-turn chat

Each conversation uses a server-side `session_id`. Follow-ups receive compacted history, the previous `QuerySpec`, and a short summary of prior top results so the query LLM can reinterpret the request (including re-emitting filters when appropriate).

Every turn searches the **full active corpus** — there is no hard pool limited to previous hits. Sessions are **in-memory**; restarting the process clears them.

## Setup

### 1. Python deps

Requires **Python 3.11** (matches Docker and CI).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Embeddings, captioning, query parsing, conversational replies, and reranking run through cloud APIs (Bedrock by default). No local ML model downloads are required.

### 2. Tesseract OCR (Windows)

1. Install from https://github.com/UB-Mannheim/tesseract/wiki.
2. Typical path: `C:\Program Files\Tesseract-OCR\tesseract.exe`.
3. Set `TESSERACT_CMD` in `.env`, or leave empty if `tesseract` is on `PATH`.

Without OCR, ingest continues and OCR text is blank.

### 3. AWS Bedrock (default)

| Role | Model | API |
|------|-------|-----|
| Image embeddings | `amazon.titan-embed-image-v1` | `invoke_model` |
| Caption-text embeddings | `amazon.titan-embed-text-v2:0` | `invoke_model` |
| VLM captioning | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | `converse` |
| Query parsing / replies | same | `converse` |
| Reranking (deck + similar text leg) | `cohere.rerank-v3-5:0` | `invoke_model` |

1. In the Bedrock console for your region, enable the models above. Prefer **`us-east-1` or `us-west-2`**: Cohere Rerank 3.5 is not available in every region (e.g. not in `us-east-2`). Some regions without Titan need Cohere Embed v4 instead — see comments in [`.env.example`](.env.example).
2. Copy env and edit:

   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

3. Auth:
   - **Bedrock API key:** `AWS_BEARER_TOKEN_BEDROCK=...` (short-lived; refresh when it expires).
   - **Standard AWS credentials / instance role:** principal needs `bedrock:InvokeModel` and `bedrock:Converse` for the configured model IDs.

4. Leave `AWS_REGION=us-east-1` unless you intentionally use another region (and matching inference-profile prefixes where required).

With captions enabled, ingest makes **three Bedrock calls per image** (VLM caption + Titan image embed + Titan text embed). Use `--skip-caption` to skip the VLM (image embed still runs).

**Required for production:** set a strong `ADMIN_API_KEY` in `.env`. If it is empty, admin and ingest APIs return **503** (“Admin API is not configured”).

### 4. Optional: OpenAI or Anthropic for VLM / LLM

```
VLM_PROVIDER=openai      # or anthropic
LLM_PROVIDER=openai
VLM_MODEL=gpt-4o-mini
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...    # or ANTHROPIC_API_KEY=...
```

Embeddings and reranking remain on Bedrock.

## Production operations

### Security boundaries

| Surface | Auth |
|---------|------|
| Chat, similar, deck suggest, image/source downloads | **Open** (no API key) |
| Ingest (`/api/ingest*`) and all `/api/admin/*` | `ADMIN_API_KEY` via `X-Admin-Api-Key` or `Authorization: Bearer <key>` |
| Pipeline Lab (`/lab`, `/api/lab/*`) | **Open** (experimental; disable by removing the experiments router if undesired) |

The admin UI stores the key in the browser after login. For scripts:

```http
X-Admin-Api-Key: <ADMIN_API_KEY>
```

CORS is hardcoded to localhost origins (`127.0.0.1` / `localhost` on ports 5173, 8080, 8081). Same-origin browser use behind a reverse proxy is fine; cross-origin browser clients on other hostnames are not. Prefer TLS termination and network controls at the load balancer / reverse proxy — this app does not terminate HTTPS itself.

There is no end-user SSO yet. Optional `X-User-Id` labels telemetry only (default `anonymous`).

### Health and readiness

| Endpoint | Use |
|----------|-----|
| `GET /api/health` | Liveness — always `{"status":"ok"}` if the process is up. Used by Docker/Compose healthchecks. |
| `GET /api/status` | Index counts + `is_healthy` / `stores_in_sync` |
| `GET /api/ready` | Readiness — **503** with issue list when the index is unhealthy |
| `GET /api/admin/index/health` | Full health report (requires admin key) |
| `POST /api/admin/ingest/preflight` | Writable data dir, SQLite, S3 round trip, model access (requires admin key) |

Wire load balancers so **liveness** uses `/api/health` and **readiness** uses `/api/ready` (or fail closed on `/api/status` when `is_healthy` is false).

### Startup behavior

On `serve-web` / container start the app:

1. Optionally **auto-restores** the search index from S3 when remote record count exceeds local (`INDEX_AUTO_RESTORE_ON_STARTUP`, default `true` when applicable).
2. Assesses index health and runs **safe reconcile** if `INDEX_RECONCILE_ON_STARTUP=true` (default).
3. Starts background **bootstrap ingest** only if the index is empty and `BOOTSTRAP_CORPUS_DIR` is set.
4. Starts the **ingest job runner**.

With `BLOB_STORAGE_BACKEND=s3`, rolling checkpoints to S3 are enabled by default (`INDEX_CHECKPOINT_ENABLED` auto-true; `INDEX_CHECKPOINT_EVERY_N=10`). Live SQLite/Chroma/BM25/hubness always remain on the local data volume — S3 holds blobs and index snapshots, not the live query path.

### Post-deploy checklist

1. Set `APP_BUILD_ID` to the deployed commit SHA before `docker compose build`.
2. Confirm `ADMIN_API_KEY`, Bedrock auth, and (if used) `S3_BUCKET` / `BLOB_STORAGE_BACKEND=s3`.
3. `GET /api/ready` returns 200.
4. Admin → **Ingestions → Ingest preflight** succeeds.
5. `python -m imagecb.cli validate-reranker` (needed for deck / similar text leg).
6. Run a smoke chat query and open a result image.

### Backup and restore

**Requires** `BLOB_STORAGE_BACKEND=s3` and `S3_BUCKET`.

| Mechanism | Purpose |
|-----------|---------|
| Rolling `checkpoint-latest` | Automatic during ingest when checkpointing is enabled |
| Versioned `index-backups/{id}/` | Explicit snapshots via Admin → Corpus or CLI |

```powershell
python -m imagecb.cli list-index-backups
python -m imagecb.cli backup-index
python -m imagecb.cli restore-index <backup_id> --yes
```

`backup-index` quiesces ingest while snapshotting. `restore-index` **replaces** the live local index; `--yes` is mandatory.

For `BLOB_STORAGE_BACKEND=local`, back up the entire `./data` volume (SQLite, Chroma, BM25, hubness, uploads, image cache). Telemetry/search events live in the same SQLite database as corpus metadata.

## Docker

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) or Docker Engine + Compose v2
- A `.env` with Bedrock (or other provider) credentials and `ADMIN_API_KEY`

### Start

```powershell
docker compose up --build
```

Open http://localhost:8080. The image builds the React UI and serves it via FastAPI. Compose mounts `./data` → `/app/data` and `./corpus` → `/corpus` (read-only), sets `restart: unless-stopped`, and healthchecks `GET /api/health`.

Root Compose clears `BOOTSTRAP_CORPUS_DIR` so the image’s baked-in smoke corpus is **not** auto-ingested. To seed on empty boot, set `BOOTSTRAP_CORPUS_DIR` in `.env` (for example `/corpus/smoke` if that path exists in the container).

```powershell
docker compose run --rm imagecb python -m imagecb.cli ingest /corpus/<scenario>
docker compose run --rm imagecb python -m imagecb.cli status
docker compose run --rm imagecb python -m imagecb.cli repair-index
```

### Reset local Docker search state

```powershell
docker compose down
# Clear or replace host ./data, then:
docker compose up --build
```

Resets local indexes and local blobs; does not delete private S3 objects. If SQLite still has `s3://` URIs from an earlier S3 run, clear `./data` (or restore a local-only copy) before comparing timings.

### Private S3 corpus storage (EC2)

S3 stores uploads, display PNGs, timing reports, and index snapshots. The live search index stays on a persistent local volume (`./data` / `/app/data`).

```dotenv
BLOB_STORAGE_BACKEND=s3
S3_BUCKET=your-private-corpus-bucket
S3_PREFIX=imagecb
S3_REGION=us-east-1
S3_READ_TIMEOUT=120
S3_PRESIGN_EXPIRY_SEC=3600
BOOTSTRAP_CORPUS_DIR=
ADMIN_API_KEY=replace-with-long-random-secret
```

(`S3_READ_TIMEOUT` defaults to `30` in code; raise it for large transfers.)

Object layout under the prefix:

- `uploads/` — original source files
- `staging/` — browser uploads awaiting validation
- `images/` — display PNGs
- `thumbs/` — JPEG display thumbnails (`{image_id}.jpg`)
- `index-backups/` — versioned index snapshots
- `ingest-logs/` / `query-logs/` — timing reports when enabled

**IAM (instance role preferred):** `s3:GetObject`, `s3:PutObject`, `s3:DeleteObject` on `arn:aws:s3:::your-private-corpus-bucket/imagecb/*`, plus `s3:ListBucket` scoped to that prefix; Bedrock invoke/converse for configured models. Do not put long-lived AWS access keys in `.env`.

Configure bucket CORS for browser PUTs to your real web origin, and a lifecycle rule that expires `imagecb/staging/` after about seven days.

Migrate existing local blobs:

```powershell
python -m imagecb.cli migrate-blobs-to-s3
python -m imagecb.cli migrate-blobs-to-s3 --apply
```

(The MinIO playground under `.local/s3-playground/` is separate and unused by root Compose.)

### Docker troubleshooting

| Issue | What to try |
|-------|-------------|
| Cannot reach :8080 | Confirm the container is up; it binds `0.0.0.0` inside the image |
| Bedrock errors | Refresh `AWS_BEARER_TOKEN_BEDROCK` or check region / model access |
| `ThrottlingException` / OOM during ingest | Lower `INGEST_WORKERS` and `BEDROCK_MAX_CONCURRENT` (large concurrent ingest can OOM around ~125 images on small instances) |
| Admin / ingest 503 | Set `ADMIN_API_KEY` in `.env` and recreate the container |

## Usage

### Ingest a corpus

```powershell
python -m imagecb.cli ingest "C:\path\to\corpus"
```

Accepts a file or directory (recursive). Supported: `.pptx`, `.pdf`, and common image extensions (`.png .jpg .jpeg .webp .bmp .gif .tif .tiff`). Re-runs are idempotent (content-hash dedupe).

```powershell
python -m imagecb.cli ingest "C:\path\to\corpus" --skip-caption
python -m imagecb.cli ingest "C:\path\to\corpus" --force --workers 2 --batch-size 25
```

| Flag | Effect |
|------|--------|
| `--workers N` | Parallel images (default from `INGEST_WORKERS`, typically `2`) |
| `--batch-size 25` | Process in batches; rebuild BM25 once at end |
| `--no-defer-bm25` | Rebuild BM25 after every batch |
| `--skip-ocr` | Skip Tesseract |
| `--max-image-side 1024` | Cap longest edge sent to the VLM |
| `--force` | Re-process duplicates |

Shared knobs: `INGEST_WORKERS=2`, `BEDROCK_MAX_CONCURRENT=2`, `INGEST_BATCH_SIZE=25`.

To re-embed an existing index without re-extracting decks:

```powershell
python -m imagecb.cli reindex-embeddings --workers 4
```

### Index health and repair

After ingest, missing PNGs / failed captions / missing Chroma vectors are repaired automatically unless `POST_INGEST_REPAIR_ENABLED=false`.

```powershell
python -m imagecb.cli status
python -m imagecb.cli status --verbose
python -m imagecb.cli reconcile-index
python -m imagecb.cli repair-index
python -m imagecb.cli repair-index --dry-run
python -m imagecb.cli repair-captions --workers 4
```

Rows missing both cached PNG and original source are **unrecoverable** (not deleted automatically). Do not run parallel CLI ingests against the same corpus.

### Launch the web UI

```powershell
python -m imagecb.cli serve-web
```

Defaults to `127.0.0.1:8080`. For remote access (non-Docker):

```powershell
python -m imagecb.cli serve-web --host 0.0.0.0 --port 8080
```

The repo ships a pre-built React bundle at `imagecb/web/frontend_dist/` (no local `npm` required). Startup should print `UI bundle: ATLAS (React, imagecb/web/frontend_dist)`.

| URL | Surface |
|-----|---------|
| http://127.0.0.1:8080/ | Chat search |
| http://127.0.0.1:8080/deck | Deck suggest |
| http://127.0.0.1:8080/admin | Admin (`ADMIN_API_KEY`) |
| http://127.0.0.1:8080/lab | Pipeline Lab (experimental) |

If `frontend_dist` is missing, `serve-web` falls back to `imagecb/web/static/` (chat only, no admin).

**Frontend developers:**

```powershell
cd frontend
npm ci
npm run build
cd ..
python scripts/sync_frontend_dist.py
```

CI fails if `frontend_dist` drifts from `frontend/`. Optional: `npm run dev` → http://localhost:5173 (API proxy to `:8080`).

**Legacy Gradio** (deprecated): `python -m imagecb.cli serve` → http://127.0.0.1:7860.

## Features

### Chat (`/`)

| Feature | Behavior |
|---------|----------|
| Layout | Search column (sidebar + messages + composer) and results column |
| Startup | Brief loading screen; `GET /api/status` for indexed count in header/footer |
| Conversations | Multi-chat sidebar; titles and turns in browser `localStorage`; each chat has a server `session_id` |
| Sidebar search | Client-side search over titles and turn text; jump to a matching turn |
| New / delete chat | Creates or removes local conversation state (server session is not deleted; `POST /api/session/reset` exists but the React UI does not call it) |
| Empty state | Starter suggestion chips from `POST /api/suggestions` |
| Streaming search | `POST /api/chat/stream` (SSE: metadata with results/`parsed_query`, then tokens, then done). Sync `POST /api/chat` also exists |
| Turn selection | Clicking a prior turn restores that turn’s cached results in the results panel and telemetry `search_event_id` |
| Follow-up chips | After each assistant reply when enabled (`ENABLE_FOLLOW_UP_SUGGESTIONS`) |
| Composer | Enter sends; Shift+Enter newline; max results 1–50 (`top_k`) |
| Advanced controls | Min match % (0–100); similarity axis (balanced / subject / style / layout) for **similar** search only; open state persisted |
| Camera / image search | Uploads a reference image → new turn → `POST /api/similar` (multipart) |
| Deck link | Composer icon → `/deck` |
| Results sort | Relevance, newest, oldest, name, source — applied client-side and sent as `sort` on chat/similar APIs |
| Result cards | Rank, calibrated match %, caption, tags, asset type, provenance (source/slide/page/date/author), use case, recommended case / match hints |
| Find similar | From a result card → new chat turn via `POST /api/similar` with that `image_id` |
| Open source file | `GET /api/sources/{id}` (original pptx/pdf/image) |
| Download image | `GET /api/images/{id}` display PNG |
| Telemetry | Card view / download / similar → `POST /api/telemetry/interaction` linked by `search_event_id` |
| Multi-turn refine | Follow-ups re-query the **full corpus** with history + prior `QuerySpec` + prior top-result summary (no locked previous-hit pool) |
| Parsed query | Stream metadata includes `parsed_query` and interpretation notes; the React UI stores them on the turn but does not render a dedicated interpretation panel |

Error banners surface API failures at the top of the chat view.

### Add to Database (corpus drawer)

Requires an `ADMIN_API_KEY` already stored from Admin login.

| Feature | Behavior |
|---------|----------|
| Upload | Images, PDFs, `.pptx`; drag/drop or file picker; optional skip caption, skip OCR, force, workers |
| S3 path | Preferred: `POST /api/ingest/jobs/s3` → browser presigned PUTs → finalize → background job |
| Local / fallback | Server-staged job API (`POST /api/ingest/jobs` + file batches) when S3 direct upload is unavailable |
| Progress | Polls job status; cancel via admin ingest cancel |
| Catalog preview | `GET /api/corpus/catalog` (recent corpus rows with sort) |
| Stuck job recovery | Clear client lock via `?clearIngest=1`, `#clearIngest`, or clearing `atlas.activeIngestJobId` in localStorage |

Sync `POST /api/ingest` remains available for scripts; the drawer uses the **job** APIs.

### Deck suggest (`/deck`)

| Feature | Behavior |
|---------|----------|
| Upload | `.pptx` up to `DECK_MAX_UPLOAD_BYTES` (default 50 MB), max `DECK_MAX_SLIDES` (default 200) |
| Pipeline | Extract slides → batched LLM descriptions (`image_needed` / `no_image_needed`) → hybrid + Cohere rerank per needed slide |
| Cache | Disk cache under `DECK_CACHE_DIR`; UI can show full-deck or per-slide cache hits |
| Controls | Top-K (UI caps lower than chat), min match %, sort |
| Accept / Dismiss | **Browser-only** (localStorage per deck hash); not stored on the server |
| Force image | `POST /api/deck/force` re-runs search for a `no_image_needed` slide |
| Extracted text | Per-slide toggle to preview title/body/notes used for the LLM |
| Match % | Cohere rerank scale |

### Admin (`/admin`)

Gate with `ADMIN_API_KEY` (entered in the browser after unlock; kept in sessionStorage only - build-time env fallbacks were removed so the key can never ship in the JS bundle).

| Route | Features |
|-------|----------|
| `/admin` | 7-day KPIs: searches, zero/weak/no-interaction rates, interactions; caption health (failed/weak) |
| `/admin/quality` | Tables of zero-result, weak-result, and no-interaction searches with stage timing columns |
| `/admin/corpus` | Index health; reconcile; full repair; purge unrecoverable; S3 backup/restore; corpus grid (quality filter, sort); bulk repair failed/weak captions; per-image regenerate caption, reindex, soft-delete; orphans; soft-deleted restore; near-duplicate clusters |
| `/admin/ingestions` | Active/recent jobs; cancel; runtime diagnostics (`APP_BUILD_ID` mismatch warning); ingest preflight; deep link `?job=` |
| `/admin/audit` | Admin action audit log |

`GET /api/admin/analytics/funnel` exists for per–search-event funnel detail but has **no Admin UI page** yet.

**Per-image ops:** regenerate caption re-runs the VLM and reindexes; reindex re-embeds/reindexes without a full VLM caption pass when possible.

### Pipeline Lab (`/lab`)

Open experimental UI (no admin key) to compare ranking variants for one query (`GET /api/lab/variants`, `POST /api/lab/compare` / stream). **Not** production chat. Variant labels in the lab UI may still describe older 3-lane + rerank baselines; production chat is 2-lane RRF fusion only (see [How it works](#how-it-works)). Remove by deleting `imagecb/experiments` and its router registration in `imagecb/api/server.py`.

### Notable APIs

| Endpoint | Role |
|----------|------|
| `POST /api/chat/stream` | Primary chat path (React) |
| `POST /api/chat` | Same search, sync JSON reply |
| `POST /api/similar` | Similar by `image_id` or uploaded image |
| `POST /api/session/reset` | Clear in-memory session (not used by React UI) |
| `POST /api/suggestions` | Empty-state starter chips |
| `GET /api/corpus/catalog` | Browse recent corpus rows |
| `POST /api/telemetry/interaction` | view / download / similar |
| `GET /api/images/{id}` / `GET /api/sources/{id}` | Display PNG / original source |
| `POST /api/deck/suggest` / `POST /api/deck/force` | Deck pipeline |
| `POST /api/ingest*` / `/api/ingest/jobs*` | Admin-keyed ingest |

## How it works

### Ingest pipeline

Supported inputs: `.pptx`, `.pdf`, and common image extensions. Entry points: CLI `ingest`, sync `POST /api/ingest`, background job APIs, S3 direct-upload jobs, optional `BOOTSTRAP_CORPUS_DIR` on empty boot.

Per image (typical):

1. **Extract** — slide/page/image bytes + provenance (source path, slide/page, author, modified time).
2. **OCR** — Tesseract when enabled; blank if skipped/unavailable.
3. **VLM caption** — structured JSON (short caption, tags, use cases, recommended searches, asset type, scene, etc.). Quality classified ok / weak / failed with one retry on failure.
4. **Normalize** — closed asset-type taxonomy; tag vocab normalization against corpus terms.
5. **Embed** — Titan multimodal image embedding (optional slide/PDF title/notes as `inputText`) + Titan text embedding of the caption document for the caption-text lane.
6. **Persist** — SQLite metadata; Chroma image + caption-text vectors; source blob + display PNG (local `data/` or S3 `uploads/` / `images/`).
7. **Index maintenance** — BM25 rebuild (caption + OCR + slide context); hubness stats rebuild from embeddings; optional post-ingest repair, reconcile, and S3 index checkpoint.

Images are de-duplicated by **content hash** unless `--force` / force flag. With captions: **three** Bedrock calls per image (VLM + image embed + text embed). `--skip-caption` skips the VLM; image embed still runs.

### QuerySpec (query understanding)

The query LLM maps each chat turn (+ history + prior filters/results summary) into a `QuerySpec`:

| Field | Role |
|-------|------|
| `semantic_query` | Primary retrieval phrase |
| `must_have_keywords` | Appended to dense/BM25 query text (**enrichment**, not a hard gate) |
| `must_avoid_keywords` | After fusion, drop candidates whose caption/OCR/slide text contains these substrings |
| `source_filters` | SQLite pre-filter: `file_types`, `asset_types`, `filename_contains`, `authors` |
| `time_filter` | SQLite pre-filter on modified `before` / `after` |
| `top_k` | Result count (1–50) |
| `is_refinement` | Flag for reply copy / suggestions; does **not** restrict search to prior IDs |

Natural-language filters (e.g. “from Q3_Review.pptx”, “only diagrams”, “after May 2026”) only apply when the LLM emits them into these fields.

### Chat retrieval (production)

```
parse_query → metadata pre-filter → visual dense + caption-text dense
  (+ BM25 retrieved for diagnostics, sparse_weight=0)
→ 2-lane RRF → normalize fusion score → near-dupe collapse
→ min-match % (relax if none qualify) → sort
→ conversational reply + follow-up suggestions + search_event
```

- **No Cohere rerank** on this path.
- **No visual-fallback / short-query visual bypass** in `ChatSession.ask` (those env flags exist for rerank/lab experiments; production chat always returns fusion ranking).
- **Hubness correction** and **asset-type rerank boost** run inside the Cohere `rerank()` helper — used by deck and similar’s text leg, **not** by chat fusion ranking.
- Near-duplicate collapse: same content hash or embedding cosine ≥ `RESULT_DEDUPLICATE_SIMILARITY_THRESHOLD` when enabled.
- Assistant text: LLM conversational reply when `ENABLE_CONVERSATIONAL_LLM=true`; otherwise a template summary.

### Similar search

1. Load reference (corpus `image_id` or upload) → Titan **image** embedding → visual nearest neighbors (exclude self when searching by id).
2. Build caption facets from the stored record or a live VLM `query_image` for uploads.
3. If facets are usable: **text leg** = hybrid search + **Cohere rerank** (axis-specific semantic focus and keywords).
4. Fuse visual hits + text leg with **axis-weighted** 2-lane RRF (`balanced` / `subject` / `style` / `layout`).
5. Dedupe, min-match filter, sort; update session via `apply_similar_results` (new anchor, `is_refinement=False`).
6. Record `search_kind=similar` telemetry. Similar-as-chat-turn uses a template assistant line (not the streaming chat reply path).

### Deck suggest

1. Hash and extract slides from the `.pptx`.
2. Batch slides through `LLM_MODEL` → per-slide `image_needed` + search description (or `no_image_needed`).
3. For needed slides: `search_for_description` = `QuerySpec` from description → hybrid → **Cohere rerank** (no query-parse LLM).
4. Cache LLM + search results on disk by content hash / slide text.
5. Force path re-runs search for one skipped slide.

### Supporting systems

| System | Behavior |
|--------|----------|
| Blobs | `BLOB_STORAGE_BACKEND=local` or `s3`; image/source routes stream from disk or S3 |
| Telemetry | `search_events` on every chat/similar; `interaction_events` for view/download/similar; admin audit log for mutations |
| Query timing | Stage timings when `QUERY_TIMING_LOG=true`; optional persist under `query-logs/`; surfaced in Admin Search quality |
| Soft delete | Drop Chroma vectors + rebuild BM25; keep SQLite + PNGs; restore re-embeds |
| Hubness index | Rebuilt at ingest; applied when `rerank()` / visual-only ranking runs (not chat fusion) |

### CLI reference

```powershell
python -m imagecb.cli <command>
```

| Command | Purpose |
|---------|---------|
| `ingest` | Index a file or directory |
| `serve-web` | FastAPI + React UI (default `127.0.0.1:8080`) |
| `serve` | Legacy Gradio UI (`:7860`) |
| `status` | Index health summary |
| `reconcile-index` | Purge orphan vectors; rebuild BM25 if stale |
| `repair-index` / `repair-captions` / `rescan-captions` | Repair and quality tools |
| `reindex-embeddings` | Re-embed cached images |
| `validate-reranker` | Smoke-test Bedrock Cohere rerank |
| `backup-index` / `restore-index` / `list-index-backups` | S3 index snapshots |
| `migrate-blobs-to-s3` | Migrate local blobs to S3 |
| `parse-query` | Debug `QuerySpec` |
| `backfill-asset-types` / `audit-asset-types` / `freeze-asset-types` | Asset-type taxonomy |
| `repair-search-terms` | Re-enrich tag aliases |
| `eval-search` / `eval-suggest` | Golden-set evaluation |

### Search evaluation harness

Offline retrieval measurement against `./data` (not required for deploy).

```powershell
python -m imagecb.cli eval-suggest "operational dashboards" --top-k 15
python -m imagecb.cli eval-search
python -m imagecb.cli eval-search --mode retrieval --k 1,5,10
```

Modes: **`chat`** = production fusion path; **`retrieval`** = hybrid + Cohere rerank; **`similar`** = similar-image path. Label cases in [`eval/golden.json`](eval/golden.json).

## Configuration

Full list: [`.env.example`](.env.example). Highlights below. **Path** column: which runtime paths the knob affects.

| Variable | Path | Purpose |
|----------|------|---------|
| `ADMIN_API_KEY` | Admin / ingest | Required; empty → 503 on admin/ingest |
| `APP_BUILD_ID` | Deploy | Commit SHA for diagnostics / image label |
| `VLM_PROVIDER` / `LLM_PROVIDER` | Ingest / chat / deck | `bedrock` (default), `openai`, or `anthropic` |
| `VLM_MODEL` / `LLM_MODEL` | Ingest / chat / deck | Captioning and query/reply / deck LLM |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | Ingest / all search | Multimodal embedder (default Titan `1024`) |
| `TEXT_EMBEDDING_MODEL` / `TEXT_EMBEDDING_DIM` | Ingest / chat / similar / deck | Caption-text dense lane |
| `CAPTION_TEXT_LANE_ENABLED` | All hybrid search | Caption-text dense lane (default `true`) |
| `RERANKER_MODEL` | Deck / similar text leg | Cohere Rerank (not used by chat fusion) |
| `AWS_REGION` / `AWS_BEARER_TOKEN_BEDROCK` | All Bedrock | Region and optional API key |
| `DATA_DIR` | All | SQLite, Chroma, BM25, hubness, local blobs |
| `BLOB_STORAGE_BACKEND` | Blobs / backups | `local` (default) or `s3` |
| `S3_BUCKET` / `S3_PREFIX` / `S3_REGION` | S3 mode | Required when backend is `s3` |
| `S3_READ_TIMEOUT` / `S3_CONNECT_TIMEOUT` / `S3_MAX_RETRIES` | S3 | Client resilience |
| `TESSERACT_CMD` | Ingest | Path to `tesseract` if not on `PATH` |
| `INGEST_WORKERS` | Ingest | Parallel workers (default `2`) |
| `INGEST_BATCH_SIZE` / `INGEST_JOB_CHUNK_SIZE` | Ingest | CLI batching / API job chunking |
| `INGEST_IMAGE_TIMEOUT_SEC` | Ingest | Per-image worker timeout (default `300`) |
| `EMBED_CONTEXT_MAX_CHARS` | Ingest | Truncate slide/PDF context in image embed (default `480`) |
| `BEDROCK_MAX_CONCURRENT` | Ingest / models | Max concurrent Bedrock calls (default `2`) |
| `BEDROCK_READ_TIMEOUT` / `BEDROCK_CONNECT_TIMEOUT` / `BEDROCK_MAX_RETRIES` | Bedrock | Resilience |
| `POST_INGEST_REPAIR_ENABLED` | Ingest | Auto-repair after ingest (default `true`) |
| `INDEX_RECONCILE_ON_STARTUP` / `INDEX_RECONCILE_AFTER_INGEST` | Startup / ingest | Safe reconcile (default `true`) |
| `INDEX_CHECKPOINT_ENABLED` | S3 ingest | Rolling checkpoints (auto-on when backend is `s3`) |
| `INDEX_CHECKPOINT_EVERY_N` | S3 ingest | Checkpoint cadence (default `10`) |
| `INDEX_AUTO_RESTORE_ON_STARTUP` | Startup | Restore when remote ahead of local (default `true`) |
| `BOOTSTRAP_CORPUS_DIR` | Startup | Seed ingest when index empty |
| `INGEST_TIMING_LOG` / `QUERY_TIMING_LOG` / `QUERY_TIMING_PERSIST` | Ops | Timing reports (default `true`) |
| `ENABLE_CONVERSATIONAL_LLM` | Chat replies | LLM vs template assistant text (default `true`) |
| `ENABLE_FOLLOW_UP_SUGGESTIONS` / `FOLLOW_UP_SUGGESTIONS_LIMIT` | Chat UI | Follow-up chips (default on, limit `3`) |
| `SUGGESTIONS_LIMIT` / `SUGGESTIONS_CACHE_TTL_SEC` | Empty state | Starter chips |
| `RESULT_DEDUPLICATE_ENABLED` / `RESULT_DEDUPLICATE_SIMILARITY_THRESHOLD` | Chat / similar | Near-dupe collapse (default on, cosine `0.98`) |
| `WEAK_RESULT_SCORE_THRESHOLD` | Admin analytics | Soft floor for “weak” results (default `0.25`) |
| `DUPLICATE_SIMILARITY_THRESHOLD` | Admin corpus | Near-duplicate cluster detection (default `0.95`) |
| `HUBNESS_CORRECTION_ENABLED` / `HUBNESS_KNN` / `HUBNESS_PENALTY_WEIGHT` | **Rerank / visual-only paths** | CSLS hubness; rebuilt at ingest; **not applied to chat fusion ranking** |
| `ASSET_TYPE_RERANK_BOOST` | **Rerank paths** | Boost matching asset types inside `rerank()` (deck / similar text leg); **not chat fusion** |
| `SHORT_QUERY_MAX_TOKENS` / `SHORT_QUERY_RERANK_TOP_N` / `SHORT_QUERY_RETRIEVAL_TOP_K` | **Rerank / retrieval helpers** | Wider pools for short queries when rerank path runs |
| `VISUAL_FALLBACK_*` / `VISUAL_SHORT_QUERY_*` | Config only for chat | Present in `.env.example`; **production `ChatSession.ask` does not apply them** (lab/experiments territory) |
| `DECK_CACHE_DIR` / `DECK_LLM_BATCH_SIZE` / `DECK_MAX_SLIDES` / `DECK_MAX_UPLOAD_BYTES` / `DECK_CACHE_ENABLED` | Deck | Limits and cache |

## Project layout

```
imagecb/           Python package (API, ingest, retrieval, admin, deck, models, storage)
frontend/          React / Vite source
imagecb/web/       Pre-built frontend_dist + fallback static UI
eval/              Golden-set evaluation
scripts/           sync_frontend_dist.py, smoke helpers
corpus/            Optional seed/test files (Docker /corpus mount)
data/              Runtime state (gitignored locally)
tests/             Pytest suite
.github/workflows/ CI (pytest, frontend_dist drift check)
```

## Notes / limitations

- Single-process deployment. Chat/search/deck/lab are open unless you front the service with auth or network policy; admin and ingest require `ADMIN_API_KEY`.
- Chat sessions are in-memory; process restart clears multi-turn context (conversation UI history in the browser is separate).
- No live folder watcher — re-run `ingest` or use **Add to Database**.
- Aimed at hundreds to low thousands of images; for larger corpora consider Qdrant/Milvus and OpenSearch/Tantivy.
- All model inference runs through cloud APIs — no local GPU required.
- SQLite schema (including telemetry) migrates automatically on startup; still take a data-volume or S3 index backup before major upgrades.
- Pipeline Lab variant descriptions may lag the production chat path; trust [How it works](#how-it-works) over lab labels.
