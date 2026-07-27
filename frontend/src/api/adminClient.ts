import type { IngestJob, ResultSort } from "../types";

const ADMIN_KEY_STORAGE = "imagecb.adminApiKey";

export function getAdminApiKey(): string | null {
  return (
    sessionStorage.getItem(ADMIN_KEY_STORAGE) ||
    import.meta.env.VITE_ADMIN_API_KEY ||
    null
  );
}

export function setAdminApiKey(key: string): void {
  sessionStorage.setItem(ADMIN_KEY_STORAGE, key);
}

export function clearAdminApiKey(): void {
  sessionStorage.removeItem(ADMIN_KEY_STORAGE);
}

async function adminRequest<T>(
  path: string,
  init?: RequestInit,
  explicitKey?: string,
): Promise<T> {
  const key = explicitKey ?? getAdminApiKey();
  if (!key) {
    throw new Error("Admin API key not set");
  }
  const headers: Record<string, string> = {
    "X-Admin-Api-Key": key,
    ...(init?.headers as Record<string, string> | undefined),
  };
  const res = await fetch(path, { ...init, headers, cache: "no-store" });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* ignore */
    }
    throw new Error(String(detail));
  }
  return res.json() as Promise<T>;
}

/** Probe a protected admin endpoint with a candidate key (does not persist it). */
export async function verifyAdminApiKey(key: string): Promise<void> {
  await adminRequest("/api/admin/corpus/health", undefined, key);
}

export interface AnalyticsSummary {
  since: string;
  total_searches: number;
  zero_result_count: number;
  weak_result_count: number;
  no_interaction_count: number;
  searches_with_results: number;
  interaction_count: number;
  interaction_rate: number;
  zero_result_rate: number;
  weak_result_rate: number;
  no_interaction_rate: number;
  weak_score_threshold: number;
}

export interface SearchQualityLists {
  zero_result: SearchQualityItem[];
  weak_result: SearchQualityItem[];
  no_interaction: SearchQualityItem[];
  weak_score_threshold: number;
}

export interface SearchQualityItem {
  search_event_id: string;
  created_at: string | null;
  query_text: string;
  user_message?: string;
  display_query: string;
  parsed_semantic_query?: string | null;
  user_id: string;
  result_count: number;
  top_score: number | null;
  top_score_kind: string | null;
  total_ms?: number | null;
  ask_ms?: number | null;
  reply_ms?: number | null;
  timings?: Record<string, number>;
  timing_log?: string | null;
  category: string;
}

export type CaptionQualityFilter = "all" | "ok" | "weak" | "failed";

export interface CorpusImage {
  image_id: string;
  caption_short?: string | null;
  image_name?: string | null;
  source_file?: string;
  source_type?: string;
  author?: string | null;
  image_url: string;
  thumb_url?: string;
  caption_quality?: string;
  needs_regeneration?: boolean;
  created_at?: string | null;
}

export interface CorpusHealth {
  total_images: number;
  total_records?: number;
  chroma_vectors?: number;
  text_vector_count?: number;
  bm25_doc_count?: number;
  thumb_count?: number;
  missing_thumb_count?: number;
  orphan_thumb_count?: number;
  orphan_image_count?: number;
  failed_caption_count: number;
  weak_caption_count: number;
  needs_regeneration_count: number;
  is_healthy: boolean;
  stores_in_sync?: boolean;
  orphan_chroma_count?: number;
  orphan_text_vector_count?: number;
  bm25_stale?: boolean;
  missing_cache_count?: number;
  unrecoverable_source_missing_count?: number;
  unrecoverable_image_ids?: string[];
}

export interface IndexReconcileResult {
  ok: boolean;
  dry_run: boolean;
  orphan_chroma_purged: number;
  orphan_text_purged: number;
  bm25_rebuilt: boolean;
  stores_in_sync_before: boolean;
  stores_in_sync_after: boolean;
  is_healthy_before: boolean;
  is_healthy_after: boolean;
  elapsed_sec?: number;
}

export interface IndexRepairResult {
  ok: boolean;
  skipped?: boolean;
  is_healthy?: boolean;
  elapsed_sec?: number;
}

export interface RepairCaptionsResult {
  ok: boolean;
  attempted: number;
  repaired: number;
  errors: number;
  elapsed_sec?: number;
  scope?: string;
}

export interface RegenerateMissingThumbsResult {
  ok: boolean;
  scanned: number;
  created: number;
  skipped: number;
  failed: number;
  elapsed_sec?: number;
  errors?: { image_id: string; error: string }[];
}

export interface PurgeUnrecoverableResult {
  ok: boolean;
  candidates: number;
  deleted: number;
  skipped: number;
  files_deleted: number;
  files_skipped: number;
  image_ids: string[];
}

export interface RegenerateCaptionResult {
  ok: boolean;
  image_id: string;
  caption_quality: string;
  needs_regeneration: boolean;
  caption_short?: string | null;
  caption_detailed?: string | null;
  image_name?: string | null;
  tags?: string[];
}

export interface ReindexImageResult {
  ok: boolean;
  image_id: string;
  reindexed: boolean;
  caption_short?: string | null;
  caption_quality?: string;
}

export interface IngestDiagnostics {
  build_id: string;
  runtime_id: string;
  pid: number;
  storage_backend: string;
  aws_region: string;
  s3_region: string;
  s3_bucket?: string | null;
  s3_prefix: string;
  embedding_model: string;
  text_embedding_model: string;
  vlm_model: string;
  sqlite_path: string;
  sqlite_identity: string;
  runner: {
    runner_id: string;
    alive: boolean;
    thread_name?: string | null;
  };
  index_checkpoint_enabled?: boolean;
  index_checkpoint_every_n?: number;
  index_auto_restore_on_startup?: boolean;
  last_checkpoint?: {
    backup_id?: string | null;
    total_records?: number | null;
    chroma_vectors?: number | null;
    label?: string | null;
    created_at?: string | null;
    error?: string | null;
  };
  startup_restore?: {
    attempted?: boolean;
    restored?: boolean;
    backup_id?: string | null;
    total_records?: number | null;
    error?: string | null;
    skipped?: string | null;
  };
}

export interface IngestPreflight {
  ok: boolean;
  runtime: IngestDiagnostics;
  checks: Array<{
    name: string;
    ok: boolean;
    detail: string;
    elapsed_ms: number;
  }>;
}

export function fetchAnalyticsSummary(days = 7): Promise<AnalyticsSummary> {
  return adminRequest(`/api/admin/analytics/summary?days=${days}`);
}

export function fetchSearchQuality(limit = 50): Promise<SearchQualityLists> {
  return adminRequest(`/api/admin/analytics/search-quality?limit=${limit}`);
}

export function fetchFunnel(searchEventId: string): Promise<unknown> {
  return adminRequest(
    `/api/admin/analytics/funnel?search_event_id=${encodeURIComponent(searchEventId)}`,
  );
}

export function fetchAudit(limit = 100, offset = 0): Promise<{ entries: unknown[] }> {
  return adminRequest(`/api/admin/audit?limit=${limit}&offset=${offset}`);
}

export function fetchCorpusImages(
  sort?: ResultSort,
  captionQuality?: CaptionQualityFilter,
): Promise<{ images: CorpusImage[] }> {
  const params = new URLSearchParams();
  if (sort) params.set("sort", sort);
  if (captionQuality && captionQuality !== "all") {
    params.set("caption_quality", captionQuality);
  }
  const qs = params.toString();
  return adminRequest(`/api/admin/corpus/images${qs ? `?${qs}` : ""}`);
}

export function fetchCorpusHealth(): Promise<CorpusHealth> {
  return adminRequest("/api/admin/corpus/health");
}

export function reconcileIndex(): Promise<IndexReconcileResult> {
  return adminRequest("/api/admin/index/reconcile", { method: "POST" });
}

export function repairIndex(includeWeak = false): Promise<IndexRepairResult> {
  return adminRequest(
    `/api/admin/index/repair?include_weak=${includeWeak ? "true" : "false"}`,
    { method: "POST" },
  );
}

export interface IndexBackupInfo {
  id: string;
  created_at?: string | null;
  label?: string | null;
  size_bytes?: number | null;
  sha256?: string | null;
  prefix?: string;
  s3_uri?: string;
  artifacts?: Record<string, unknown>;
  build_id?: string | null;
}

export interface IndexBackupResult {
  ok: boolean;
  backup_id: string;
  s3_uri?: string;
  archive_bytes?: number;
  archive_sha256?: string;
  label?: string | null;
  quiesced?: boolean;
  cancelled_job_ids?: string[];
}

export interface IndexRestoreResult {
  ok: boolean;
  backup_id: string;
  s3_uri?: string;
  archive_sha256?: string;
  quiesced?: boolean;
  restart_required?: boolean;
  cancelled_job_ids?: string[];
}

export function listIndexBackups(): Promise<{ backups: IndexBackupInfo[] }> {
  return adminRequest("/api/admin/index/backups");
}

export function backupIndex(label?: string): Promise<IndexBackupResult> {
  return adminRequest("/api/admin/index/backup", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(label ? { label } : {}),
  });
}

export function restoreIndex(backupId: string): Promise<IndexRestoreResult> {
  return adminRequest("/api/admin/index/restore", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ backup_id: backupId, confirm: true }),
  });
}

export function repairCaptions(
  scope: "failed" | "weak",
): Promise<RepairCaptionsResult> {
  return adminRequest(
    `/api/admin/corpus/repair-captions?scope=${encodeURIComponent(scope)}`,
    { method: "POST" },
  );
}

export function regenerateMissingThumbnails(): Promise<RegenerateMissingThumbsResult> {
  return adminRequest("/api/admin/corpus/regenerate-missing-thumbs", {
    method: "POST",
  });
}

export function purgeUnrecoverable(
  imageIds?: string[],
): Promise<PurgeUnrecoverableResult> {
  return adminRequest("/api/admin/corpus/purge-unrecoverable", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(
      imageIds && imageIds.length > 0 ? { image_ids: imageIds } : {},
    ),
  });
}

export interface OrphanBlobPurgeResult {
  ok: boolean;
  dry_run: boolean;
  min_age_hours: number;
  orphan_image_count: number;
  orphan_thumb_count: number;
  orphan_upload_count: number;
  orphan_staging_count: number;
  skipped_too_new_count: number;
  purgeable_count: number;
  deleted_count: number;
  failed_count: number;
  elapsed_sec?: number;
}

export function fetchOrphanBlobs(
  minAgeHours = 1,
): Promise<OrphanBlobPurgeResult> {
  return adminRequest(
    `/api/admin/corpus/orphan-blobs?min_age_hours=${encodeURIComponent(String(minAgeHours))}`,
  );
}

export function purgeOrphanBlobs(options?: {
  dryRun?: boolean;
  minAgeHours?: number;
}): Promise<OrphanBlobPurgeResult> {
  return adminRequest("/api/admin/corpus/purge-orphan-blobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dry_run: options?.dryRun ?? true,
      min_age_hours: options?.minAgeHours ?? 1,
    }),
  });
}

export function fetchOrphans(neverInteracted = false): Promise<{ orphans: unknown[] }> {
  return adminRequest(
    `/api/admin/corpus/orphans?never_interacted=${neverInteracted}`,
  );
}

export function fetchDeleted(): Promise<{ deleted: unknown[] }> {
  return adminRequest("/api/admin/corpus/deleted");
}

export function fetchDuplicateClusters(): Promise<{
  clusters: unknown[];
  error?: string | null;
}> {
  return adminRequest("/api/admin/corpus/duplicate-clusters");
}

export function softDeleteImage(imageId: string): Promise<unknown> {
  return adminRequest(`/api/admin/images/${imageId}/soft-delete`, {
    method: "POST",
  });
}

export function restoreImage(imageId: string): Promise<unknown> {
  return adminRequest(`/api/admin/images/${imageId}/restore`, {
    method: "POST",
  });
}

export function regenerateCaption(imageId: string): Promise<RegenerateCaptionResult> {
  return adminRequest(`/api/admin/images/${imageId}/regenerate-caption`, {
    method: "POST",
  });
}

export function reindexImage(imageId: string): Promise<ReindexImageResult> {
  return adminRequest(`/api/admin/images/${imageId}/reindex`, {
    method: "POST",
  });
}

export function fetchIngestJobs(limit = 100): Promise<{ jobs: IngestJob[] }> {
  return adminRequest(`/api/ingest/jobs?limit=${limit}`);
}

export function fetchIngestDiagnostics(): Promise<IngestDiagnostics> {
  return adminRequest("/api/admin/ingest/diagnostics");
}

export function runIngestPreflight(): Promise<IngestPreflight> {
  return adminRequest("/api/admin/ingest/preflight", { method: "POST" });
}

export function cancelIngestJob(jobId: string): Promise<IngestJob> {
  return adminRequest(
    `/api/ingest/jobs/${encodeURIComponent(jobId)}/cancel`,
    { method: "POST" },
  );
}
