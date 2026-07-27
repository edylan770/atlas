import { getAdminApiKey } from "./adminClient";
import { getUserId } from "./telemetry";
import type {
  ChatStreamCallbacks,
  ChatStreamMetadata,
  CorpusCatalogResponse,
  IngestJob,
  ParsedQuery,
  ResultSort,
  SimilarResponse,
  StatusResponse,
  SuggestionsResponse,
} from "../types";

const SUPPORTED_EXT = new Set([
  ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff", ".pdf", ".pptx",
]);

export function filterSupportedFiles(files: File[]): File[] {
  return files.filter((f) => {
    const name = f.name.toLowerCase();
    const dot = name.lastIndexOf(".");
    const ext = dot >= 0 ? name.slice(dot) : "";
    return SUPPORTED_EXT.has(ext);
  });
}

const API_BASE = "";

function withUserHeaders(init?: RequestInit): RequestInit {
  const headers: Record<string, string> = {
    ...(init?.headers as Record<string, string> | undefined),
  };
  const uid = getUserId();
  if (uid) headers["X-User-Id"] = uid;
  return { ...init, headers };
}

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, withUserHeaders(init));
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? body.message ?? detail;
      if (Array.isArray(detail)) {
        detail = detail.map((d) => d.msg ?? JSON.stringify(d)).join("; ");
      }
    } catch {
      /* ignore */
    }
    throw new Error(String(detail));
  }
  return res.json() as Promise<T>;
}

export async function fetchStatus(): Promise<StatusResponse> {
  return request<StatusResponse>("/api/status");
}

export async function fetchSuggestions(
  limit = 4,
  init?: RequestInit,
): Promise<SuggestionsResponse> {
  return request<SuggestionsResponse>("/api/suggestions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ limit }),
    ...init,
  });
}


type StreamEvent =
  | {
      type: "metadata";
      session_id: string;
      search_event_id?: string | null;
      results: ChatStreamMetadata["results"];
      parsed_query?: ParsedQuery | null;
    }
  | { type: "token"; text: string }
  | { type: "done"; assistant_message: string; follow_up_suggestions?: string[] }
  | { type: "error"; detail: string };

function parseSseBuffer(
  buffer: string,
  onEvent: (event: StreamEvent) => void,
): string {
  const parts = buffer.split("\n\n");
  const remainder = parts.pop() ?? "";
  for (const part of parts) {
    const line = part
      .split("\n")
      .find((l) => l.startsWith("data: "));
    if (!line) continue;
    try {
      onEvent(JSON.parse(line.slice(6)) as StreamEvent);
    } catch {
      /* ignore malformed */
    }
  }
  return remainder;
}

export async function sendChatStream(
  message: string,
  sessionId: string | null,
  topK: number,
  minMatchPercent: number,
  callbacks: ChatStreamCallbacks,
  sort?: ResultSort,
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/chat/stream`, withUserHeaders({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      session_id: sessionId,
      top_k: topK,
      min_match_percent: minMatchPercent,
      sort,
    }),
  }));

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? body.message ?? detail;
      if (Array.isArray(detail)) {
        detail = detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join("; ");
      }
    } catch {
      /* ignore */
    }
    callbacks.onError(String(detail));
    return;
  }

  const reader = res.body?.getReader();
  if (!reader) {
    callbacks.onError("No response body");
    return;
  }

  const decoder = new TextDecoder();
  let buffer = "";
  let streamError: string | null = null;

  const handleEvent = (event: StreamEvent) => {
    switch (event.type) {
      case "metadata":
        callbacks.onMetadata({
          session_id: event.session_id,
          search_event_id: event.search_event_id ?? null,
          results: event.results,
          parsed_query: event.parsed_query ?? null,
        });
        break;
      case "token":
        callbacks.onToken(event.text);
        break;
      case "done":
        callbacks.onDone(
          event.assistant_message,
          event.follow_up_suggestions ?? [],
        );
        break;
      case "error":
        streamError = event.detail;
        break;
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    buffer = parseSseBuffer(buffer, handleEvent);
    if (streamError) break;
  }
  if (!streamError) {
    buffer += decoder.decode();
    parseSseBuffer(buffer + "\n\n", handleEvent);
  }
  if (streamError) {
    callbacks.onError(streamError);
  }
}


export interface IngestFlags {
  skipCaption: boolean;
  skipOcr: boolean;
  force: boolean;
  workers?: number;
}


export async function createEmptyIngestJob(
  flags: IngestFlags,
  options: { signal?: AbortSignal } = {},
): Promise<IngestJob> {
  const form = new FormData();
  form.append("skip_caption", String(flags.skipCaption));
  form.append("skip_ocr", String(flags.skipOcr));
  form.append("force", String(flags.force));
  if (flags.workers != null) form.append("workers", String(flags.workers));
  form.append("start", "false");
  const key = getAdminApiKey();
  if (!key) {
    throw new Error("Admin API key required for ingest (set in Admin settings)");
  }
  return request<IngestJob>("/api/ingest/jobs", {
    method: "POST",
    headers: { "X-Admin-Api-Key": key },
    body: form,
    signal: options.signal,
  });
}

class UploadRequestError extends Error {
  constructor(
    message: string,
    readonly retryable: boolean,
  ) {
    super(message);
    this.name = "UploadRequestError";
  }
}

interface UploadBatchOptions {
  signal?: AbortSignal;
  timeoutMs: number;
  onByteProgress?: (loaded: number, total: number) => void;
}

function uploadIngestJobBatch(
  jobId: string,
  batchId: string,
  files: File[],
  options: UploadBatchOptions,
): Promise<IngestJob> {
  const key = getAdminApiKey();
  if (!key) {
    return Promise.reject(
      new Error("Admin API key required for ingest (set in Admin settings)"),
    );
  }
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const form = new FormData();
    for (const file of files) form.append("files", file);
    form.append("batch_id", batchId);
    const abort = () => xhr.abort();
    const cleanup = () => options.signal?.removeEventListener("abort", abort);

    xhr.open(
      "POST",
      `${API_BASE}/api/ingest/jobs/${encodeURIComponent(jobId)}/files`,
    );
    xhr.setRequestHeader("X-Admin-Api-Key", key);
    xhr.timeout = options.timeoutMs;
    xhr.upload.onprogress = (event) => {
      const fallbackTotal = files.reduce((sum, file) => sum + file.size, 0);
      options.onByteProgress?.(
        event.loaded,
        event.lengthComputable ? event.total : fallbackTotal,
      );
    };
    xhr.onload = () => {
      cleanup();
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as IngestJob);
        } catch {
          reject(new UploadRequestError("Upload returned invalid JSON.", false));
        }
        return;
      }
      let detail = xhr.statusText || `HTTP ${xhr.status}`;
      try {
        const body = JSON.parse(xhr.responseText) as { detail?: unknown };
        if (body.detail) detail = String(body.detail);
      } catch {
        /* use status text */
      }
      reject(
        new UploadRequestError(
          detail,
          xhr.status === 408 || xhr.status === 429 || xhr.status >= 500,
        ),
      );
    };
    xhr.onerror = () => {
      cleanup();
      reject(new UploadRequestError("Network error while uploading batch.", true));
    };
    xhr.ontimeout = () => {
      cleanup();
      reject(new UploadRequestError("Upload batch timed out.", true));
    };
    xhr.onabort = () => {
      cleanup();
      reject(new DOMException("Upload cancelled.", "AbortError"));
    };
    if (options.signal?.aborted) {
      reject(new DOMException("Upload cancelled.", "AbortError"));
      return;
    }
    options.signal?.addEventListener("abort", abort, { once: true });
    xhr.send(form);
  });
}


export async function startIngestJob(
  jobId: string,
  options: { signal?: AbortSignal } = {},
): Promise<IngestJob> {
  const key = getAdminApiKey();
  if (!key) {
    throw new Error("Admin API key required for ingest (set in Admin settings)");
  }
  return request<IngestJob>(`/api/ingest/jobs/${encodeURIComponent(jobId)}/start`, {
    method: "POST",
    headers: { "X-Admin-Api-Key": key },
    signal: options.signal,
  });
}

export async function fetchIngestJob(jobId: string): Promise<IngestJob> {
  const key = getAdminApiKey();
  if (!key) throw new Error("Admin API key required");
  return request<IngestJob>(`/api/ingest/jobs/${encodeURIComponent(jobId)}`, {
    headers: { "X-Admin-Api-Key": key },
  });
}

export interface BatchedIngestProgress {
  batchIndex: number;
  batchCount: number;
  filesDone: number;
  filesTotal: number;
  lastMessage?: string;
}

export interface IngestJobUploadProgress {
  phase: "uploading";
  batchIndex: number;
  batchCount: number;
  batchesDone: number;
  activeBatches: number;
  retryingBatches: number;
  filesDone: number;
  filesTotal: number;
  bytesDone: number;
  bytesTotal: number;
  jobId?: string;
}

interface PresignedIngestUpload {
  file_id: string;
  filename: string;
  size: number;
  url: string;
  headers: Record<string, string>;
}

interface S3IngestJobResponse {
  job: IngestJob;
  uploads: PresignedIngestUpload[];
  expires_in: number;
}

export class DirectS3UnavailableError extends Error {
  constructor() {
    super("Direct S3 uploads are not enabled");
    this.name = "DirectS3UnavailableError";
  }
}

function putPresignedFile(
  upload: PresignedIngestUpload,
  file: File,
  options: UploadBatchOptions,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const abort = () => xhr.abort();
    const cleanup = () => options.signal?.removeEventListener("abort", abort);
    xhr.open("PUT", upload.url);
    for (const [name, value] of Object.entries(upload.headers)) {
      xhr.setRequestHeader(name, value);
    }
    xhr.timeout = options.timeoutMs;
    xhr.upload.onprogress = (event) => {
      options.onByteProgress?.(
        event.loaded,
        event.lengthComputable ? event.total : file.size,
      );
    };
    xhr.onload = () => {
      cleanup();
      if (xhr.status >= 200 && xhr.status < 300) resolve();
      else reject(new UploadRequestError(`S3 upload returned HTTP ${xhr.status}`, xhr.status >= 500));
    };
    xhr.onerror = () => {
      cleanup();
      reject(new UploadRequestError("Network error while uploading to S3.", true));
    };
    xhr.ontimeout = () => {
      cleanup();
      reject(new UploadRequestError("S3 upload timed out.", true));
    };
    xhr.onabort = () => {
      cleanup();
      reject(new DOMException("Upload cancelled.", "AbortError"));
    };
    if (options.signal?.aborted) {
      reject(new DOMException("Upload cancelled.", "AbortError"));
      return;
    }
    options.signal?.addEventListener("abort", abort, { once: true });
    xhr.send(file);
  });
}

export async function createIngestJobDirectS3(
  files: File[],
  flags: IngestFlags,
  options: {
    concurrency?: number;
    maxRetries?: number;
    timeoutMs?: number;
    onProgress?: (p: IngestJobUploadProgress) => void;
    signal?: AbortSignal;
  } = {},
): Promise<IngestJob> {
  const supported = filterSupportedFiles(files);
  if (!supported.length) throw new Error("No supported files selected.");
  const key = getAdminApiKey();
  if (!key) throw new Error("Admin API key required for ingest (set in Admin settings)");

  const response = await fetch(`${API_BASE}/api/ingest/jobs/s3`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Admin-Api-Key": key,
    },
    body: JSON.stringify({
      files: supported.map((file) => ({
        filename: file.name,
        size: file.size,
        content_type: file.type || null,
      })),
      skip_caption: flags.skipCaption,
      skip_ocr: flags.skipOcr,
      force: flags.force,
      workers: flags.workers,
    }),
    signal: options.signal,
  });
  if (response.status === 409) throw new DirectS3UnavailableError();
  if (!response.ok) {
    const body = await response.json().catch(() => ({})) as { detail?: string };
    throw new Error(body.detail || `Could not prepare S3 upload (HTTP ${response.status})`);
  }
  const prepared = await response.json() as S3IngestJobResponse;
  if (prepared.uploads.length !== supported.length) {
    throw new Error("Upload manifest did not match selected files.");
  }

  const concurrency = Math.max(1, options.concurrency ?? 4);
  const maxRetries = Math.max(0, options.maxRetries ?? 3);
  const timeoutMs = Math.max(1_000, options.timeoutMs ?? 600_000);
  const completed = new Set<number>();
  const active = new Set<number>();
  const retrying = new Set<number>();
  const loaded = new Map<number, number>();
  const bytesTotal = supported.reduce((sum, file) => sum + file.size, 0);
  const emit = () => options.onProgress?.({
    phase: "uploading",
    batchIndex: completed.size,
    batchCount: supported.length,
    batchesDone: completed.size,
    activeBatches: active.size,
    retryingBatches: retrying.size,
    filesDone: completed.size,
    filesTotal: supported.length,
    bytesDone: supported.reduce(
      (sum, file, index) => sum + (completed.has(index) ? file.size : Math.min(loaded.get(index) ?? 0, file.size)),
      0,
    ),
    bytesTotal,
    jobId: prepared.job.job_id,
  });
  emit();

  let next = 0;
  const uploadOne = async (index: number) => {
    for (let attempt = 0; ; attempt += 1) {
      active.add(index);
      loaded.set(index, 0);
      emit();
      try {
        await putPresignedFile(prepared.uploads[index]!, supported[index]!, {
          signal: options.signal,
          timeoutMs,
          onByteProgress: (value) => {
            loaded.set(index, value);
            emit();
          },
        });
        completed.add(index);
        retrying.delete(index);
        return;
      } catch (error) {
        if (
          options.signal?.aborted ||
          !(error instanceof UploadRequestError) ||
          !error.retryable ||
          attempt >= maxRetries
        ) {
          throw new Error(`File ${index + 1}/${supported.length} (${supported[index]!.name}) failed: ${error instanceof Error ? error.message : String(error)}`);
        }
        retrying.add(index);
        emit();
        await new Promise((resolve) => globalThis.setTimeout(resolve, 500 * 2 ** attempt));
      } finally {
        active.delete(index);
        emit();
      }
    }
  };
  const worker = async () => {
    while (true) {
      const index = next++;
      if (index >= supported.length) return;
      await uploadOne(index);
    }
  };
  await Promise.all(Array.from(
    { length: Math.min(concurrency, supported.length) },
    () => worker(),
  ));
  return request<IngestJob>(
    `/api/ingest/jobs/${encodeURIComponent(prepared.job.job_id)}/s3/finalize`,
    {
      method: "POST",
      headers: { "X-Admin-Api-Key": key },
      signal: options.signal,
    },
  );
}

export async function createIngestJobBatched(
  files: File[],
  flags: IngestFlags,
  options: {
    batchSize?: number;
    concurrency?: number;
    maxRetries?: number;
    timeoutMs?: number;
    onProgress?: (p: IngestJobUploadProgress) => void;
    signal?: AbortSignal;
  } = {},
): Promise<IngestJob> {
  const batchSize = Math.max(1, options.batchSize ?? 5);
  const concurrency = Math.max(1, options.concurrency ?? 3);
  const maxRetries = Math.max(0, options.maxRetries ?? 2);
  const timeoutMs = Math.max(1_000, options.timeoutMs ?? 180_000);
  const supported = filterSupportedFiles(files);
  if (supported.length === 0) {
    throw new Error("No supported files selected.");
  }
  const batches: Array<{ id: string; files: File[]; bytes: number }> = [];
  for (let i = 0; i < supported.length; i += batchSize) {
    const batchFiles = supported.slice(i, i + batchSize);
    batches.push({
      id:
        typeof crypto !== "undefined" && "randomUUID" in crypto
          ? crypto.randomUUID()
          : `${Date.now()}-${i}-${Math.random().toString(36).slice(2)}`,
      files: batchFiles,
      bytes: batchFiles.reduce((sum, file) => sum + file.size, 0),
    });
  }

  const job = await createEmptyIngestJob(flags, { signal: options.signal });
  const completed = new Set<number>();
  const active = new Set<number>();
  const retrying = new Set<number>();
  const loadedByBatch = new Map<number, number>();
  const bytesTotal = batches.reduce((sum, batch) => sum + batch.bytes, 0);
  const emitProgress = () => {
    const filesDone = [...completed].reduce(
      (sum, index) => sum + batches[index]!.files.length,
      0,
    );
    const bytesDone = batches.reduce((sum, batch, index) => {
      if (completed.has(index)) return sum + batch.bytes;
      return sum + Math.min(loadedByBatch.get(index) ?? 0, batch.bytes);
    }, 0);
    options.onProgress?.({
      phase: "uploading",
      batchIndex: completed.size,
      batchCount: batches.length,
      batchesDone: completed.size,
      activeBatches: active.size,
      retryingBatches: retrying.size,
      filesDone,
      filesTotal: supported.length,
      bytesDone,
      bytesTotal,
      jobId: job.job_id,
    });
  };
  emitProgress();

  const uploadAbort = new AbortController();
  const abortUploads = () => uploadAbort.abort();
  if (options.signal?.aborted) uploadAbort.abort();
  options.signal?.addEventListener("abort", abortUploads, { once: true });
  let nextBatch = 0;
  const uploadOne = async (index: number) => {
    const batch = batches[index]!;
    for (let attempt = 0; ; attempt += 1) {
      active.add(index);
      loadedByBatch.set(index, 0);
      emitProgress();
      try {
        await uploadIngestJobBatch(job.job_id, batch.id, batch.files, {
          signal: uploadAbort.signal,
          timeoutMs,
          onByteProgress: (loaded) => {
            loadedByBatch.set(index, loaded);
            emitProgress();
          },
        });
        completed.add(index);
        retrying.delete(index);
        return;
      } catch (error) {
        if (
          uploadAbort.signal.aborted ||
          !(error instanceof UploadRequestError) ||
          !error.retryable ||
          attempt >= maxRetries
        ) {
          throw new Error(
            `Batch ${index + 1}/${batches.length} failed: ${
              error instanceof Error ? error.message : String(error)
            }`,
          );
        }
        retrying.add(index);
        emitProgress();
        await new Promise<void>((resolve, reject) => {
          const timer = globalThis.setTimeout(resolve, 500 * 2 ** attempt);
          uploadAbort.signal.addEventListener(
            "abort",
            () => {
              globalThis.clearTimeout(timer);
              reject(new DOMException("Upload cancelled.", "AbortError"));
            },
            { once: true },
          );
        });
      } finally {
        active.delete(index);
        emitProgress();
      }
    }
  };
  const worker = async () => {
    while (true) {
      const index = nextBatch++;
      if (index >= batches.length) return;
      await uploadOne(index);
    }
  };

  try {
    await Promise.all(
      Array.from({ length: Math.min(concurrency, batches.length) }, () => worker()),
    );
  } catch (error) {
    uploadAbort.abort();
    throw error;
  } finally {
    options.signal?.removeEventListener("abort", abortUploads);
  }
  return startIngestJob(job.job_id, { signal: options.signal });
}


export async function fetchCorpusCatalog(
  limit = 40,
  sort?: ResultSort,
): Promise<CorpusCatalogResponse> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (sort) params.set("sort", sort);
  return request<CorpusCatalogResponse>(`/api/corpus/catalog?${params.toString()}`);
}

export type SimilarityAxis = "balanced" | "subject" | "style" | "layout";

export async function searchSimilarByImage(
  imageFile: File,
  sessionId: string | null,
  topK: number,
  minMatchPercent: number,
  similarityAxis: SimilarityAxis = "balanced",
  sort?: ResultSort,
): Promise<SimilarResponse> {
  const form = new FormData();
  form.append("file", imageFile);
  form.append("top_k", String(topK));
  form.append("min_match_percent", String(minMatchPercent));
  form.append("similarity_axis", similarityAxis);
  if (sort) form.append("sort", sort);
  if (sessionId) form.append("session_id", sessionId);
  return request<SimilarResponse>("/api/similar", {
    method: "POST",
    body: form,
  });
}

export async function searchSimilarByImageId(
  imageId: string,
  sessionId: string | null,
  topK: number,
  minMatchPercent: number,
  similarityAxis: SimilarityAxis = "balanced",
  sort?: ResultSort,
): Promise<SimilarResponse> {
  return request<SimilarResponse>("/api/similar", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_id: imageId,
      session_id: sessionId,
      top_k: topK,
      min_match_percent: minMatchPercent,
      similarity_axis: similarityAxis,
      sort,
    }),
  });
}

export async function sendSimilar(
  imageId: string,
  sessionId: string | null,
  topK: number,
  minMatchPercent: number,
  similarityAxis: SimilarityAxis = "balanced",
  sort?: ResultSort,
): Promise<SimilarResponse> {
  return searchSimilarByImageId(
    imageId,
    sessionId,
    topK,
    minMatchPercent,
    similarityAxis,
    sort,
  );
}
