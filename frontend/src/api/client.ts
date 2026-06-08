import { getAdminApiKey } from "./adminClient";
import { getUserId } from "./telemetry";
import type {
  ChatResponse,
  ChatStreamCallbacks,
  ChatStreamMetadata,
  CorpusCatalogResponse,
  IngestResponse,
  ParsedQuery,
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
  recentTitles: string[],
  limit = 4,
): Promise<SuggestionsResponse> {
  return request<SuggestionsResponse>("/api/suggestions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ recent_titles: recentTitles, limit }),
  });
}

export async function sendChat(
  message: string,
  sessionId: string | null,
  topK: number,
  minMatchPercent: number,
): Promise<ChatResponse> {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      session_id: sessionId,
      top_k: topK,
      min_match_percent: minMatchPercent,
    }),
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
  | { type: "done"; assistant_message: string }
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
): Promise<void> {
  const res = await fetch(`${API_BASE}/api/chat/stream`, withUserHeaders({
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message,
      session_id: sessionId,
      top_k: topK,
      min_match_percent: minMatchPercent,
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
        callbacks.onDone(event.assistant_message);
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

export async function resetSession(sessionId: string): Promise<void> {
  await request("/api/session/reset", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
}

export interface IngestFlags {
  skipCaption: boolean;
  skipOcr: boolean;
  force: boolean;
  workers?: number;
}

export async function ingestFiles(
  files: File[],
  flags: IngestFlags,
): Promise<IngestResponse> {
  const form = new FormData();
  for (const f of files) {
    form.append("files", f);
  }
  form.append("skip_caption", String(flags.skipCaption));
  form.append("skip_ocr", String(flags.skipOcr));
  form.append("force", String(flags.force));
  if (flags.workers != null) {
    form.append("workers", String(flags.workers));
  }
  const key = getAdminApiKey();
  if (!key) {
    throw new Error("Admin API key required for ingest (set in Admin settings)");
  }
  return request<IngestResponse>("/api/ingest", {
    method: "POST",
    headers: { "X-Admin-Api-Key": key },
    body: form,
  });
}

export interface BatchedIngestProgress {
  batchIndex: number;
  batchCount: number;
  filesDone: number;
  filesTotal: number;
  lastMessage?: string;
}

export async function ingestFilesBatched(
  files: File[],
  flags: IngestFlags,
  options: {
    batchSize?: number;
    onProgress?: (p: BatchedIngestProgress) => void;
  } = {},
): Promise<IngestResponse> {
  const batchSize = Math.max(1, options.batchSize ?? 25);
  const supported = filterSupportedFiles(files);
  if (supported.length === 0) {
    throw new Error("No supported files selected.");
  }
  const batches: File[][] = [];
  for (let i = 0; i < supported.length; i += batchSize) {
    batches.push(supported.slice(i, i + batchSize));
  }
  let last: IngestResponse = {
    message: "",
    indexed_count: 0,
    stats: {},
  };
  const messages: string[] = [];
  for (let i = 0; i < batches.length; i++) {
    const batch = batches[i]!;
    options.onProgress?.({
      batchIndex: i + 1,
      batchCount: batches.length,
      filesDone: Math.min((i + 1) * batchSize, supported.length),
      filesTotal: supported.length,
    });
    last = await ingestFiles(batch, flags);
    messages.push(`Batch ${i + 1}/${batches.length}: ${last.message}`);
    options.onProgress?.({
      batchIndex: i + 1,
      batchCount: batches.length,
      filesDone: Math.min((i + 1) * batchSize, supported.length),
      filesTotal: supported.length,
      lastMessage: last.message,
    });
  }
  return {
    ...last,
    message: messages.join("\n\n"),
  };
}

export async function fetchCorpusCatalog(
  limit = 40,
): Promise<CorpusCatalogResponse> {
  return request<CorpusCatalogResponse>(
    `/api/corpus/catalog?limit=${encodeURIComponent(String(limit))}`,
  );
}

export type SimilarityAxis = "balanced" | "subject" | "style" | "layout";

export async function searchSimilarByImage(
  imageFile: File,
  sessionId: string | null,
  topK: number,
  minMatchPercent: number,
  similarityAxis: SimilarityAxis = "balanced",
): Promise<SimilarResponse> {
  const form = new FormData();
  form.append("file", imageFile);
  form.append("top_k", String(topK));
  form.append("min_match_percent", String(minMatchPercent));
  form.append("similarity_axis", similarityAxis);
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
    }),
  });
}

export async function sendSimilar(
  imageId: string,
  sessionId: string | null,
  topK: number,
  minMatchPercent: number,
  similarityAxis: SimilarityAxis = "balanced",
): Promise<SimilarResponse> {
  return searchSimilarByImageId(
    imageId,
    sessionId,
    topK,
    minMatchPercent,
    similarityAxis,
  );
}
