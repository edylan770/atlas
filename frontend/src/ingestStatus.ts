import type { IngestJob } from "./types";

const ACTIVE = new Set(["queued", "running", "cancel_requested"]);

export function formatIngestPhase(job: IngestJob): string {
  if (job.status === "cancel_requested") return "Cancelling…";
  if (job.status === "staging") return "Uploading files…";
  return (
    job.status_detail ||
    job.phase?.replace(/_/g, " ") ||
    (job.status === "queued" ? "Queued" : "Ingesting")
  );
}

export function heartbeatAgeSeconds(
  heartbeatAt?: string | null,
  now = Date.now(),
): number | null {
  if (!heartbeatAt) return null;
  const parsed = Date.parse(heartbeatAt);
  if (Number.isNaN(parsed)) return null;
  return Math.max(0, Math.round((now - parsed) / 1000));
}

export function isStaleIngestJob(
  job: IngestJob,
  now = Date.now(),
  staleAfterSeconds = 20,
): boolean {
  const age = heartbeatAgeSeconds(job.heartbeat_at, now);
  return ACTIVE.has(job.status) && age !== null && age > staleAfterSeconds;
}

export function isMissingIngestJobError(error: unknown): boolean {
  const msg = error instanceof Error ? error.message : String(error);
  return /not found|404/i.test(msg);
}

/** Polling can never succeed without a valid admin key; retrying won't help. */
export function isIngestAuthError(error: unknown): boolean {
  const msg = error instanceof Error ? error.message : String(error);
  return /admin api key required|invalid admin api key|admin api is not configured/i.test(
    msg,
  );
}

export const INGEST_POLL_MAX_FAILURES = 20;

/** Retry delay for the Nth consecutive poll failure (3s doubling, capped at 30s). */
export function ingestPollBackoffMs(consecutiveFailures: number): number {
  return Math.min(30_000, 3000 * 2 ** Math.max(0, consecutiveFailures - 1));
}
