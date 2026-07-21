import type { IngestJob } from "./types";

const ACTIVE = new Set(["queued", "running", "cancel_requested"]);

export function formatIngestPhase(job: IngestJob): string {
  if (job.status === "cancel_requested") return "Cancelling…";
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
