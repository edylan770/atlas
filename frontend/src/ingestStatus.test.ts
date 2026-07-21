import { describe, expect, it } from "vitest";

import {
  formatIngestPhase,
  heartbeatAgeSeconds,
  isMissingIngestJobError,
  isStaleIngestJob,
} from "./ingestStatus";
import type { IngestJob } from "./types";

function job(overrides: Partial<IngestJob> = {}): IngestJob {
  return {
    job_id: "job-1",
    status: "running",
    files: ["image.png"],
    files_total: 1,
    files_done: 0,
    images_seen: 0,
    images_processed: 0,
    options: {},
    stats: {},
    stage_errors: [],
    cancellable: true,
    ...overrides,
  };
}

describe("ingest status diagnostics", () => {
  it("shows the persisted dependency detail", () => {
    expect(
      formatIngestPhase(
        job({
          phase: "source_blob_write",
          status_detail: "Persisting image.png",
        }),
      ),
    ).toBe("Persisting image.png");
  });

  it("detects a stale active worker heartbeat", () => {
    const now = Date.parse("2026-07-21T12:00:30Z");
    const running = job({ heartbeat_at: "2026-07-21T12:00:00Z" });
    expect(heartbeatAgeSeconds(running.heartbeat_at, now)).toBe(30);
    expect(isStaleIngestJob(running, now)).toBe(true);
    expect(isStaleIngestJob({ ...running, status: "failed" }, now)).toBe(false);
  });

  it("detects a missing ingest job API error", () => {
    expect(isMissingIngestJobError(new Error("ingest job not found"))).toBe(true);
    expect(isMissingIngestJobError(new Error("Not Found"))).toBe(true);
    expect(isMissingIngestJobError(new Error("network timeout"))).toBe(false);
  });
});
