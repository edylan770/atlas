import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./adminClient", () => ({
  getAdminApiKey: () => "test-admin-key",
}));
vi.mock("./telemetry", () => ({
  getUserId: () => null,
}));

import {
  createIngestJobDirectS3,
  createIngestJobBatched,
  type IngestJobUploadProgress,
} from "./client";
import type { IngestJob } from "../types";

class FakeFormData {
  private values = new Map<string, unknown[]>();

  append(name: string, value: unknown) {
    const current = this.values.get(name) ?? [];
    current.push(value);
    this.values.set(name, current);
  }

  get(name: string) {
    return this.values.get(name)?.[0] ?? null;
  }
}

const stagingJob = {
  job_id: "job-upload",
  status: "staging",
  files: [],
  files_total: 0,
  files_done: 0,
  images_seen: 0,
  images_processed: 0,
  options: {},
  stats: {},
  stage_errors: [],
  cancellable: true,
} satisfies IngestJob;

function file(name: string, size = 100): File {
  return { name, size } as File;
}

describe("createIngestJobBatched", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    vi.stubGlobal("FormData", FakeFormData);
  });

  it("limits concurrency, retries with the same batch id, and starts last", async () => {
    const attempts = new Map<string, number>();
    let active = 0;
    let maxActive = 0;
    let startObservedActive = -1;
    let fetchCount = 0;
    let failedBatchId: string | null = null;

    class FakeXMLHttpRequest {
      status = 200;
      statusText = "OK";
      responseText = JSON.stringify(stagingJob);
      timeout = 0;
      upload: { onprogress: ((event: ProgressEvent) => void) | null } = {
        onprogress: null,
      };
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      ontimeout: (() => void) | null = null;
      onabort: (() => void) | null = null;
      private aborted = false;

      open() {}
      setRequestHeader() {}
      send(body: FakeFormData) {
        const batchId = String(body.get("batch_id"));
        failedBatchId ??= batchId;
        const attempt = (attempts.get(batchId) ?? 0) + 1;
        attempts.set(batchId, attempt);
        active += 1;
        maxActive = Math.max(maxActive, active);
        this.upload.onprogress?.({ loaded: 500, total: 500, lengthComputable: true } as ProgressEvent);
        globalThis.setTimeout(() => {
          if (this.aborted) return;
          active -= 1;
          if (batchId === failedBatchId && attempt === 1) {
            this.onerror?.();
          } else {
            this.onload?.();
          }
        }, 5);
      }
      abort() {
        if (this.aborted) return;
        this.aborted = true;
        active = Math.max(0, active - 1);
        this.onabort?.();
      }
    }

    vi.stubGlobal("XMLHttpRequest", FakeXMLHttpRequest);
    vi.stubGlobal(
      "fetch",
      vi.fn(async (_url: string) => {
        fetchCount += 1;
        if (fetchCount === 2) startObservedActive = active;
        return {
          ok: true,
          status: fetchCount === 1 ? 202 : 200,
          statusText: "OK",
          json: async () =>
            fetchCount === 1
              ? stagingJob
              : { ...stagingJob, status: "queued", files_total: 16 },
        } as Response;
      }),
    );
    const progress: IngestJobUploadProgress[] = [];

    const result = await createIngestJobBatched(
      Array.from({ length: 16 }, (_, index) => file(`${index}.png`)),
      { skipCaption: false, skipOcr: false, force: false },
      {
        batchSize: 5,
        concurrency: 3,
        maxRetries: 1,
        onProgress: (value) => progress.push({ ...value }),
      },
    );

    expect(result.status).toBe("queued");
    expect(maxActive).toBe(3);
    expect(attempts.size).toBe(4);
    expect([...attempts.values()].sort()).toEqual([1, 1, 1, 2]);
    expect(startObservedActive).toBe(0);
    expect(progress[progress.length - 1]).toMatchObject({
      batchesDone: 4,
      activeBatches: 0,
      filesDone: 16,
      filesTotal: 16,
    });
  });

  it("aborts all active upload requests", async () => {
    let aborted = 0;

    class HangingXMLHttpRequest {
      timeout = 0;
      upload = { onprogress: null };
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      ontimeout: (() => void) | null = null;
      onabort: (() => void) | null = null;
      open() {}
      setRequestHeader() {}
      send() {}
      abort() {
        aborted += 1;
        this.onabort?.();
      }
    }

    vi.stubGlobal("XMLHttpRequest", HangingXMLHttpRequest);
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        status: 202,
        statusText: "OK",
        json: async () => stagingJob,
      })),
    );
    const controller = new AbortController();
    const promise = createIngestJobBatched(
      [file("a.png"), file("b.png")],
      { skipCaption: false, skipOcr: false, force: false },
      { batchSize: 1, concurrency: 2, signal: controller.signal },
    );
    await new Promise((resolve) => globalThis.setTimeout(resolve, 0));
    controller.abort();

    await expect(promise).rejects.toThrow("Batch");
    expect(aborted).toBe(2);
  });
});

describe("createIngestJobDirectS3", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("uploads 235 files with bounded concurrency and finalizes last", async () => {
    const selected = Array.from({ length: 235 }, (_, index) =>
      file(`image-${index}.png`, 100 + index),
    );
    let active = 0;
    let maxActive = 0;
    let completed = 0;
    let finalizeObservedCompleted = -1;

    class DirectUploadXMLHttpRequest {
      status = 200;
      timeout = 0;
      upload: { onprogress: ((event: ProgressEvent) => void) | null } = {
        onprogress: null,
      };
      onload: (() => void) | null = null;
      onerror: (() => void) | null = null;
      ontimeout: (() => void) | null = null;
      onabort: (() => void) | null = null;
      open() {}
      setRequestHeader() {}
      send(body: File) {
        active += 1;
        maxActive = Math.max(maxActive, active);
        this.upload.onprogress?.({
          loaded: body.size,
          total: body.size,
          lengthComputable: true,
        } as ProgressEvent);
        globalThis.setTimeout(() => {
          active -= 1;
          completed += 1;
          this.onload?.();
        }, 0);
      }
      abort() {
        this.onabort?.();
      }
    }
    vi.stubGlobal("XMLHttpRequest", DirectUploadXMLHttpRequest);

    let fetchCount = 0;
    vi.stubGlobal("fetch", vi.fn(async () => {
      fetchCount += 1;
      if (fetchCount === 1) {
        return {
          ok: true,
          status: 202,
          json: async () => ({
            job: { ...stagingJob, files_total: selected.length },
            uploads: selected.map((item, index) => ({
              file_id: String(index),
              filename: item.name,
              size: item.size,
              url: `https://s3.test/${index}`,
              headers: { "Content-Type": "image/png" },
            })),
            expires_in: 3600,
          }),
        } as Response;
      }
      finalizeObservedCompleted = completed;
      return {
        ok: true,
        status: 200,
        json: async () => ({ ...stagingJob, status: "queued", files_total: selected.length }),
      } as Response;
    }));

    const result = await createIngestJobDirectS3(
      selected,
      { skipCaption: false, skipOcr: false, force: false },
      { concurrency: 4 },
    );

    expect(result.status).toBe("queued");
    expect(maxActive).toBe(4);
    expect(completed).toBe(235);
    expect(finalizeObservedCompleted).toBe(235);
  });
});
