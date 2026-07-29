import { useCallback, useEffect, useRef, useState } from "react";
import {
  fetchCorpusCatalog,
  fetchIngestJob,
  fetchStatus,
  createIngestJobDirectS3,
  createIngestJobBatched,
  DirectS3UnavailableError,
  type IngestJobUploadProgress,
} from "../api/client";
import { cancelIngestJob } from "../api/adminClient";
import { CorpusDrawer } from "../components/CorpusDrawer";
import { formatIngestPhase, heartbeatAgeSeconds, isMissingIngestJobError } from "../ingestStatus";
import { defaultCatalogSort } from "../sortResults";
import type { CatalogItem, ResultSort } from "../types";

const ACTIVE_INGEST_JOB_KEY = "atlas.activeIngestJobId";

function shouldClearIngestFromUrl(): boolean {
  const params = new URLSearchParams(window.location.search);
  if (params.get("clearIngest") === "1") return true;
  const hash = window.location.hash.replace(/^#/, "");
  return hash === "clearIngest";
}

function stripClearIngestFromUrl(): void {
  const url = new URL(window.location.href);
  if (url.searchParams.has("clearIngest")) {
    url.searchParams.delete("clearIngest");
  }
  if (url.hash === "#clearIngest") {
    url.hash = "";
  }
  window.history.replaceState({}, "", `${url.pathname}${url.search}${url.hash}`);
}

/** Cleared once on boot when ?clearIngest=1 or #clearIngest is present. */
let clearedStuckIngestOnBoot = false;

export interface CorpusIngestHostProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

/**
 * Owns corpus drawer ingest/catalog state for the admin shell.
 * Mount once under AdminLayout so it persists across admin tabs.
 */
export function CorpusIngestHost({ open, onOpenChange }: CorpusIngestHostProps) {
  const [skipCaption, setSkipCaption] = useState(false);
  const [skipOcr, setSkipOcr] = useState(false);
  const [force, setForce] = useState(false);
  const [ingestWorkers, setIngestWorkers] = useState(4);
  const [ingesting, setIngesting] = useState(false);
  const [ingestCancelling, setIngestCancelling] = useState(false);
  const [activeIngestJobId, setActiveIngestJobId] = useState<string | null>(() => {
    if (shouldClearIngestFromUrl()) {
      window.localStorage.removeItem(ACTIVE_INGEST_JOB_KEY);
      stripClearIngestFromUrl();
      clearedStuckIngestOnBoot = true;
      return null;
    }
    return window.localStorage.getItem(ACTIVE_INGEST_JOB_KEY);
  });
  const [ingestMessage, setIngestMessage] = useState<string | null>(() =>
    clearedStuckIngestOnBoot
      ? "Cleared stuck ingest state. You can upload again."
      : null,
  );
  const [ingestProgress, setIngestProgress] = useState<{
    filesDone: number;
    filesTotal: number;
    batchLabel: string;
    bytesDone?: number;
    bytesTotal?: number;
  } | null>(null);
  const [catalog, setCatalog] = useState<CatalogItem[]>([]);
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [catalogSortBy, setCatalogSortBy] = useState<ResultSort>(defaultCatalogSort());
  const [indexedCount, setIndexedCount] = useState(0);

  const ingestUploadAbortRef = useRef<AbortController | null>(null);
  const stagingIngestJobIdRef = useRef<string | null>(null);
  const ingestUploadingRef = useRef(false);

  const refreshStatus = useCallback(async () => {
    try {
      const s = await fetchStatus();
      setIndexedCount(s.total_records ?? s.indexed_count);
    } catch {
      /* status is best-effort for catalog refresh triggers */
    }
  }, []);

  const refreshCatalog = useCallback(async () => {
    setCatalogLoading(true);
    setCatalogError(null);
    try {
      const res = await fetchCorpusCatalog(40, catalogSortBy);
      setCatalog(res.items);
    } catch (e) {
      setCatalog([]);
      setCatalogError(e instanceof Error ? e.message : String(e));
    } finally {
      setCatalogLoading(false);
    }
  }, [catalogSortBy]);

  const clearActiveIngest = useCallback(() => {
    window.localStorage.removeItem(ACTIVE_INGEST_JOB_KEY);
    setActiveIngestJobId(null);
    setIngesting(false);
    setIngestCancelling(false);
    setIngestProgress(null);
  }, []);

  useEffect(() => {
    if (!activeIngestJobId) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    const poll = async () => {
      try {
        const job = await fetchIngestJob(activeIngestJobId);
        if (stopped) return;
        const active = ["staging", "queued", "running", "cancel_requested"].includes(
          job.status,
        );
        setIngesting(active);
        setIngestCancelling(job.status === "cancel_requested");
        const browserIsUploading =
          job.status === "staging" && ingestUploadingRef.current;
        if (!browserIsUploading) {
          setIngestProgress(
            active
              ? {
                  filesDone: job.files_done,
                  filesTotal: job.files_total,
                  batchLabel: formatIngestPhase(job),
                }
              : null,
          );
        }
        if (active && !browserIsUploading) {
          const heartbeatAge = heartbeatAgeSeconds(job.heartbeat_at);
          setIngestMessage(
            `Job ${job.job_id}\n${job.status_detail ?? job.phase ?? job.status}` +
              (heartbeatAge == null ? "" : `\nWorker heartbeat: ${heartbeatAge}s ago`),
          );
        }
        if (!active) {
          const processed =
            Number(job.stats.images_added ?? 0) +
            Number(job.stats.images_updated ?? 0);
          setIngestMessage(
            job.status === "succeeded"
              ? `Ingest complete: ${processed} image(s) added or updated.`
              : job.status === "cancelled"
                ? `Ingest cancelled. ${processed} completed image(s) were kept.`
                : `Ingest failed: ${job.error ?? "Unknown error"}`,
          );
          clearActiveIngest();
          await refreshStatus();
          void refreshCatalog();
          return;
        }
        timer = window.setTimeout(() => void poll(), 1000);
      } catch (e) {
        if (stopped) return;
        if (isMissingIngestJobError(e)) {
          setIngestMessage(
            "Cleared a stuck ingest that was no longer on the server. You can upload again.",
          );
          clearActiveIngest();
          return;
        }
        setIngestMessage(e instanceof Error ? e.message : String(e));
        setIngestProgress((prev) =>
          prev ?? { filesDone: 0, filesTotal: 1, batchLabel: "Stuck — job unreachable" },
        );
        timer = window.setTimeout(() => void poll(), 3000);
      }
    };

    setIngesting(true);
    setIngestProgress((prev) =>
      prev ?? { filesDone: 0, filesTotal: 1, batchLabel: "Checking ingest status…" },
    );
    void poll();
    return () => {
      stopped = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [activeIngestJobId, clearActiveIngest, refreshCatalog, refreshStatus]);

  useEffect(() => {
    if (open) {
      void refreshCatalog();
    }
  }, [open, refreshCatalog, indexedCount]);

  const handleIngest = async (files: File[]) => {
    if (!files.length) return;
    setIngesting(true);
    setIngestCancelling(false);
    setIngestMessage(
      "Keep this tab open until the upload finishes. Processing can continue after that.",
    );
    setIngestProgress({
      filesDone: 0,
      filesTotal: files.length,
      batchLabel: "Uploading…",
    });

    const abort = new AbortController();
    ingestUploadAbortRef.current?.abort();
    ingestUploadAbortRef.current = abort;
    stagingIngestJobIdRef.current = null;
    ingestUploadingRef.current = true;

    const onBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", onBeforeUnload);

    try {
      const flags = {
        skipCaption,
        skipOcr,
        force,
        workers: ingestWorkers,
      };
      const onProgress = (p: IngestJobUploadProgress) => {
        if (p.jobId) {
          stagingIngestJobIdRef.current = p.jobId;
          if (activeIngestJobId !== p.jobId) {
            window.localStorage.setItem(ACTIVE_INGEST_JOB_KEY, p.jobId);
            setActiveIngestJobId(p.jobId);
          }
        }
        const retryLabel =
          p.retryingBatches > 0 ? `, ${p.retryingBatches} retrying` : "";
        setIngestProgress({
          filesDone: p.filesDone,
          filesTotal: p.filesTotal,
          bytesDone: p.bytesDone,
          bytesTotal: p.bytesTotal,
          batchLabel:
            `Uploading to S3: ${p.filesDone}/${p.filesTotal} files ` +
            `(${p.activeBatches} active${retryLabel})`,
        });
      };
      let job;
      try {
        job = await createIngestJobDirectS3(files, flags, {
          concurrency: 4,
          signal: abort.signal,
          onProgress,
        });
      } catch (error) {
        if (!(error instanceof DirectS3UnavailableError)) throw error;
        setIngestMessage(
          "Direct S3 upload is unavailable; using server staging. Keep this tab open until upload finishes.",
        );
        job = await createIngestJobBatched(files, flags, {
          batchSize: 5,
          concurrency: 3,
          timeoutMs: 600_000,
          signal: abort.signal,
          onProgress,
        });
      }
      stagingIngestJobIdRef.current = null;
      window.localStorage.setItem(ACTIVE_INGEST_JOB_KEY, job.job_id);
      setActiveIngestJobId(job.job_id);
      const stageNote =
        job.stage_errors?.length > 0
          ? `\n${job.stage_errors.length} file(s) could not be staged.`
          : "";
      setIngestMessage(
        `Ingest queued (${job.files_total} file(s)). You may close this tab.${stageNote}`,
      );
    } catch (e) {
      const stagingId = stagingIngestJobIdRef.current;
      if (stagingId) {
        try {
          await cancelIngestJob(stagingId);
        } catch {
          /* best-effort cleanup after a failed or cancelled upload */
        }
      }
      if (abort.signal.aborted) {
        setIngestMessage("Upload cancelled.");
      } else {
        setIngestMessage(e instanceof Error ? e.message : String(e));
      }
      clearActiveIngest();
      stagingIngestJobIdRef.current = null;
    } finally {
      ingestUploadingRef.current = false;
      if (ingestUploadAbortRef.current === abort) {
        ingestUploadAbortRef.current = null;
      }
      window.removeEventListener("beforeunload", onBeforeUnload);
    }
  };

  const handleCancelIngest = async () => {
    if (ingestCancelling) return;
    setIngestCancelling(true);

    if (ingestUploadingRef.current) {
      ingestUploadAbortRef.current?.abort();
      const stagingId = stagingIngestJobIdRef.current;
      if (stagingId) {
        try {
          await cancelIngestJob(stagingId);
        } catch {
          /* best-effort */
        }
        stagingIngestJobIdRef.current = null;
      }
      setIngestMessage("Upload cancelled.");
      clearActiveIngest();
      return;
    }

    if (!activeIngestJobId) {
      setIngestMessage("Cleared stuck ingest status. You can upload again.");
      clearActiveIngest();
      return;
    }

    try {
      const job = await cancelIngestJob(activeIngestJobId);
      if (["cancelled", "succeeded", "failed"].includes(job.status)) {
        const processed =
          Number(job.stats.images_added ?? 0) + Number(job.stats.images_updated ?? 0);
        setIngestMessage(
          job.status === "cancelled"
            ? `Ingest cancelled. ${processed} completed image(s) were kept.`
            : job.status === "succeeded"
              ? `Ingest complete: ${processed} image(s) added or updated.`
              : `Ingest failed: ${job.error ?? "Unknown error"}`,
        );
        clearActiveIngest();
        await refreshStatus();
        void refreshCatalog();
        return;
      }
      setIngestMessage(
        job.status_detail ?? "Cancellation requested. Waiting for the worker to stop…",
      );
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setIngestMessage(
        isMissingIngestJobError(e)
          ? "Cleared a stuck ingest that was no longer on the server. You can upload again."
          : `Could not cancel on the server (${msg}). Cleared the local ingest lock so you can upload again.`,
      );
      clearActiveIngest();
    }
  };

  return (
    <CorpusDrawer
      open={open}
      onClose={() => onOpenChange(false)}
      skipCaption={skipCaption}
      skipOcr={skipOcr}
      force={force}
      ingestWorkers={ingestWorkers}
      onSkipCaptionChange={setSkipCaption}
      onSkipOcrChange={setSkipOcr}
      onForceChange={setForce}
      onIngestWorkersChange={setIngestWorkers}
      onIngest={handleIngest}
      onCancelIngest={() => void handleCancelIngest()}
      ingestMessage={ingestMessage}
      ingesting={ingesting}
      ingestCancelling={ingestCancelling}
      ingestProgress={ingestProgress}
      activeIngestJobId={activeIngestJobId}
      catalog={catalog}
      catalogLoading={catalogLoading}
      catalogError={catalogError}
      catalogSortBy={catalogSortBy}
      onCatalogSortChange={setCatalogSortBy}
    />
  );
}
