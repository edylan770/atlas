import { useCallback, useEffect, useState } from "react";
import { Route, Routes, Link } from "react-router-dom";
import {
  fetchAnalyticsSummary,
  fetchAudit,
  fetchCorpusHealth,
  fetchIngestDiagnostics,
  fetchIngestJobs,
  runIngestPreflight,
  reconcileIndex,
  repairIndex,
  listIndexBackups,
  backupIndex,
  restoreIndex,
  purgeUnrecoverable,
  fetchOrphanBlobs,
  purgeOrphanBlobs,
  fetchCorpusImages,
  fetchDeleted,
  fetchDuplicateClusters,
  fetchOrphans,
  fetchSearchQuality,
  regenerateCaption,
  reindexImage,
  repairCaptions,
  regenerateMissingThumbnails,
  restoreImage,
  softDeleteImage,
  cancelIngestJob,
  fetchPendingEdits,
  acceptPendingEdit,
  declinePendingEdit,
  type AnalyticsSummary,
  type CaptionQualityFilter,
  type CorpusHealth,
  type CorpusImage,
  type IndexBackupInfo,
  type IngestDiagnostics,
  type IngestPreflight,
  type PendingEditItem,
  type SearchQualityItem,
  type SearchQualityLists,
} from "../api/adminClient";
import { SortSelect } from "../components/SortSelect";
import { heartbeatAgeSeconds, isStaleIngestJob } from "../ingestStatus";
import { defaultCatalogSort } from "../sortResults";
import type { ResultSort } from "../types";
import type { IngestJob } from "../types";
import { AdminLayout } from "./AdminShell";

function queryTooltip(row: SearchQualityItem): string {
  const user = row.user_message ?? row.query_text ?? "";
  const semantic = row.parsed_semantic_query ?? "";
  const parts: string[] = [];
  if (user) parts.push(`User: ${user}`);
  if (semantic && semantic !== user) parts.push(`Interpreted: ${semantic}`);
  if (row.timings && Object.keys(row.timings).length > 0) {
    const stages = Object.entries(row.timings)
      .sort((a, b) => b[1] - a[1])
      .map(([k, v]) => `${k}=${Math.round(v)}ms`)
      .join(", ");
    parts.push(`Timing: ${stages}`);
  } else if (row.total_ms != null) {
    parts.push(`Total: ${Math.round(row.total_ms)}ms`);
  }
  if (row.timing_log) parts.push(`Log: ${row.timing_log}`);
  return parts.join("\n") || row.display_query;
}

function formatMs(ms: number | null | undefined): string {
  if (ms == null || Number.isNaN(ms)) return "—";
  return `${Math.round(ms)}`;
}

function StatCard({
  label,
  value,
  subtitle,
}: {
  label: string;
  value: string | number;
  subtitle?: string;
}) {
  return (
    <div className="rounded-xl bg-white p-4 shadow-sm ring-1 ring-navy-200">
      <p className="text-xs font-medium uppercase tracking-wide text-navy-500">{label}</p>
      <p className="mt-1 text-2xl font-semibold text-navy-900">{value}</p>
      {subtitle && <p className="mt-0.5 text-xs text-navy-500">{subtitle}</p>}
    </div>
  );
}

function DashboardPage() {
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [health, setHealth] = useState<CorpusHealth | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [days, setDays] = useState<7 | 30 | 90>(90);

  useEffect(() => {
    setSummary(null);
    fetchAnalyticsSummary(days)
      .then(setSummary)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [days]);

  useEffect(() => {
    fetchCorpusHealth()
      .then(setHealth)
      .catch((e) => setHealthError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error) return <p className="text-red-600">{error}</p>;
  if (!summary) return <p className="text-navy-500">Loading…</p>;

  const corpusSubtitle = health
    ? `of ${health.total_images} indexed`
    : healthError
      ? "unavailable"
      : "…";

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-navy-900">Dashboard</h2>
          <p className="text-xs text-navy-500">Last {days} days</p>
        </div>
        <div className="flex gap-1 rounded-lg bg-navy-100 p-1" role="group" aria-label="Time window">
          {([7, 30, 90] as const).map((n) => (
            <button
              key={n}
              type="button"
              onClick={() => setDays(n)}
              className={
                days === n
                  ? "rounded-md bg-white px-3 py-1 text-xs font-medium text-navy-900 shadow-sm"
                  : "rounded-md px-3 py-1 text-xs font-medium text-navy-600 hover:text-navy-900"
              }
            >
              {n}d
            </button>
          ))}
        </div>
      </div>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Total searches" value={summary.total_searches} />
        <StatCard
          label="Zero-result rate"
          value={`${(summary.zero_result_rate * 100).toFixed(1)}%`}
        />
        <StatCard
          label="Weak-result rate"
          value={`${(summary.weak_result_rate * 100).toFixed(1)}%`}
        />
        <StatCard
          label="No-interaction rate"
          value={`${(summary.no_interaction_rate * 100).toFixed(1)}%`}
        />
        <StatCard label="Interactions" value={summary.interaction_count} />
        <StatCard
          label="Interaction rate"
          value={`${(summary.interaction_rate * 100).toFixed(1)}%`}
        />
        <StatCard
          label="Failed captions"
          value={health?.failed_caption_count ?? "…"}
          subtitle={corpusSubtitle}
        />
        <StatCard
          label="Weak captions"
          value={health?.weak_caption_count ?? "…"}
          subtitle={corpusSubtitle}
        />
        <StatCard
          label="Thumbnails"
          value={
            health
              ? `${health.thumb_count ?? 0}/${health.total_records ?? health.total_images}`
              : "…"
          }
          subtitle={
            health
              ? (health.missing_thumb_count ?? 0) > 0
                ? `${health.missing_thumb_count} missing`
                : "complete"
              : corpusSubtitle
          }
        />
      </div>
      {healthError && (
        <p className="text-xs text-amber-700">Caption health: {healthError}</p>
      )}
      <p className="text-xs text-navy-500">
        Weak threshold (raw score): {summary.weak_score_threshold}
      </p>
    </div>
  );
}

function QualityTable({ title, items }: { title: string; items: SearchQualityItem[] }) {
  return (
    <section className="space-y-2">
      <h3 className="font-medium text-navy-800">
        {title} ({items.length})
      </h3>
      <div className="overflow-x-auto rounded-lg bg-white ring-1 ring-navy-200">
        <table className="min-w-full text-left text-xs">
          <thead className="bg-navy-50 text-navy-600">
            <tr>
              <th className="px-3 py-2">Time</th>
              <th className="px-3 py-2">Query</th>
              <th className="px-3 py-2">Results</th>
              <th className="px-3 py-2">Top score</th>
              <th className="px-3 py-2" title="End-to-end request (ms)">
                Total ms
              </th>
              <th className="px-3 py-2" title="Parse + retrieval until results ready (ms)">
                Ask ms
              </th>
              <th className="px-3 py-2" title="Conversational reply (ms)">
                Reply ms
              </th>
            </tr>
          </thead>
          <tbody>
            {items.map((row) => (
              <tr
                key={row.search_event_id}
                className="border-t border-navy-100"
                title={queryTooltip(row)}
              >
                <td className="whitespace-nowrap px-3 py-2 text-navy-800">{row.created_at}</td>
                <td
                  className="max-w-md truncate px-3 py-2 text-navy-800"
                  title={queryTooltip(row)}
                >
                  {row.display_query}
                </td>
                <td className="px-3 py-2 text-navy-800">{row.result_count}</td>
                <td className="px-3 py-2 text-navy-800">
                  {row.top_score != null
                    ? `${row.top_score.toFixed(3)} (${row.top_score_kind})`
                    : "—"}
                </td>
                <td className="whitespace-nowrap px-3 py-2 text-navy-800">
                  {formatMs(row.total_ms)}
                </td>
                <td className="whitespace-nowrap px-3 py-2 text-navy-800">
                  {formatMs(row.ask_ms)}
                </td>
                <td className="whitespace-nowrap px-3 py-2 text-navy-800">
                  {formatMs(row.reply_ms)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function QualityPage() {
  const [data, setData] = useState<SearchQualityLists | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [days, setDays] = useState<7 | 30 | 90>(90);

  useEffect(() => {
    setData(null);
    fetchSearchQuality(50, days)
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, [days]);

  if (error) return <p className="text-red-600">{error}</p>;
  if (!data) return <p className="text-navy-500">Loading…</p>;

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <h2 className="text-lg font-semibold text-navy-900">Search quality</h2>
        <div className="flex gap-1 rounded-lg bg-navy-100 p-1" role="group" aria-label="Time window">
          {([7, 30, 90] as const).map((n) => (
            <button
              key={n}
              type="button"
              onClick={() => setDays(n)}
              className={
                days === n
                  ? "rounded-md bg-white px-3 py-1 text-xs font-medium text-navy-900 shadow-sm"
                  : "rounded-md px-3 py-1 text-xs font-medium text-navy-600 hover:text-navy-900"
              }
            >
              {n}d
            </button>
          ))}
        </div>
      </div>
      <QualityTable title="Recent searches" items={data.recent ?? []} />
      <QualityTable title="Zero results" items={data.zero_result} />
      <QualityTable title="Weak results" items={data.weak_result} />
      <QualityTable title="No interaction" items={data.no_interaction} />
    </div>
  );
}

function CorpusPage() {
  const [images, setImages] = useState<CorpusImage[]>([]);
  const [corpusError, setCorpusError] = useState<string | null>(null);
  const [corpusLoading, setCorpusLoading] = useState(true);
  const [orphans, setOrphans] = useState<unknown[]>([]);
  const [orphansError, setOrphansError] = useState<string | null>(null);
  const [deleted, setDeleted] = useState<unknown[]>([]);
  const [deletedError, setDeletedError] = useState<string | null>(null);
  const [clusters, setClusters] = useState<unknown[]>([]);
  const [clustersError, setClustersError] = useState<string | null>(null);
  const [pendingAction, setPendingAction] = useState<
    Record<string, "regenerate" | "reindex">
  >({});
  const [cardErrors, setCardErrors] = useState<Record<string, string>>({});
  const [corpusSortBy, setCorpusSortBy] = useState<ResultSort>(defaultCatalogSort());
  const [corpusQualityFilter, setCorpusQualityFilter] =
    useState<CaptionQualityFilter>("all");
  const [corpusHealth, setCorpusHealth] = useState<CorpusHealth | null>(null);
  const [bulkRepairScope, setBulkRepairScope] = useState<"failed" | "weak" | null>(
    null,
  );
  const [bulkRepairResult, setBulkRepairResult] = useState<string | null>(null);
  const [bulkRepairError, setBulkRepairError] = useState<string | null>(null);
  const [thumbRegenPending, setThumbRegenPending] = useState(false);
  const [thumbRegenResult, setThumbRegenResult] = useState<string | null>(null);
  const [thumbRegenError, setThumbRegenError] = useState<string | null>(null);
  const [indexAction, setIndexAction] = useState<
    | "reconcile"
    | "repair"
    | "purge"
    | "backup"
    | "restore"
    | "orphan-preview"
    | "orphan-purge"
    | null
  >(null);
  const [indexActionResult, setIndexActionResult] = useState<string | null>(null);
  const [indexActionError, setIndexActionError] = useState<string | null>(null);
  const [indexBackups, setIndexBackups] = useState<IndexBackupInfo[]>([]);
  const [selectedBackupId, setSelectedBackupId] = useState("");
  const [backupsLoading, setBackupsLoading] = useState(false);

  const loadHealth = useCallback(() => {
    fetchCorpusHealth()
      .then(setCorpusHealth)
      .catch(() => setCorpusHealth(null));
  }, []);

  const loadCorpus = useCallback(() => {
    setCorpusLoading(true);
    setCorpusError(null);
    fetchCorpusImages(corpusSortBy, corpusQualityFilter)
      .then((r) => setImages(r.images))
      .catch((e) => setCorpusError(e instanceof Error ? e.message : String(e)))
      .finally(() => setCorpusLoading(false));
  }, [corpusSortBy, corpusQualityFilter]);

  const loadSecondary = useCallback(() => {
    fetchOrphans()
      .then((r) => {
        setOrphans(r.orphans);
        setOrphansError(null);
      })
      .catch((e) =>
        setOrphansError(e instanceof Error ? e.message : String(e)),
      );
    fetchDeleted()
      .then((r) => {
        setDeleted(r.deleted);
        setDeletedError(null);
      })
      .catch((e) =>
        setDeletedError(e instanceof Error ? e.message : String(e)),
      );
    fetchDuplicateClusters()
      .then((r) => {
        setClusters(r.clusters);
        setClustersError(r.error ?? null);
      })
      .catch((e) =>
        setClustersError(e instanceof Error ? e.message : String(e)),
      );
  }, []);

  const reloadAll = useCallback(() => {
    loadCorpus();
    loadSecondary();
    loadHealth();
  }, [loadCorpus, loadSecondary, loadHealth]);

  useEffect(() => {
    loadCorpus();
    loadSecondary();
    loadHealth();
  }, [loadCorpus, loadSecondary, loadHealth]);

  const hasPendingActions = Object.keys(pendingAction).length > 0;

  const handleReconcile = async () => {
    setIndexActionError(null);
    setIndexActionResult(null);
    setIndexAction("reconcile");
    try {
      const result = await reconcileIndex();
      setIndexActionResult(
        `Reconciled: purged ${result.orphan_chroma_purged} Chroma + ` +
          `${result.orphan_text_purged} text vector(s)` +
          (result.bm25_rebuilt ? "; BM25 rebuilt" : ""),
      );
      reloadAll();
    } catch (e) {
      setIndexActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setIndexAction(null);
    }
  };

  const handleFullRepair = async () => {
    if (
      !window.confirm(
        "Run full index repair? This may re-caption, re-embed, and rebuild indexes using Bedrock APIs.",
      )
    ) {
      return;
    }
    setIndexActionError(null);
    setIndexActionResult(null);
    setIndexAction("repair");
    try {
      const result = await repairIndex(false);
      if (result.skipped) {
        setIndexActionResult("Index already healthy; no repair needed.");
      } else {
        setIndexActionResult(
          `Repair complete in ${result.elapsed_sec ?? "?"}s. Healthy: ${result.is_healthy ?? false}`,
        );
      }
      reloadAll();
    } catch (e) {
      setIndexActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setIndexAction(null);
    }
  };

  const handlePurgeUnrecoverable = async () => {
    setIndexActionError(null);
    setIndexActionResult(null);
    setIndexAction("purge");
    try {
      const freshHealth = await fetchCorpusHealth();
      setCorpusHealth(freshHealth);
      const ids = freshHealth.unrecoverable_image_ids ?? [];
      const count =
        ids.length || (freshHealth.unrecoverable_source_missing_count ?? 0);
      if (count === 0) {
        setIndexActionResult(
          "Nothing currently unrecoverable; no images were purged.",
        );
        return;
      }
      if (
        !window.confirm(
          `Permanently delete ${count} image(s) missing both their cached image and recoverable source? ` +
            "Rows leave search, and residual files are removed when safe.",
        )
      ) {
        return;
      }

      const result = await purgeUnrecoverable(ids.length ? ids : undefined);
      if (result.deleted === 0) {
        setIndexActionResult(
          result.candidates === 0
            ? "Nothing currently unrecoverable; no images were purged."
            : `No images were purged; ${result.skipped} candidate(s) changed state before deletion.`,
        );
      } else {
        setIndexActionResult(
          `Permanently deleted ${result.deleted} of ${result.candidates} unrecoverable image(s)` +
            (result.files_deleted > 0
              ? ` and removed ${result.files_deleted} residual file(s)`
              : "") +
            "." +
            (result.skipped > 0
              ? ` ${result.skipped} candidate(s) were skipped because their state changed.`
              : ""),
        );
      }
      reloadAll();
    } catch (e) {
      setIndexActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setIndexAction(null);
    }
  };

  const formatOrphanBlobSummary = (result: {
    orphan_image_count: number;
    orphan_thumb_count: number;
    orphan_upload_count: number;
    orphan_staging_count: number;
    skipped_too_new_count: number;
    deleted_count?: number;
  }) =>
    `images=${result.orphan_image_count}, thumbs=${result.orphan_thumb_count}, ` +
    `uploads=${result.orphan_upload_count}, staging=${result.orphan_staging_count}` +
    (result.skipped_too_new_count > 0
      ? `, skipped_too_new=${result.skipped_too_new_count}`
      : "");

  const handlePreviewOrphanBlobs = async () => {
    setIndexAction("orphan-preview");
    setIndexActionError(null);
    setIndexActionResult(null);
    try {
      const result = await fetchOrphanBlobs(1);
      const total =
        result.orphan_image_count +
        result.orphan_thumb_count +
        result.orphan_upload_count +
        result.orphan_staging_count;
      setIndexActionResult(
        total === 0
          ? "No orphan S3 blobs found (or all candidates are too new)."
          : `Would purge ${total} orphan blob(s): ${formatOrphanBlobSummary(result)}.`,
      );
    } catch (e) {
      setIndexActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setIndexAction(null);
    }
  };

  const handlePurgeOrphanBlobs = async () => {
    setIndexAction("orphan-purge");
    setIndexActionError(null);
    setIndexActionResult(null);
    try {
      const preview = await fetchOrphanBlobs(1);
      const total =
        preview.orphan_image_count +
        preview.orphan_thumb_count +
        preview.orphan_upload_count +
        preview.orphan_staging_count;
      if (total === 0) {
        setIndexActionResult(
          "No orphan S3 blobs to purge (or all candidates are too new).",
        );
        return;
      }
      if (
        !window.confirm(
          `Permanently delete ${total} orphan S3 object(s) not referenced by any SQLite row?\n` +
            `${formatOrphanBlobSummary(preview)}\n` +
            "Soft-deleted images keep their blobs. Ingest jobs must be idle.",
        )
      ) {
        return;
      }
      const result = await purgeOrphanBlobs({ dryRun: false, minAgeHours: 1 });
      setIndexActionResult(
        `Deleted ${result.deleted_count} orphan blob(s)` +
          (result.failed_count > 0 ? ` (${result.failed_count} failed)` : "") +
          `: ${formatOrphanBlobSummary(result)}.`,
      );
      reloadAll();
    } catch (e) {
      setIndexActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setIndexAction(null);
    }
  };

  const loadIndexBackups = useCallback(async () => {
    setBackupsLoading(true);
    try {
      const result = await listIndexBackups();
      setIndexBackups(result.backups);
      setSelectedBackupId((prev) => {
        if (prev && result.backups.some((b) => b.id === prev)) return prev;
        return result.backups[0]?.id ?? "";
      });
    } catch (e) {
      setIndexBackups([]);
      setSelectedBackupId("");
      setIndexActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setBackupsLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadIndexBackups();
  }, [loadIndexBackups]);

  const handleBackupIndex = async () => {
    if (
      !window.confirm(
        "Backup the live search index to S3? Ingest will pause briefly while SQLite, Chroma, BM25, and hubness are packaged. Image blobs are not included.",
      )
    ) {
      return;
    }
    setIndexActionError(null);
    setIndexActionResult(null);
    setIndexAction("backup");
    try {
      const result = await backupIndex();
      setIndexActionResult(
        `Backup ${result.backup_id} uploaded` +
          (result.archive_bytes != null
            ? ` (${result.archive_bytes} bytes)`
            : "") +
          (result.s3_uri ? ` → ${result.s3_uri}` : ""),
      );
      await loadIndexBackups();
      reloadAll();
    } catch (e) {
      setIndexActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setIndexAction(null);
    }
  };

  const handleRestoreIndex = async () => {
    if (!selectedBackupId) {
      setIndexActionError("Select a completed S3 backup first.");
      return;
    }
    if (
      !window.confirm(
        `Replace the live EC2 search index with backup ${selectedBackupId}? Ingest will pause; this overwrites local SQLite/Chroma/BM25/hubness from the S3 vault.`,
      )
    ) {
      return;
    }
    setIndexActionError(null);
    setIndexActionResult(null);
    setIndexAction("restore");
    try {
      const result = await restoreIndex(selectedBackupId);
      setIndexActionResult(
        `Restored backup ${result.backup_id}` +
          (result.s3_uri ? ` from ${result.s3_uri}` : ""),
      );
      reloadAll();
    } catch (e) {
      setIndexActionError(e instanceof Error ? e.message : String(e));
    } finally {
      setIndexAction(null);
    }
  };

  const handleBulkRepair = async (scope: "failed" | "weak") => {
    const count =
      scope === "failed"
        ? (corpusHealth?.failed_caption_count ?? 0)
        : (corpusHealth?.weak_caption_count ?? 0);
    if (count === 0) return;

    const message =
      scope === "failed"
        ? `Re-run the vision model for ${count} failed caption(s)? This may take several minutes and uses the VLM API.`
        : `Re-run the vision model for ${count} weak-quality caption(s) only (not failed ones)? This may take several minutes and uses the VLM API.`;
    if (!window.confirm(message)) return;

    setBulkRepairError(null);
    setBulkRepairResult(null);
    setBulkRepairScope(scope);
    try {
      const result = await repairCaptions(scope);
      setBulkRepairResult(
        `Repaired ${result.repaired} of ${result.attempted}` +
          (result.errors > 0 ? ` (${result.errors} error(s))` : ""),
      );
      reloadAll();
    } catch (e) {
      setBulkRepairError(e instanceof Error ? e.message : String(e));
    } finally {
      setBulkRepairScope(null);
    }
  };

  const handleRegenerateMissingThumbs = async () => {
    const missing = corpusHealth?.missing_thumb_count ?? 0;
    if (
      !window.confirm(
        missing > 0
          ? `Generate JPEG thumbnails for ${missing} indexed image(s) missing one? Existing thumbnails are skipped.`
          : "Generate JPEG thumbnails for any indexed images that are missing one? Existing thumbnails are skipped.",
      )
    ) {
      return;
    }
    setThumbRegenError(null);
    setThumbRegenResult(null);
    setThumbRegenPending(true);
    try {
      const result = await regenerateMissingThumbnails();
      const freshHealth = await fetchCorpusHealth();
      setCorpusHealth(freshHealth);
      setThumbRegenResult(
        `Thumbnails: created ${result.created}, skipped ${result.skipped}, failed ${result.failed} (scanned ${result.scanned}). ` +
          `Coverage now ${freshHealth.thumb_count ?? 0}/${freshHealth.total_records ?? freshHealth.total_images}` +
          ((freshHealth.missing_thumb_count ?? 0) > 0
            ? ` (still missing ${freshHealth.missing_thumb_count})`
            : ""),
      );
      reloadAll();
    } catch (e) {
      setThumbRegenError(e instanceof Error ? e.message : String(e));
    } finally {
      setThumbRegenPending(false);
    }
  };

  const handleSoftDelete = async (imageId: string) => {
    await softDeleteImage(imageId);
    reloadAll();
  };

  const handleRestore = async (imageId: string) => {
    await restoreImage(imageId);
    reloadAll();
  };

  const handleRegenerate = async (imageId: string) => {
    if (
      !window.confirm(
        "Re-run the vision model to generate a new caption and refresh search indexes? This may take a minute and uses the VLM API.",
      )
    ) {
      return;
    }
    setCardErrors((prev) => {
      const next = { ...prev };
      delete next[imageId];
      return next;
    });
    setPendingAction((prev) => ({ ...prev, [imageId]: "regenerate" }));
    try {
      await regenerateCaption(imageId);
      reloadAll();
    } catch (e) {
      setCardErrors((prev) => ({
        ...prev,
        [imageId]: e instanceof Error ? e.message : String(e),
      }));
    } finally {
      setPendingAction((prev) => {
        const next = { ...prev };
        delete next[imageId];
        return next;
      });
    }
  };

  const handleReindex = async (imageId: string) => {
    setCardErrors((prev) => {
      const next = { ...prev };
      delete next[imageId];
      return next;
    });
    setPendingAction((prev) => ({ ...prev, [imageId]: "reindex" }));
    try {
      await reindexImage(imageId);
      reloadAll();
    } catch (e) {
      setCardErrors((prev) => ({
        ...prev,
        [imageId]: e instanceof Error ? e.message : String(e),
      }));
    } finally {
      setPendingAction((prev) => {
        const next = { ...prev };
        delete next[imageId];
        return next;
      });
    }
  };

  return (
    <div className="space-y-8">
      <h2 className="text-lg font-semibold text-navy-900">Corpus curation</h2>

      <section className="mb-6 rounded-lg border border-navy-200 bg-navy-50/50 p-4">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-navy-500">
          Index consistency
        </h3>
        {corpusHealth ? (
          <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-navy-700">
            <span
              className={
                corpusHealth.stores_in_sync
                  ? "rounded bg-emerald-100 px-2 py-0.5 font-medium text-emerald-800"
                  : "rounded bg-amber-100 px-2 py-0.5 font-medium text-amber-900"
              }
            >
              {corpusHealth.stores_in_sync ? "Stores in sync" : "Stores out of sync"}
            </span>
            <span>SQLite: {corpusHealth.total_records ?? corpusHealth.total_images}</span>
            <span>Chroma: {corpusHealth.chroma_vectors ?? "—"}</span>
            <span>Text: {corpusHealth.text_vector_count ?? "—"}</span>
            <span>BM25: {corpusHealth.bm25_doc_count ?? "—"}</span>
            <span
              className={
                (corpusHealth.missing_thumb_count ?? 0) > 0
                  ? "rounded bg-amber-100 px-2 py-0.5 font-medium text-amber-900"
                  : undefined
              }
            >
              Thumbs: {corpusHealth.thumb_count ?? 0}/
              {corpusHealth.total_records ?? corpusHealth.total_images}
              {(corpusHealth.missing_thumb_count ?? 0) > 0
                ? ` (missing ${corpusHealth.missing_thumb_count})`
                : ""}
            </span>
            {(corpusHealth.orphan_thumb_count ?? 0) > 0 && (
              <span>Orphan thumbs: {corpusHealth.orphan_thumb_count}</span>
            )}
            {(corpusHealth.orphan_image_count ?? 0) > 0 && (
              <span>Orphan images: {corpusHealth.orphan_image_count}</span>
            )}
            {(corpusHealth.orphan_chroma_count ?? 0) > 0 && (
              <span>Orphans: {corpusHealth.orphan_chroma_count}</span>
            )}
            {(corpusHealth.missing_cache_count ?? 0) > 0 && (
              <span>Missing cache: {corpusHealth.missing_cache_count}</span>
            )}
            {(corpusHealth.unrecoverable_source_missing_count ?? 0) > 0 && (
              <span>
                Unrecoverable: {corpusHealth.unrecoverable_source_missing_count}
              </span>
            )}
          </div>
        ) : (
          <p className="mt-2 text-xs text-navy-500">Loading health…</p>
        )}
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <button
            type="button"
            className="rounded-md border border-navy-300 bg-white px-3 py-1.5 text-xs font-medium text-navy-800 hover:bg-navy-50 disabled:opacity-50"
            disabled={indexAction !== null || bulkRepairScope !== null}
            onClick={() => void handleReconcile()}
          >
            {indexAction === "reconcile" ? "Reconciling…" : "Reconcile (safe)"}
          </button>
          <button
            type="button"
            className="rounded-md border border-brand-300 bg-brand-50 px-3 py-1.5 text-xs font-medium text-brand-900 hover:bg-brand-100 disabled:opacity-50"
            disabled={indexAction !== null || bulkRepairScope !== null}
            onClick={() => void handleFullRepair()}
          >
            {indexAction === "repair" ? "Repairing…" : "Full repair"}
          </button>
          <button
            type="button"
            className="rounded-md border border-amber-400 bg-amber-50 px-3 py-1.5 text-xs font-medium text-amber-950 hover:bg-amber-100 disabled:opacity-50"
            disabled={
              indexAction !== null ||
              bulkRepairScope !== null ||
              (corpusHealth?.unrecoverable_source_missing_count ?? 0) === 0
            }
            onClick={() => void handlePurgeUnrecoverable()}
          >
            {indexAction === "purge" ? "Purging…" : "Purge unrecoverable"}
          </button>
          <button
            type="button"
            className="rounded-md border border-navy-300 bg-white px-3 py-1.5 text-xs font-medium text-navy-800 hover:bg-navy-50 disabled:opacity-50"
            disabled={indexAction !== null || bulkRepairScope !== null}
            onClick={() => void handlePreviewOrphanBlobs()}
          >
            {indexAction === "orphan-preview"
              ? "Scanning orphans…"
              : "Preview orphan blobs"}
          </button>
          <button
            type="button"
            className="rounded-md border border-rose-300 bg-rose-50 px-3 py-1.5 text-xs font-medium text-rose-950 hover:bg-rose-100 disabled:opacity-50"
            disabled={indexAction !== null || bulkRepairScope !== null}
            onClick={() => void handlePurgeOrphanBlobs()}
          >
            {indexAction === "orphan-purge"
              ? "Purging orphans…"
              : "Purge orphan blobs"}
          </button>
          <button
            type="button"
            className="rounded-md border border-navy-300 bg-white px-3 py-1.5 text-xs font-medium text-navy-800 hover:bg-navy-50 disabled:opacity-50"
            disabled={indexAction !== null || bulkRepairScope !== null}
            onClick={() => void handleBackupIndex()}
          >
            {indexAction === "backup" ? "Backing up…" : "Backup index to S3"}
          </button>
          <select
            className="max-w-xs rounded-md border border-navy-300 bg-white px-2 py-1.5 text-xs text-navy-800 disabled:opacity-50"
            value={selectedBackupId}
            disabled={
              indexAction !== null ||
              bulkRepairScope !== null ||
              backupsLoading ||
              indexBackups.length === 0
            }
            onChange={(e) => setSelectedBackupId(e.target.value)}
            aria-label="Select index backup"
          >
            {indexBackups.length === 0 ? (
              <option value="">
                {backupsLoading ? "Loading backups…" : "No S3 backups yet"}
              </option>
            ) : (
              indexBackups.map((backup) => (
                <option key={backup.id} value={backup.id}>
                  {backup.id}
                  {backup.label ? ` (${backup.label})` : ""}
                </option>
              ))
            )}
          </select>
          <button
            type="button"
            className="rounded-md border border-navy-300 bg-white px-3 py-1.5 text-xs font-medium text-navy-800 hover:bg-navy-50 disabled:opacity-50"
            disabled={
              indexAction !== null ||
              bulkRepairScope !== null ||
              !selectedBackupId
            }
            onClick={() => void handleRestoreIndex()}
          >
            {indexAction === "restore" ? "Restoring…" : "Restore from S3"}
          </button>
          {indexActionResult && (
            <p className="text-xs text-navy-600">{indexActionResult}</p>
          )}
          {indexActionError && (
            <p className="text-xs text-red-600">{indexActionError}</p>
          )}
        </div>
        <p className="mt-2 text-xs text-navy-500">
          S3 holds versioned index snapshots only (SQLite, Chroma, BM25, hubness).
          Backup/restore pauses ingest briefly; image blobs stay under uploads/images.
        </p>
      </section>

      <section>
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <div className="flex items-baseline gap-2">
            <h3 className="text-xs font-semibold uppercase tracking-wide text-navy-500">
              Indexed corpus (
              {corpusLoading
                ? "…"
                : (corpusHealth?.total_records ??
                  corpusHealth?.total_images ??
                  images.length)}
              )
            </h3>
            {!corpusLoading && corpusQualityFilter !== "all" && (
              <span className="text-xs text-navy-500">
                Showing {images.length} filtered
              </span>
            )}
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <label className="inline-flex items-center gap-1.5 text-xs text-navy-600">
              <span className="shrink-0 font-medium">Quality</span>
              <select
                value={corpusQualityFilter}
                disabled={corpusLoading}
                onChange={(e) =>
                  setCorpusQualityFilter(e.target.value as CaptionQualityFilter)
                }
                className="rounded-md border border-navy-200 bg-white px-2 py-1 text-xs text-navy-800 disabled:opacity-50"
              >
                <option value="all">All</option>
                <option value="ok">OK</option>
                <option value="weak">Weak</option>
                <option value="failed">Failed</option>
              </select>
            </label>
            <SortSelect
              value={corpusSortBy}
              onChange={setCorpusSortBy}
              includeRelevance={false}
              disabled={corpusLoading}
            />
          </div>
        </div>
        <div className="mb-3 flex flex-wrap items-center gap-3">
          <button
            type="button"
            className="rounded-md bg-brand-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-brand-700 disabled:opacity-50"
            disabled={
              bulkRepairScope !== null ||
              thumbRegenPending ||
              hasPendingActions ||
              (corpusHealth?.failed_caption_count ?? 0) === 0
            }
            onClick={() => void handleBulkRepair("failed")}
          >
            {bulkRepairScope === "failed"
              ? "Repairing failed…"
              : `Repair all failed (${corpusHealth?.failed_caption_count ?? 0})`}
          </button>
          <button
            type="button"
            className="rounded-md border border-amber-300 bg-amber-50 px-3 py-1.5 text-xs font-medium text-amber-900 hover:bg-amber-100 disabled:opacity-50"
            disabled={
              bulkRepairScope !== null ||
              thumbRegenPending ||
              hasPendingActions ||
              (corpusHealth?.weak_caption_count ?? 0) === 0
            }
            onClick={() => void handleBulkRepair("weak")}
          >
            {bulkRepairScope === "weak"
              ? "Repairing weak…"
              : `Repair all weak (${corpusHealth?.weak_caption_count ?? 0})`}
          </button>
          <button
            type="button"
            className="rounded-md border border-navy-300 bg-white px-3 py-1.5 text-xs font-medium text-navy-800 hover:bg-navy-50 disabled:opacity-50"
            disabled={
              thumbRegenPending ||
              bulkRepairScope !== null ||
              hasPendingActions
            }
            onClick={() => void handleRegenerateMissingThumbs()}
          >
            {thumbRegenPending
              ? "Generating thumbnails…"
              : (corpusHealth?.missing_thumb_count ?? 0) > 0
                ? `Regenerate missing thumbnails (${corpusHealth?.missing_thumb_count})`
                : "Regenerate missing thumbnails"}
          </button>
          {bulkRepairResult && (
            <p className="text-xs text-navy-600">{bulkRepairResult}</p>
          )}
          {bulkRepairError && (
            <p className="text-xs text-red-600">{bulkRepairError}</p>
          )}
          {thumbRegenResult && (
            <p className="text-xs text-navy-600">{thumbRegenResult}</p>
          )}
          {thumbRegenError && (
            <p className="text-xs text-red-600">{thumbRegenError}</p>
          )}
        </div>
        {corpusError && (
          <p className="mb-2 text-sm text-red-600">{corpusError}</p>
        )}
        {corpusLoading ? (
          <p className="text-sm text-navy-500">Loading corpus…</p>
        ) : images.length === 0 ? (
          <div className="rounded-lg border border-navy-200 bg-navy-50/80 p-4 text-sm text-navy-700">
            <p className="font-medium text-navy-900">No indexed images in the local search index.</p>
            <p className="mt-2 text-navy-600">
              This page lists metadata from SQLite on the app host — it does not
              scan the S3 bucket. Image bytes may still live in S3, and old chat
              turns in the browser can show images even when this list is empty.
              If a large ingest crashed, the app should auto-restore the latest
              S3 index checkpoint on startup; check Ingestions diagnostics for
              startup restore status before re-uploading.
            </p>
            {((corpusHealth?.total_records ?? corpusHealth?.total_images ?? 0) === 0 &&
              (corpusHealth?.chroma_vectors ?? 0) === 0) && (
              <div className="mt-3 space-y-2 text-xs text-navy-600">
                <p className="font-medium text-navy-800">If this is an AWS deployment:</p>
                <ul className="list-disc space-y-1 pl-5">
                  <li>
                    In the browser Network tab, confirm{" "}
                    <code className="rounded bg-white px-1">GET /api/status</code> and{" "}
                    <code className="rounded bg-white px-1">GET /api/admin/corpus/images</code>{" "}
                    return empty counts, then run a new chat search
                    (not an old conversation) and check{" "}
                    <code className="rounded bg-white px-1">POST /api/chat</code>.
                  </li>
                  <li>
                    Ask whoever manages the host to verify the persistent{" "}
                    <code className="rounded bg-white px-1">./data → /app/data</code>{" "}
                    volume survives container recreate, and that{" "}
                    <code className="rounded bg-white px-1">imagecb.db</code> and{" "}
                    <code className="rounded bg-white px-1">chroma/</code> are
                    non-empty after ingest.
                  </li>
                  <li>
                    Compare runtime and SQLite identity under Ingestions
                    diagnostics; re-ingest only after the data volume is fixed.
                  </li>
                </ul>
                <p>
                  <Link
                    to="/admin/ingestions"
                    className="font-medium text-brand-700 hover:underline"
                  >
                    Open ingest diagnostics →
                  </Link>
                </p>
              </div>
            )}
            {corpusQualityFilter !== "all" && (
              <p className="mt-3 text-xs text-navy-500">
                Quality filter is set to “{corpusQualityFilter}”. Try “All” if you
                expect images with other caption qualities.
              </p>
            )}
          </div>
        ) : (
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {images.map((img) => {
              const pending = pendingAction[img.image_id];
              const cardError = cardErrors[img.image_id];
              const quality = (img.caption_quality || "ok").toLowerCase();
              return (
                <article
                  key={img.image_id}
                  className="flex flex-col overflow-hidden rounded-lg bg-white ring-1 ring-navy-200"
                >
                  <div className="aspect-video bg-navy-50">
                    <img
                      src={img.thumb_url || img.image_url}
                      alt={img.caption_short || img.image_id}
                      className="h-full w-full object-contain"
                      loading="lazy"
                    />
                  </div>
                  <div className="flex flex-1 flex-col gap-1 p-2 text-xs">
                    {img.needs_regeneration && (
                      <span
                        className={
                          quality === "failed"
                            ? "self-start rounded bg-red-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-red-700"
                            : "self-start rounded bg-amber-100 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-amber-800"
                        }
                      >
                        {quality}
                      </span>
                    )}
                    <p className="line-clamp-2 text-navy-800">
                      {img.caption_short || "(no caption)"}
                    </p>
                    <p className="truncate text-navy-500" title={img.source_file}>
                      {img.source_file}
                    </p>
                    {cardError && (
                      <p className="text-red-600">{cardError}</p>
                    )}
                    <div className="mt-auto flex flex-wrap gap-x-3 gap-y-1">
                      <button
                        type="button"
                        className="font-medium text-brand-600 hover:underline disabled:opacity-50"
                        disabled={pending !== undefined}
                        onClick={() => void handleRegenerate(img.image_id)}
                      >
                        {pending === "regenerate" ? "Regenerating…" : "Regenerate"}
                      </button>
                      <button
                        type="button"
                        className="text-navy-600 hover:underline disabled:opacity-50"
                        disabled={pending !== undefined}
                        onClick={() => void handleReindex(img.image_id)}
                      >
                        {pending === "reindex" ? "Re-indexing…" : "Re-index"}
                      </button>
                      <button
                        type="button"
                        className="text-red-600 hover:underline disabled:opacity-50"
                        disabled={pending !== undefined}
                        onClick={() => void handleSoftDelete(img.image_id)}
                      >
                        Soft delete
                      </button>
                    </div>
                  </div>
                </article>
              );
            })}
          </div>
        )}
      </section>

      <section>
        <h3 className="mb-2 font-medium text-navy-800">Orphan images (never served)</h3>
        {orphansError && (
          <p className="mb-2 text-sm text-amber-700">{orphansError}</p>
        )}
        <ul className="space-y-1 text-sm">
          {(orphans as { image_id: string; caption_short?: string }[]).map((o) => (
            <li
              key={o.image_id}
              className="flex items-center gap-2 rounded bg-white px-3 py-2 ring-1 ring-navy-200"
            >
              <span className="font-mono text-xs text-navy-700">{o.image_id.slice(0, 8)}…</span>
              <span className="flex-1 truncate text-navy-800">{o.caption_short || "(no caption)"}</span>
              <button
                type="button"
                className="text-xs text-red-600 hover:underline"
                onClick={() => void handleSoftDelete(o.image_id)}
              >
                Soft delete
              </button>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h3 className="mb-2 font-medium text-navy-800">Soft-deleted (recoverable)</h3>
        {deletedError && (
          <p className="mb-2 text-sm text-amber-700">{deletedError}</p>
        )}
        <ul className="space-y-1 text-sm">
          {(deleted as { image_id: string; deleted_at?: string }[]).map((d) => (
            <li
              key={d.image_id}
              className="flex items-center gap-2 rounded bg-white px-3 py-2 ring-1 ring-navy-200"
            >
              <span className="font-mono text-xs text-navy-700">{d.image_id.slice(0, 8)}…</span>
              <span className="text-navy-500">{d.deleted_at}</span>
              <button
                type="button"
                className="text-xs font-medium text-brand-600 hover:underline"
                onClick={() => void handleRestore(d.image_id)}
              >
                Restore
              </button>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h3 className="mb-2 font-medium text-navy-800">Near-duplicate clusters</h3>
        {clustersError && (
          <p className="mb-2 text-sm text-amber-700">{clustersError}</p>
        )}
        {(clusters as { cluster_id: string; size: number; max_similarity: number; images: { image_id: string; caption_short?: string }[] }[]).map(
          (cl) => (
            <details key={cl.cluster_id} className="mb-2 rounded bg-white p-3 ring-1 ring-navy-200">
              <summary className="cursor-pointer text-sm font-medium text-navy-900">
                Cluster ({cl.size} images, max sim {cl.max_similarity})
              </summary>
              <ul className="mt-2 space-y-1 text-xs text-navy-600">
                {cl.images.map((img) => (
                  <li key={img.image_id}>
                    {img.image_id.slice(0, 8)}… — {img.caption_short || "—"}
                  </li>
                ))}
              </ul>
            </details>
          ),
        )}
      </section>
    </div>
  );
}

function AuditPage() {
  const [entries, setEntries] = useState<unknown[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchAudit()
      .then((r) => setEntries(r.entries))
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error) return <p className="text-red-600">{error}</p>;

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold text-navy-900">Audit log</h2>
      <ul className="space-y-2 text-sm">
        {(entries as { created_at: string; actor: string; action: string; target_id: string }[]).map(
          (e) => (
            <li
              key={`${e.created_at}-${e.target_id}-${e.action}`}
              className="rounded bg-white px-3 py-2 ring-1 ring-navy-200"
            >
              <span className="text-navy-500">{e.created_at}</span> —{" "}
              <span className="text-navy-800">{e.actor}</span> —{" "}
              <strong className="text-navy-900">{e.action}</strong> on {e.target_id}
            </li>
          ),
        )}
      </ul>
    </div>
  );
}

function IngestionsPage() {
  const [jobs, setJobs] = useState<IngestJob[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState<string | null>(null);
  const [diagnostics, setDiagnostics] = useState<IngestDiagnostics | null>(null);
  const [preflight, setPreflight] = useState<IngestPreflight | null>(null);
  const [preflightRunning, setPreflightRunning] = useState(false);
  const selectedJobId = new URLSearchParams(window.location.search).get("job");
  const frontendBuildId = import.meta.env.VITE_APP_BUILD_ID || "development";

  const load = useCallback(() => {
    fetchIngestJobs()
      .then((result) => {
        setJobs(result.jobs);
        setError(null);
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  useEffect(() => {
    load();
    fetchIngestDiagnostics()
      .then(setDiagnostics)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
    const timer = window.setInterval(load, 2000);
    return () => window.clearInterval(timer);
  }, [load]);

  const handlePreflight = async () => {
    setPreflightRunning(true);
    try {
      const result = await runIngestPreflight();
      setPreflight(result);
      setDiagnostics(result.runtime);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setPreflightRunning(false);
    }
  };

  const handleCancel = async (job: IngestJob) => {
    if (
      !window.confirm(
        `Cancel ingest ${job.job_id.slice(0, 8)}? Images already completed will be kept.`,
      )
    ) {
      return;
    }
    setCancelling(job.job_id);
    try {
      await cancelIngestJob(job.job_id);
      load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setCancelling(null);
    }
  };

  const active = jobs.filter((job) =>
    ["queued", "running", "cancel_requested"].includes(job.status),
  );
  const recent = jobs.filter(
    (job) => !["queued", "running", "cancel_requested"].includes(job.status),
  );

  const table = (items: IngestJob[]) => (
    <div className="overflow-x-auto rounded-lg bg-white ring-1 ring-navy-200">
      <table className="min-w-full text-left text-xs">
        <thead className="bg-navy-50 text-navy-600">
          <tr>
            <th className="px-3 py-2">Job</th>
            <th className="px-3 py-2">Status</th>
            <th className="px-3 py-2">Files</th>
            <th className="px-3 py-2">Images</th>
            <th className="px-3 py-2">Started</th>
            <th className="px-3 py-2">Options</th>
            <th className="px-3 py-2">Action</th>
          </tr>
        </thead>
        <tbody>
          {items.map((job) => {
            const heartbeatAge = heartbeatAgeSeconds(job.heartbeat_at);
            const stale = isStaleIngestJob(job);
            return (
            <tr
              key={job.job_id}
              className={`border-t align-top ${
                job.job_id === selectedJobId
                  ? "border-brand-300 bg-brand-50"
                  : "border-navy-100"
              }`}
            >
              <td className="px-3 py-2 font-mono text-navy-800" title={job.job_id}>
                {job.job_id.slice(0, 8)}…
              </td>
              <td className="px-3 py-2">
                <span className="rounded bg-navy-100 px-2 py-0.5 font-medium text-navy-800">
                  {job.status.replace("_", " ")}
                </span>
                {job.phase && <p className="mt-1 text-navy-600">{job.phase.replace(/_/g, " ")}</p>}
                {job.status_detail && <p className="mt-1 max-w-xs text-navy-500">{job.status_detail}</p>}
                {heartbeatAge !== null && (
                  <p className={stale ? "mt-1 font-medium text-red-600" : "mt-1 text-navy-400"}>
                    heartbeat {heartbeatAge}s ago{stale ? " — worker may be stale" : ""}
                  </p>
                )}
                {job.error && <p className="mt-1 max-w-xs text-red-600">{job.error}</p>}
              </td>
              <td className="px-3 py-2 text-navy-700">
                {job.files_done}/{job.files_total}
                <p className="max-w-xs truncate text-navy-500" title={job.files.join(", ")}>
                  {job.files.join(", ")}
                </p>
              </td>
              <td className="px-3 py-2 text-navy-700">
                {job.images_processed}/{job.images_seen || "?"}
              </td>
              <td className="whitespace-nowrap px-3 py-2 text-navy-600">
                {job.started_at ? new Date(job.started_at).toLocaleString() : "—"}
              </td>
              <td className="px-3 py-2 text-navy-600">
                workers={String(job.options.workers ?? "—")}
                {job.options.force ? ", force" : ""}
                {job.options.skip_caption ? ", no captions" : ""}
                {job.options.skip_ocr ? ", no OCR" : ""}
              </td>
              <td className="px-3 py-2">
                {job.cancellable ? (
                  <button
                    type="button"
                    className="font-medium text-red-600 hover:underline disabled:opacity-50"
                    disabled={cancelling === job.job_id || job.status === "cancel_requested"}
                    onClick={() => void handleCancel(job)}
                  >
                    {cancelling === job.job_id || job.status === "cancel_requested"
                      ? "Cancelling…"
                      : "Cancel"}
                  </button>
                ) : (
                  <span className="text-navy-400">—</span>
                )}
              </td>
            </tr>
          )})}
        </tbody>
      </table>
    </div>
  );

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-lg font-semibold text-navy-900">Ingestions</h2>
        <p className="text-xs text-navy-500">
          Durable ingest jobs continue when browser tabs close.
        </p>
      </div>
      {error && <p className="text-sm text-red-600">{error}</p>}
      {selectedJobId && !jobs.some((job) => job.job_id === selectedJobId) && (
        <p className="text-sm font-medium text-red-600">
          Job {selectedJobId} is visible to the upload page but missing from this runtime.
          Compare the build, runtime, and SQLite identities below.
        </p>
      )}
      <section className="space-y-3 rounded-lg bg-white p-4 ring-1 ring-navy-200">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h3 className="font-medium text-navy-800">Runtime diagnostics</h3>
            <p className="text-xs text-navy-500">
              Frontend build {frontendBuildId}; backend build {diagnostics?.build_id ?? "loading…"}
            </p>
          </div>
          <button
            type="button"
            disabled={preflightRunning}
            onClick={() => void handlePreflight()}
            className="rounded bg-brand-600 px-3 py-1.5 text-xs font-medium text-white disabled:opacity-50"
          >
            {preflightRunning ? "Running preflight…" : "Run ingest preflight"}
          </button>
        </div>
        {diagnostics && (
          <div className="grid gap-1 text-xs text-navy-600 sm:grid-cols-2">
            <span>Runtime: {diagnostics.runtime_id}</span>
            <span>Runner: {diagnostics.runner.alive ? "healthy" : "not running"}</span>
            <span>Storage: {diagnostics.storage_backend}</span>
            <span>SQLite: {diagnostics.sqlite_identity}</span>
            {diagnostics.sqlite_path && (
              <span className="sm:col-span-2">
                SQLite path: {diagnostics.sqlite_path}
              </span>
            )}
            <span>
              Checkpointing:{" "}
              {diagnostics.index_checkpoint_enabled ? "enabled" : "disabled"}
              {diagnostics.index_checkpoint_every_n != null
                ? ` (every ${diagnostics.index_checkpoint_every_n})`
                : ""}
            </span>
            <span>
              Auto-restore:{" "}
              {diagnostics.index_auto_restore_on_startup ? "enabled" : "disabled"}
            </span>
            {diagnostics.last_checkpoint?.backup_id && (
              <span className="sm:col-span-2">
                Last checkpoint: {diagnostics.last_checkpoint.backup_id}
                {diagnostics.last_checkpoint.total_records != null
                  ? ` (${diagnostics.last_checkpoint.total_records} records)`
                  : ""}
              </span>
            )}
            {diagnostics.last_checkpoint?.error && (
              <span className="sm:col-span-2 text-red-600">
                Last checkpoint error: {diagnostics.last_checkpoint.error}
              </span>
            )}
            {diagnostics.startup_restore && (
              <span className="sm:col-span-2">
                Startup restore:{" "}
                {diagnostics.startup_restore.restored
                  ? `restored ${diagnostics.startup_restore.backup_id ?? ""} (${diagnostics.startup_restore.total_records ?? "?"} records)`
                  : diagnostics.startup_restore.error
                    ? `failed — ${diagnostics.startup_restore.error}`
                    : diagnostics.startup_restore.skipped
                      ? `skipped (${diagnostics.startup_restore.skipped})`
                      : diagnostics.startup_restore.attempted
                        ? "attempted"
                        : "not attempted"}
              </span>
            )}
          </div>
        )}
        {diagnostics && (
          <p className="text-xs text-navy-500">
            Live search indexes are a local cache under{" "}
            <code className="rounded bg-navy-50 px-1">DATA_DIR</code>
            . With S3 durability enabled, ingest checkpoints that cache to S3 and
            empty startups restore <code className="rounded bg-navy-50 px-1">checkpoint-latest</code>.
            Prefer a persistent host volume plus checkpoints; an empty path here
            with chat history still showing images usually means a lost local index.
          </p>
        )}
        {diagnostics && diagnostics.build_id !== frontendBuildId && (
          <p className="text-xs font-medium text-red-600">
            Frontend and backend build IDs differ. Rebuild and redeploy one Docker image.
          </p>
        )}
        {preflight && (
          <div className="space-y-1 text-xs">
            {preflight.checks.map((check) => (
              <p key={check.name} className={check.ok ? "text-navy-600" : "font-medium text-red-600"}>
                {check.ok ? "Pass" : "Fail"} — {check.name}: {check.detail} ({check.elapsed_ms}ms)
              </p>
            ))}
          </div>
        )}
      </section>
      <section className="space-y-2">
        <h3 className="font-medium text-navy-800">Current ({active.length})</h3>
        {active.length ? table(active) : <p className="text-sm text-navy-500">No active ingests.</p>}
      </section>
      <section className="space-y-2">
        <h3 className="font-medium text-navy-800">Recent ({recent.length})</h3>
        {recent.length ? table(recent) : <p className="text-sm text-navy-500">No ingest history.</p>}
      </section>
    </div>
  );
}

function PendingAdditionsPage() {
  const [items, setItems] = useState<PendingEditItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchPendingEdits(100);
      setItems(res.items);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const handleAccept = async (pendingId: string) => {
    setBusyId(pendingId);
    setActionError(null);
    try {
      await acceptPendingEdit(pendingId);
      await load();
    } catch (err: unknown) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyId(null);
    }
  };

  const handleDecline = async (pendingId: string) => {
    setBusyId(pendingId);
    setActionError(null);
    try {
      await declinePendingEdit(pendingId);
      await load();
    } catch (err: unknown) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusyId(null);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-navy-900">Pending additions</h2>
          <p className="mt-1 text-sm text-navy-600">
            Nano Banana edits submitted by chat users. Accept runs a full ingest as a
            new corpus image; decline deletes the staged files only.
          </p>
        </div>
        <button
          type="button"
          onClick={() => void load()}
          className="rounded-md border border-navy-200 bg-white px-3 py-1.5 text-xs font-medium text-navy-800 hover:bg-navy-50"
        >
          Refresh
        </button>
      </div>
      {error && <p className="text-sm text-red-700">{error}</p>}
      {actionError && <p className="text-sm text-red-700">{actionError}</p>}
      {loading ? (
        <p className="text-sm text-navy-500">Loading…</p>
      ) : items.length === 0 ? (
        <p className="rounded-lg bg-white p-6 text-sm text-navy-500 ring-1 ring-navy-200">
          No pending additions.
        </p>
      ) : (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {items.map((item) => {
            const busy = busyId === item.pending_id;
            return (
              <article
                key={item.pending_id}
                className="flex flex-col overflow-hidden rounded-lg bg-white ring-1 ring-navy-200"
              >
                <div className="aspect-video bg-navy-50">
                  <img
                    src={item.thumb_url || item.image_url}
                    alt={item.last_prompt || item.pending_id}
                    className="h-full w-full object-contain"
                    loading="lazy"
                  />
                </div>
                <div className="flex flex-1 flex-col gap-1 p-3 text-xs">
                  <p className="text-navy-500">
                    Source:{" "}
                    <a
                      className="font-medium text-brand-700 hover:underline"
                      href={`/api/images/${item.source_image_id}`}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {item.source_image_id}
                    </a>
                  </p>
                  {item.created_at && (
                    <p className="text-navy-500">Submitted: {item.created_at}</p>
                  )}
                  {item.last_prompt && (
                    <p className="line-clamp-3 text-navy-800" title={item.last_prompt}>
                      {item.last_prompt}
                    </p>
                  )}
                  <div className="mt-auto flex flex-wrap gap-2 pt-2">
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => void handleAccept(item.pending_id)}
                      className="rounded-md bg-brand-500 px-2.5 py-1 text-xs font-semibold text-white hover:bg-brand-600 disabled:opacity-50"
                    >
                      {busy ? "Working…" : "Accept"}
                    </button>
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => void handleDecline(item.pending_id)}
                      className="rounded-md border border-red-200 bg-red-50 px-2.5 py-1 text-xs font-semibold text-red-800 hover:bg-red-100 disabled:opacity-50"
                    >
                      Decline
                    </button>
                  </div>
                </div>
              </article>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default function AdminApp() {
  return (
    <AdminLayout>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/quality" element={<QualityPage />} />
        <Route path="/corpus" element={<CorpusPage />} />
        <Route path="/ingestions" element={<IngestionsPage />} />
        <Route path="/pending" element={<PendingAdditionsPage />} />
        <Route path="/audit" element={<AuditPage />} />
      </Routes>
    </AdminLayout>
  );
}
