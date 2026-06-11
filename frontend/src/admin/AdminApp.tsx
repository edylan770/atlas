import { useCallback, useEffect, useState } from "react";
import { Route, Routes } from "react-router-dom";
import {
  fetchAnalyticsSummary,
  fetchAudit,
  fetchCorpusHealth,
  reconcileIndex,
  repairIndex,
  fetchCorpusImages,
  fetchDeleted,
  fetchDuplicateClusters,
  fetchOrphans,
  fetchSearchQuality,
  regenerateCaption,
  reindexImage,
  repairCaptions,
  restoreImage,
  softDeleteImage,
  type AnalyticsSummary,
  type CaptionQualityFilter,
  type CorpusHealth,
  type CorpusImage,
  type SearchQualityItem,
  type SearchQualityLists,
} from "../api/adminClient";
import { SortSelect } from "../components/SortSelect";
import { defaultCatalogSort } from "../sortResults";
import type { ResultSort } from "../types";
import { AdminLayout } from "./AdminShell";

function queryTooltip(row: SearchQualityItem): string {
  const user = row.user_message ?? row.query_text ?? "";
  const semantic = row.parsed_semantic_query ?? "";
  const parts: string[] = [];
  if (user) parts.push(`User: ${user}`);
  if (semantic && semantic !== user) parts.push(`Interpreted: ${semantic}`);
  return parts.join("\n") || row.display_query;
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

  useEffect(() => {
    fetchAnalyticsSummary()
      .then(setSummary)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
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
      <div>
        <h2 className="text-lg font-semibold text-navy-900">Dashboard</h2>
        <p className="text-xs text-navy-500">Last 7 days</p>
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
            </tr>
          </thead>
          <tbody>
            {items.map((row) => (
              <tr key={row.search_event_id} className="border-t border-navy-100">
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

  useEffect(() => {
    fetchSearchQuality()
      .then(setData)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  if (error) return <p className="text-red-600">{error}</p>;
  if (!data) return <p className="text-navy-500">Loading…</p>;

  return (
    <div className="space-y-8">
      <h2 className="text-lg font-semibold text-navy-900">Search quality</h2>
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
  const [indexAction, setIndexAction] = useState<"reconcile" | "repair" | null>(
    null,
  );
  const [indexActionResult, setIndexActionResult] = useState<string | null>(null);
  const [indexActionError, setIndexActionError] = useState<string | null>(null);

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
            {(corpusHealth.orphan_chroma_count ?? 0) > 0 && (
              <span>Orphans: {corpusHealth.orphan_chroma_count}</span>
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
          {indexActionResult && (
            <p className="text-xs text-navy-600">{indexActionResult}</p>
          )}
          {indexActionError && (
            <p className="text-xs text-red-600">{indexActionError}</p>
          )}
        </div>
      </section>

      <section>
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-navy-500">
            Indexed corpus ({corpusLoading ? "…" : images.length})
          </h3>
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
              hasPendingActions ||
              (corpusHealth?.weak_caption_count ?? 0) === 0
            }
            onClick={() => void handleBulkRepair("weak")}
          >
            {bulkRepairScope === "weak"
              ? "Repairing weak…"
              : `Repair all weak (${corpusHealth?.weak_caption_count ?? 0})`}
          </button>
          {bulkRepairResult && (
            <p className="text-xs text-navy-600">{bulkRepairResult}</p>
          )}
          {bulkRepairError && (
            <p className="text-xs text-red-600">{bulkRepairError}</p>
          )}
        </div>
        {corpusError && (
          <p className="mb-2 text-sm text-red-600">{corpusError}</p>
        )}
        {corpusLoading ? (
          <p className="text-sm text-navy-500">Loading corpus…</p>
        ) : images.length === 0 ? (
          <p className="text-sm text-navy-500">No indexed images.</p>
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
                      src={img.image_url}
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

export default function AdminApp() {
  return (
    <AdminLayout>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/quality" element={<QualityPage />} />
        <Route path="/corpus" element={<CorpusPage />} />
        <Route path="/audit" element={<AuditPage />} />
      </Routes>
    </AdminLayout>
  );
}
