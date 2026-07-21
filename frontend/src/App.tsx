import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  fetchCorpusCatalog,
  fetchIngestJob,
  fetchStatus,
  fetchSuggestions,
  createIngestJobDirectS3,
  createIngestJobBatched,
  DirectS3UnavailableError,
  type IngestJobUploadProgress,
  searchSimilarByImage,
  searchSimilarByImageId,
  sendChatStream,
} from "./api/client";
import { cancelIngestJob } from "./api/adminClient";
import {
  createConversation,
  lastTurn,
  loadStoredState,
  newTurnId,
  saveStoredState,
  titleFromMessage,
} from "./chat/storage";
import { ChatMessageList } from "./components/ChatMessageList";
import { ChatSidebar } from "./components/ChatSidebar";
import { Composer } from "./components/Composer";
import { CorpusDrawer } from "./components/CorpusDrawer";
import { EmptyState } from "./components/EmptyState";
import { AdminNavLink } from "./components/AdminNavLink";
import { AtlasLoadingScreen, useMinDurationLoading } from "./components/AtlasLoadingScreen";
import { Header } from "./components/Header";
import { formatIngestPhase, heartbeatAgeSeconds, isMissingIngestJobError } from "./ingestStatus";
import { ResultsGrid } from "./components/ResultsGrid";
import { SortSelect } from "./components/SortSelect";
import { defaultCatalogSort, defaultSearchSort, sortResultCards } from "./sortResults";
import type {
  CatalogItem,
  Conversation,
  ConversationTurn,
  ResultCard,
  ResultSort,
} from "./types";

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

function applyTurnToPanel(
  turn: ConversationTurn | null,
  setResults: (r: ResultCard[]) => void,
) {
  setResults(turn?.results ?? []);
}

export default function App() {
  const [indexedCount, setIndexedCount] = useState(0);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(
    null,
  );
  const [selectedTurnId, setSelectedTurnId] = useState<string | null>(null);
  const [results, setResults] = useState<ResultCard[]>([]);
  const [input, setInput] = useState("");
  const [topK, setTopK] = useState(10);
  const [minMatchPercent, setMinMatchPercent] = useState(0);
  const [similarityAxis, setSimilarityAxis] = useState<
    import("./api/client").SimilarityAxis
  >("balanced");
  const [loading, setLoading] = useState(false);
  const [appReady, setAppReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true);

  const [corpusOpen, setCorpusOpen] = useState(false);
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
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);
  const [searchEventId, setSearchEventId] = useState<string | null>(null);
  const [searchSortBy, setSearchSortBy] = useState<ResultSort>(defaultSearchSort());
  const [catalogSortBy, setCatalogSortBy] = useState<ResultSort>(defaultCatalogSort());

  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const ingestUploadAbortRef = useRef<AbortController | null>(null);
  const stagingIngestJobIdRef = useRef<string | null>(null);
  const ingestUploadingRef = useRef(false);

  const activeConversation = useMemo(
    () => conversations.find((c) => c.id === activeConversationId) ?? null,
    [conversations, activeConversationId],
  );

  const turns = useMemo(
    () => activeConversation?.turns ?? [],
    [activeConversation],
  );

  const displayResults = useMemo(
    () => sortResultCards(results, searchSortBy),
    [results, searchSortBy],
  );

  const persistSoon = useCallback(
    (nextConversations: Conversation[], nextActiveId: string | null) => {
      if (saveTimer.current) clearTimeout(saveTimer.current);
      saveTimer.current = setTimeout(() => {
        saveStoredState({
          conversations: nextConversations,
          activeConversationId: nextActiveId,
        });
      }, 300);
    },
    [],
  );

  const updateConversations = useCallback(
    (
      updater: (prev: Conversation[]) => Conversation[],
      activeId: string | null = activeConversationId,
    ) => {
      setConversations((prev) => {
        const next = updater(prev);
        persistSoon(next, activeId);
        return next;
      });
    },
    [activeConversationId, persistSoon],
  );

  useEffect(() => {
    const stored = loadStoredState();
    let list = stored.conversations;
    let activeId = stored.activeConversationId;

    if (list.length === 0) {
      const c = createConversation();
      list = [c];
      activeId = c.id;
    } else if (!activeId || !list.some((c) => c.id === activeId)) {
      activeId = list.sort((a, b) => b.updatedAt - a.updatedAt)[0]!.id;
    }

    setConversations(list);
    setActiveConversationId(activeId);
    const active = list.find((c) => c.id === activeId);
    const turn = active ? lastTurn(active.turns) : null;
    applyTurnToPanel(turn, setResults);
    if (turn) setSelectedTurnId(turn.id);
  }, []);

  const refreshStatus = useCallback(async () => {
    try {
      const s = await fetchStatus();
      setIndexedCount(s.total_records ?? s.indexed_count);
      setStatusError(null);
    } catch (e) {
      setStatusError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    const bootstrapMs = 3200;
    const started = Date.now();

    void (async () => {
      await refreshStatus();
      const wait = bootstrapMs - (Date.now() - started);
      if (wait > 0) {
        await new Promise((resolve) => window.setTimeout(resolve, wait));
      }
      if (!cancelled) setAppReady(true);
    })();

    return () => {
      cancelled = true;
    };
  }, [refreshStatus]);

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
        // Stale localStorage job ID with nothing on the server — unlock uploads.
        if (isMissingIngestJobError(e)) {
          setIngestMessage(
            "Cleared a stuck ingest that was no longer on the server. You can upload again.",
          );
          clearActiveIngest();
          return;
        }
        setIngestMessage(e instanceof Error ? e.message : String(e));
        // Keep a progress row so Cancel remains available while retrying.
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
    if (corpusOpen) {
      void refreshCatalog();
    }
  }, [corpusOpen, refreshCatalog, indexedCount]);

  useEffect(() => {
    if (turns.length > 0) {
      setSuggestions([]);
      setSuggestionsLoading(false);
      return;
    }

    const controller = new AbortController();

    setSuggestionsLoading(true);
    fetchSuggestions()
      .then((res) => {
        if (controller.signal.aborted) return;
        setSuggestions(res.suggestions);
      })
      .catch(() => {
        if (controller.signal.aborted) return;
        setSuggestions([]);
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setSuggestionsLoading(false);
        }
      });

    return () => controller.abort();
  }, [turns.length, indexedCount]);

  const selectConversation = useCallback(
    (id: string, turnId?: string | null) => {
      setActiveConversationId(id);
      persistSoon(conversations, id);
      const c = conversations.find((x) => x.id === id);
      let turn = c ? lastTurn(c.turns) : null;
      if (c && turnId) {
        const matched = c.turns.find((t) => t.id === turnId);
        if (matched) turn = matched;
      }
      setSelectedTurnId(turn?.id ?? null);
      applyTurnToPanel(turn, setResults);
      setError(null);
    },
    [conversations, persistSoon],
  );

  const handleNewChat = useCallback(() => {
    const c = createConversation();
    setConversations((prev) => {
      const next = [c, ...prev];
      persistSoon(next, c.id);
      return next;
    });
    setActiveConversationId(c.id);
    setSelectedTurnId(null);
    setResults([]);
    setError(null);
    setInput("");
  }, [persistSoon]);

  const handleDeleteChat = useCallback(
    (id: string) => {
      setConversations((prev) => {
        const next = prev.filter((c) => c.id !== id);
        let newActive = activeConversationId;
        if (activeConversationId === id) {
          if (next.length === 0) {
            const c = createConversation();
            next.push(c);
            newActive = c.id;
            setActiveConversationId(c.id);
            setSelectedTurnId(null);
            setResults([]);
          } else {
            newActive = next.sort((a, b) => b.updatedAt - a.updatedAt)[0]!.id;
            setActiveConversationId(newActive);
            const active = next.find((c) => c.id === newActive)!;
            const turn = lastTurn(active.turns);
            setSelectedTurnId(turn?.id ?? null);
            applyTurnToPanel(turn, setResults);
          }
        }
        persistSoon(next, newActive);
        return next;
      });
    },
    [activeConversationId, persistSoon],
  );

  const handleSelectTurn = useCallback(
    (turnId: string) => {
      if (!activeConversation) return;
      const turn = activeConversation.turns.find((t) => t.id === turnId);
      if (!turn) return;
      setSelectedTurnId(turnId);
      setSearchEventId(turn?.searchEventId ?? null);
      applyTurnToPanel(turn, setResults);
    },
    [activeConversation],
  );

  const runSearch = async (
    text: string,
    effectiveTopK: number,
    effectiveMinMatchPercent: number,
  ) => {
    let convId = activeConversationId;
    let conv = activeConversation;
    if (!conv || !convId) {
      const c = createConversation();
      conv = c;
      convId = c.id;
      setConversations((prev) => {
        const next = [c, ...prev];
        persistSoon(next, c.id);
        return next;
      });
      setActiveConversationId(c.id);
    }

    setError(null);
    setLoading(true);
    setInput("");
    const turnId = newTurnId();
    const sessionId = conv.sessionId;

    const pendingTurn: ConversationTurn = {
      id: turnId,
      userContent: text,
      assistantContent: "",
      results: [],
      parsedQuery: null,
    };
    updateConversations((prev) =>
      prev.map((c) => {
        if (c.id !== convId) return c;
        const title = c.turns.length === 0 ? titleFromMessage(text) : c.title;
        return {
          ...c,
          title,
          updatedAt: Date.now(),
          turns: [...c.turns, pendingTurn],
        };
      }),
    );
    setSelectedTurnId(turnId);

    let streamedContent = "";

    try {
      await sendChatStream(
        text,
        sessionId,
        effectiveTopK,
        effectiveMinMatchPercent,
        {
        onMetadata: (meta) => {
          setSearchEventId(meta.search_event_id ?? null);
          updateConversations((prev) =>
            prev.map((c) => {
              if (c.id !== convId) return c;
              return {
                ...c,
                sessionId: meta.session_id,
                updatedAt: Date.now(),
                turns: c.turns.map((t) =>
                  t.id === turnId
                    ? {
                        ...t,
                        results: meta.results,
                        parsedQuery: meta.parsed_query ?? null,
                        searchEventId: meta.search_event_id ?? null,
                      }
                    : t,
                ),
              };
            }),
          );
          setResults(meta.results);
        },
        onToken: (chunk) => {
          streamedContent += chunk;
          const content = streamedContent;
          updateConversations((prev) =>
            prev.map((c) => {
              if (c.id !== convId) return c;
              return {
                ...c,
                updatedAt: Date.now(),
                turns: c.turns.map((t) =>
                  t.id === turnId ? { ...t, assistantContent: content } : t,
                ),
              };
            }),
          );
        },
        onDone: (assistantMessage, followUpSuggestions) => {
          updateConversations((prev) =>
            prev.map((c) => {
              if (c.id !== convId) return c;
              return {
                ...c,
                updatedAt: Date.now(),
                turns: c.turns.map((t) =>
                  t.id === turnId
                    ? {
                        ...t,
                        assistantContent: assistantMessage,
                        followUpSuggestions,
                      }
                    : t,
                ),
              };
            }),
          );
          setSelectedTurnId(turnId);
        },
        onError: (detail) => {
          throw new Error(detail);
        },
      },
        searchSortBy,
      );
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      const errTurn: ConversationTurn = {
        id: turnId,
        userContent: text,
        assistantContent: `**Error:** ${msg}`,
        results: [],
        parsedQuery: null,
      };
      updateConversations((prev) =>
        prev.map((c) => {
          if (c.id !== convId) return c;
          return {
            ...c,
            updatedAt: Date.now(),
            turns: c.turns.map((t) => (t.id === turnId ? errTurn : t)),
          };
        }),
      );
      setSelectedTurnId(turnId);
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const applySimilarResponse = async (
    userLabel: string,
    fetchSimilar: (sessionId: string | null) => ReturnType<typeof searchSimilarByImage>,
  ) => {
    let convId = activeConversationId;
    let conv = activeConversation;
    if (!conv || !convId) {
      const c = createConversation();
      conv = c;
      convId = c.id;
      setConversations((prev) => {
        const next = [c, ...prev];
        persistSoon(next, c.id);
        return next;
      });
      setActiveConversationId(c.id);
    }

    setError(null);
    setLoading(true);

    const turnId = newTurnId();
    const sessionId = conv.sessionId;

    const pendingTurn: ConversationTurn = {
      id: turnId,
      userContent: userLabel,
      assistantContent: "",
      results: [],
      parsedQuery: null,
    };
    updateConversations((prev) =>
      prev.map((c) => {
        if (c.id !== convId) return c;
        const title =
          c.turns.length === 0 ? titleFromMessage(userLabel) : c.title;
        return {
          ...c,
          title,
          updatedAt: Date.now(),
          turns: [...c.turns, pendingTurn],
        };
      }),
    );
    setSelectedTurnId(turnId);

    try {
      const res = await fetchSimilar(sessionId);
      updateConversations((prev) =>
        prev.map((c) => {
          if (c.id !== convId) return c;
          return {
            ...c,
            sessionId: res.session_id ?? c.sessionId,
            updatedAt: Date.now(),
            turns: c.turns.map((t) =>
              t.id === turnId
                ? {
                    ...t,
                    assistantContent: res.assistant_message,
                    results: res.results,
                    parsedQuery: res.parsed_query ?? null,
                  }
                : t,
            ),
          };
        }),
      );
      setResults(res.results);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
      updateConversations((prev) =>
        prev.map((c) => {
          if (c.id !== convId) return c;
          return {
            ...c,
            updatedAt: Date.now(),
            turns: c.turns.map((t) =>
              t.id === turnId
                ? {
                    ...t,
                    assistantContent: `**Error:** ${msg}`,
                    results: [],
                    parsedQuery: null,
                  }
                : t,
            ),
          };
        }),
      );
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const handleSimilarImageSearch = (file: File) => {
    if (loading) return;
    void applySimilarResponse(`[Image search] ${file.name}`, (sessionId) =>
      searchSimilarByImage(
        file,
        sessionId,
        topK,
        minMatchPercent,
        similarityAxis,
        searchSortBy,
      ),
    );
  };

  const handleSimilarFromResult = (imageId: string, imageName: string) => {
    if (loading) return;
    void applySimilarResponse(`[Find similar] ${imageName}`, (sessionId) =>
      searchSimilarByImageId(
        imageId,
        sessionId,
        topK,
        minMatchPercent,
        similarityAxis,
        searchSortBy,
      ),
    );
  };

  const handleSend = async () => {
    const text = input.trim();
    if (!text || loading) return;
    await runSearch(text, topK, minMatchPercent);
  };

  const handleFollowUp = (text: string) => {
    if (loading) return;
    void runSearch(text, topK, minMatchPercent);
  };

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

    // Abort an in-flight chunked upload and cancel the staging job if created.
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

    // No server job id (or only a local lock): unlock immediately.
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
      // Missing/stale job or unreachable cancel: always unlock the drawer.
      setIngestMessage(
        isMissingIngestJobError(e)
          ? "Cleared a stuck ingest that was no longer on the server. You can upload again."
          : `Could not cancel on the server (${msg}). Cleared the local ingest lock so you can upload again.`,
      );
      clearActiveIngest();
    }
  };

  const showLoadingScreen = useMinDurationLoading(!appReady, 3000);

  return (
    <div className="flex h-dvh min-h-screen flex-col overflow-hidden">
      <AtlasLoadingScreen visible={showLoadingScreen} />
      <div className="shrink-0">
        <Header
          indexedCount={indexedCount}
          statusError={statusError}
          onOpenCorpus={() => setCorpusOpen(true)}
        />
      </div>

      {error && (
        <div className="shrink-0 px-6 py-2">
          <div className="rounded-lg bg-red-50 px-4 py-2 text-sm text-red-700 ring-1 ring-red-100">
            {error}
          </div>
        </div>
      )}

      <main className="flex min-h-0 flex-1 flex-row">
        {/* Left — search & chats (slightly narrower than results) */}
        <div className="flex min-h-0 min-w-0 flex-[5] flex-col border-r border-navy-200">
          <div className="flex min-h-0 flex-1">
            <ChatSidebar
              conversations={conversations}
              activeId={activeConversationId}
              collapsed={sidebarCollapsed}
              onToggleCollapsed={() => setSidebarCollapsed((v) => !v)}
              onSelect={selectConversation}
              onNewChat={handleNewChat}
              onDelete={handleDeleteChat}
            />

            <section className="flex min-h-0 min-w-0 flex-1 flex-col bg-white">
              <div className="flex shrink-0 items-center gap-2 border-b border-navy-100 bg-navy-50 px-4 py-2">
                <span className="text-xs font-semibold uppercase tracking-wide text-navy-700">
                  Search
                </span>
                {sidebarCollapsed && (
                  <button
                    type="button"
                    onClick={() => setSidebarCollapsed(false)}
                    className="text-xs font-medium text-brand-600 hover:text-brand-500 hover:underline"
                  >
                    Show chats
                  </button>
                )}
              </div>
              <div className="min-h-0 flex-1 overflow-y-auto">
                {turns.length === 0 ? (
                  <EmptyState
                    suggestions={suggestions}
                    loading={suggestionsLoading}
                    onPickExample={handleFollowUp}
                  />
                ) : (
                  <ChatMessageList
                    turns={turns}
                    loading={loading}
                    selectedTurnId={selectedTurnId}
                    onSelectTurn={handleSelectTurn}
                    onFollowUpClick={handleFollowUp}
                  />
                )}
              </div>
              <Composer
                value={input}
                topK={topK}
                minMatchPercent={minMatchPercent}
                similarityAxis={similarityAxis}
                loading={loading}
                onChange={setInput}
                onTopKChange={setTopK}
                onMinMatchPercentChange={setMinMatchPercent}
                onSimilarityAxisChange={setSimilarityAxis}
                onSend={handleSend}
                onSimilarImageSearch={handleSimilarImageSearch}
              />
            </section>
          </div>
        </div>

        {/* Right — results (more width for image grid) */}
        <section className="flex min-h-0 min-w-0 flex-[6] flex-col bg-white">
          <div className="flex shrink-0 items-center gap-2 border-b border-navy-100 bg-navy-50 px-4 py-2">
            <span className="text-xs font-semibold uppercase tracking-wide text-navy-700">
              Results
            </span>
            {results.length > 0 && (
              <span className="text-xs text-navy-500">
                {results.length} image{results.length !== 1 ? "s" : ""}
              </span>
            )}
            <div className="ml-auto">
              <SortSelect
                value={searchSortBy}
                onChange={setSearchSortBy}
                disabled={loading}
              />
            </div>
          </div>
          <ResultsGrid
            results={displayResults}
            loading={loading}
            onFindSimilar={handleSimilarFromResult}
            searchEventId={searchEventId}
            sessionId={activeConversation?.sessionId ?? null}
            topK={topK}
            minMatchPercent={minMatchPercent}
            similarityAxis={similarityAxis}
            onSimilarResults={(similarResults, newSearchEventId) => {
              setResults(similarResults);
              setSearchEventId(newSearchEventId ?? null);
            }}
          />
        </section>
      </main>

      <footer className="flex shrink-0 items-center justify-between gap-4 border-t border-navy-800 bg-navy-950 px-5 py-1.5 text-[11px] text-white/50">
        <span>
          <span className="font-semibold text-white/80">ATLAS</span>
          {" · "}
          {statusError
            ? "index status unavailable"
            : `${indexedCount} indexed images`}
        </span>
        <AdminNavLink variant="footer" />
      </footer>

      <CorpusDrawer
        open={corpusOpen}
        onClose={() => setCorpusOpen(false)}
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
    </div>
  );
}
