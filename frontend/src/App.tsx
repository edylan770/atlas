import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  createImageSession,
  fetchStatus,
  fetchSuggestions,
  postEditTurn,
  reviseEditTurn,
  searchSimilarByImage,
  searchSimilarByImageId,
  sendChatStream,
  submitEditSession,
} from "./api/client";
import {
  createConversation,
  lastTurn,
  loadStoredState,
  newTurnId,
  saveStoredState,
  titleFromMessage,
} from "./chat/storage";
import { AdminNavLink } from "./components/AdminNavLink";
import { ChatMessageList } from "./components/ChatMessageList";
import { ChatSidebar } from "./components/ChatSidebar";
import { Composer } from "./components/Composer";
import { CreatedImagePanel } from "./components/CreatedImagePanel";
import { EmptyState } from "./components/EmptyState";
import { AtlasLoadingScreen, useMinDurationLoading } from "./components/AtlasLoadingScreen";
import { Header } from "./components/Header";
import { ResultsGrid } from "./components/ResultsGrid";
import { SortSelect } from "./components/SortSelect";
import { defaultSearchSort, sortResultCards } from "./sortResults";
import { downloadImageUrl } from "./imageDownload";
import type {
  Conversation,
  ConversationTurn,
  ResultCard,
  ResultSort,
} from "./types";

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

  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);
  const [searchEventId, setSearchEventId] = useState<string | null>(null);
  const [searchSortBy, setSearchSortBy] = useState<ResultSort>(defaultSearchSort());
  const [addingCreatedId, setAddingCreatedId] = useState<string | null>(null);

  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Read by async callbacks (streams, image creation) that outlive the render
  // they were created in; the state value in their closure goes stale.
  const activeIdRef = useRef<string | null>(null);
  const streamRef = useRef<{ controller: AbortController; convId: string } | null>(
    null,
  );
  const createAbortRef = useRef<AbortController | null>(null);

  const setActiveId = useCallback((id: string | null) => {
    activeIdRef.current = id;
    setActiveConversationId(id);
  }, []);

  /** Abort the in-flight chat stream unless it belongs to ``keepConvId``. */
  const cancelStream = useCallback((keepConvId: string | null) => {
    const stream = streamRef.current;
    if (stream && stream.convId !== keepConvId) {
      stream.controller.abort();
    }
  }, []);

  useEffect(
    () => () => {
      streamRef.current?.controller.abort();
      createAbortRef.current?.abort();
    },
    [],
  );

  const activeConversation = useMemo(
    () => conversations.find((c) => c.id === activeConversationId) ?? null,
    [conversations, activeConversationId],
  );

  const turns = useMemo(
    () => activeConversation?.turns ?? [],
    [activeConversation],
  );

  const createMode = activeConversation?.promptMode === "create";
  const selectedTurn = useMemo(
    () => turns.find((t) => t.id === selectedTurnId) ?? null,
    [turns, selectedTurnId],
  );
  const displayedCreateTurn = useMemo(() => {
    if (!createMode) return null;
    if (selectedTurn?.kind === "create" && selectedTurn.createdImageUrl) {
      return selectedTurn;
    }
    return (
      [...turns].reverse().find((t) => t.kind === "create" && t.createdImageUrl) ??
      null
    );
  }, [createMode, selectedTurn, turns]);

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
      activeId: string | null = activeIdRef.current,
    ) => {
      setConversations((prev) => {
        const next = updater(prev);
        persistSoon(next, activeId);
        return next;
      });
    },
    [persistSoon],
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
    setActiveId(activeId);
    const active = list.find((c) => c.id === activeId);
    const turn = active ? lastTurn(active.turns) : null;
    applyTurnToPanel(turn, setResults);
    setSearchEventId(turn?.searchEventId ?? null);
    if (turn) setSelectedTurnId(turn.id);
  }, [setActiveId]);

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
    const statusTimeoutMs = 5000;
    const started = Date.now();

    void (async () => {
      await Promise.race([
        refreshStatus(),
        new Promise<void>((resolve) => {
          window.setTimeout(resolve, statusTimeoutMs);
        }),
      ]);
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

  useEffect(() => {
    if (turns.length > 0) {
      setSuggestions([]);
      setSuggestionsLoading(false);
      return;
    }

    let active = true;
    const controller = new AbortController();

    setSuggestionsLoading(true);
    fetchSuggestions(4, { signal: controller.signal })
      .then((res) => {
        if (!active) return;
        setSuggestions(res.suggestions);
      })
      .catch(() => {
        if (!active) return;
        setSuggestions([]);
      })
      .finally(() => {
        if (active) {
          setSuggestionsLoading(false);
        }
      });

    return () => {
      active = false;
      controller.abort();
    };
  }, [turns.length, indexedCount]);

  const selectConversation = useCallback(
    (id: string, turnId?: string | null) => {
      cancelStream(id);
      setActiveId(id);
      persistSoon(conversations, id);
      const c = conversations.find((x) => x.id === id);
      let turn = c ? lastTurn(c.turns) : null;
      if (c && turnId) {
        const matched = c.turns.find((t) => t.id === turnId);
        if (matched) turn = matched;
      }
      setSelectedTurnId(turn?.id ?? null);
      setSearchEventId(turn?.searchEventId ?? null);
      applyTurnToPanel(turn, setResults);
      setError(null);
    },
    [conversations, persistSoon, cancelStream, setActiveId],
  );

  const handleNewChat = useCallback(() => {
    const c = createConversation();
    cancelStream(c.id);
    setConversations((prev) => {
      const next = [c, ...prev];
      persistSoon(next, c.id);
      return next;
    });
    setActiveId(c.id);
    setSelectedTurnId(null);
    setSearchEventId(null);
    setResults([]);
    setError(null);
    setInput("");
  }, [persistSoon, cancelStream, setActiveId]);

  const handleDeleteChat = useCallback(
    (id: string) => {
      if (streamRef.current?.convId === id || activeConversationId === id) {
        cancelStream(null);
      }
      setConversations((prev) => {
        const next = prev.filter((c) => c.id !== id);
        let newActive = activeConversationId;
        if (activeConversationId === id) {
          if (next.length === 0) {
            const c = createConversation();
            next.push(c);
            newActive = c.id;
            setActiveId(c.id);
            setSelectedTurnId(null);
            setSearchEventId(null);
            setResults([]);
          } else {
            newActive = next.sort((a, b) => b.updatedAt - a.updatedAt)[0]!.id;
            setActiveId(newActive);
            const active = next.find((c) => c.id === newActive)!;
            const turn = lastTurn(active.turns);
            setSelectedTurnId(turn?.id ?? null);
            setSearchEventId(turn?.searchEventId ?? null);
            applyTurnToPanel(turn, setResults);
          }
        }
        persistSoon(next, newActive);
        return next;
      });
    },
    [activeConversationId, persistSoon, cancelStream, setActiveId],
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
    options?: { resetSession?: boolean },
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
      setActiveId(c.id);
    }

    setError(null);
    setLoading(true);
    setInput("");
    const turnId = newTurnId();
    const sessionId = options?.resetSession ? null : conv.sessionId;

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

    cancelStream(null);
    const controller = new AbortController();
    streamRef.current = { controller, convId };

    let streamedContent = "";
    // Tokens arrive far faster than frames; write at most once per frame.
    let tokenFrame: number | null = null;
    const cancelTokenFrame = () => {
      if (tokenFrame !== null) {
        cancelAnimationFrame(tokenFrame);
        tokenFrame = null;
      }
    };
    const writeStreamed = () => {
      tokenFrame = null;
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
    };
    let failure = null as string | null;

    try {
      await sendChatStream(
        text,
        sessionId,
        effectiveTopK,
        effectiveMinMatchPercent,
        {
        onMetadata: (meta) => {
          if (controller.signal.aborted) return;
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
          if (controller.signal.aborted) return;
          streamedContent += chunk;
          if (tokenFrame === null) {
            tokenFrame = requestAnimationFrame(writeStreamed);
          }
        },
        onDone: (assistantMessage, followUpSuggestions) => {
          if (controller.signal.aborted) return;
          cancelTokenFrame();
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
          failure = detail;
        },
      },
        searchSortBy,
        controller.signal,
      );
      if (failure !== null) throw new Error(failure);
    } catch (e) {
      cancelTokenFrame();
      if (controller.signal.aborted) {
        const partial = streamedContent;
        updateConversations((prev) =>
          prev.map((c) => {
            if (c.id !== convId) return c;
            return {
              ...c,
              turns: c.turns.map((t) =>
                t.id === turnId
                  ? { ...t, assistantContent: partial || "_Search cancelled._" }
                  : t,
              ),
            };
          }),
        );
        return;
      }
      const raw = e instanceof Error ? e.message : String(e);
      const trimmed = raw.trim();
      const generic =
        !trimmed ||
        trimmed === "Error" ||
        trimmed === "Internal Server Error" ||
        trimmed === "Failed to fetch";
      const detail = generic
        ? "Search request failed. The server may be busy — try again in a moment."
        : trimmed.replace(/^Error:\s*/i, "");
      const msg = detail;
      setError(msg);
      updateConversations((prev) =>
        prev.map((c) => {
          if (c.id !== convId) return c;
          return {
            ...c,
            updatedAt: Date.now(),
            turns: c.turns.map((t) => {
              if (t.id !== turnId) return t;
              return {
                ...t,
                assistantContent: `**Error:** ${msg}`,
                // Keep any results already applied from SSE metadata.
                results: t.results ?? [],
                parsedQuery: t.parsedQuery ?? null,
              };
            }),
          };
        }),
      );
      setSelectedTurnId(turnId);
    } finally {
      cancelTokenFrame();
      // A newer stream may already own the loading state.
      if (!streamRef.current || streamRef.current.controller === controller) {
        streamRef.current = null;
        setLoading(false);
      }
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
      setActiveId(c.id);
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

  const handleCreateModeChange = (on: boolean) => {
    if (!activeConversationId) return;
    updateConversations((prev) =>
      prev.map((c) =>
        c.id === activeConversationId
          ? { ...c, promptMode: on ? "create" : "search", updatedAt: Date.now() }
          : c,
      ),
    );
  };

  const runCreate = async (
    text: string,
    options?: {
      fresh?: boolean;
      revise?: { sessionId: string; baseTurnIndex: number } | null;
    },
  ) => {
    let convId = activeConversationId;
    let conv = activeConversation;
    if (!conv || !convId) {
      const c = createConversation();
      c.promptMode = "create";
      conv = c;
      convId = c.id;
      setConversations((prev) => {
        const next = [c, ...prev];
        persistSoon(next, c.id);
        return next;
      });
      setActiveId(c.id);
    }

    setError(null);
    setLoading(true);
    setInput("");
    const turnId = newTurnId();
    const lastCreate = [...conv.turns]
      .reverse()
      .find((t) => t.kind === "create");
    const revising = options?.revise ?? null;
    const sessionToRefine = options?.fresh
      ? null
      : revising
        ? revising.sessionId
        : conv.createSessionId && lastCreate && !lastCreate.createSubmitted
          ? conv.createSessionId
          : null;

    const pendingTurn: ConversationTurn = {
      id: turnId,
      userContent: text,
      assistantContent: sessionToRefine ? "Refining the image…" : "Creating image…",
      results: [],
      parsedQuery: null,
      kind: "create",
      createSessionId: sessionToRefine,
    };
    updateConversations((prev) =>
      prev.map((c) => {
        if (c.id !== convId) return c;
        const title = c.turns.length === 0 ? titleFromMessage(text) : c.title;
        return {
          ...c,
          title,
          promptMode: "create",
          updatedAt: Date.now(),
          turns: [...c.turns, pendingTurn],
        };
      }),
    );
    setSelectedTurnId(turnId);

    createAbortRef.current?.abort();
    const controller = new AbortController();
    createAbortRef.current = controller;
    const opts = { signal: controller.signal };

    try {
      let session;
      let refined = Boolean(sessionToRefine);
      if (revising) {
        try {
          session = await reviseEditTurn(
            revising.sessionId,
            text,
            revising.baseTurnIndex,
            opts,
          );
        } catch (err: unknown) {
          const message = err instanceof Error ? err.message : String(err);
          if (!/not found/i.test(message)) throw err;
          refined = false;
          session = await createImageSession(text, opts);
        }
      } else {
        session = sessionToRefine
          ? await postEditTurn(sessionToRefine, text, opts)
          : await createImageSession(text, opts);
      }
      const imageUrl =
        session.turns[session.turns.length - 1]?.image_url ?? session.image_url;
      const createTurnIndex = Math.max(0, session.turns.length - 1);
      updateConversations((prev) =>
        prev.map((c) => {
          if (c.id !== convId) return c;
          return {
            ...c,
            createSessionId: session.session_id,
            updatedAt: Date.now(),
            turns: c.turns.map((t) =>
              t.id === turnId
                ? {
                    ...t,
                    assistantContent: refined
                      ? "Refined the previous image."
                      : "Created an image from your prompt.",
                    createdImageUrl: imageUrl,
                    createSessionId: session.session_id,
                    createTurnIndex,
                    createSubmitted: false,
                  }
                : t,
            ),
          };
        }),
      );
      setSelectedTurnId(turnId);
    } catch (e) {
      if (controller.signal.aborted) return;
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
                ? { ...t, assistantContent: `**Error:** ${msg}` }
                : t,
            ),
          };
        }),
      );
    } finally {
      if (createAbortRef.current === controller) {
        createAbortRef.current = null;
        setLoading(false);
      }
    }
  };

  const handleAddCreated = async (turn: ConversationTurn) => {
    const sessionId = turn.createSessionId;
    if (!sessionId || turn.createSubmitted || addingCreatedId) return;
    setAddingCreatedId(sessionId);
    setError(null);
    try {
      await submitEditSession(sessionId);
      updateConversations((prev) =>
        prev.map((c) => {
          if (c.id !== activeConversationId) return c;
          return {
            ...c,
            createSessionId:
              c.createSessionId === sessionId ? null : c.createSessionId,
            updatedAt: Date.now(),
            turns: c.turns.map((t) =>
              t.createSessionId === sessionId
                ? { ...t, createSubmitted: true }
                : t,
            ),
          };
        }),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setAddingCreatedId(null);
    }
  };

  const handleDownloadCreated = async (turn: ConversationTurn) => {
    if (!turn.createdImageUrl) return;
    setError(null);
    try {
      const slug = turn.userContent
        .trim()
        .replace(/[^\w\- ]+/g, "")
        .replace(/\s+/g, "-")
        .slice(0, 40);
      await downloadImageUrl(
        turn.createdImageUrl,
        `created-${slug || turn.id}.png`,
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  const handleSend = async () => {
    const text = input.trim();
    if (!text || loading) return;
    if (createMode) {
      await runCreate(text);
      return;
    }
    await runSearch(text, topK, minMatchPercent);
  };

  const handleFollowUp = (text: string) => {
    if (loading) return;
    void runSearch(text, topK, minMatchPercent);
  };

  const handleEditResubmit = (turnId: string, text: string) => {
    const trimmed = text.trim();
    if (!trimmed || loading || !activeConversation || !activeConversationId) {
      return;
    }
    const index = activeConversation.turns.findIndex((t) => t.id === turnId);
    if (index < 0) return;

    const editing = activeConversation.turns[index]!;
    const kept = activeConversation.turns.slice(0, index);
    const editingFirst = index === 0;

    if (editing.kind === "create") {
      const previousCreate = [...kept]
        .reverse()
        .find((t) => t.kind === "create" && t.createdImageUrl);
      const revise =
        previousCreate?.createSessionId &&
        !previousCreate.createSubmitted &&
        typeof previousCreate.createTurnIndex === "number" &&
        previousCreate.createSessionId === activeConversation.createSessionId
          ? {
              sessionId: previousCreate.createSessionId,
              baseTurnIndex: previousCreate.createTurnIndex,
            }
          : null;

      updateConversations((prev) =>
        prev.map((c) => {
          if (c.id !== activeConversationId) return c;
          return {
            ...c,
            turns: kept,
            createSessionId: null,
            updatedAt: Date.now(),
            title: editingFirst ? titleFromMessage(trimmed) : c.title,
          };
        }),
      );
      void runCreate(trimmed, revise ? { revise } : { fresh: true });
      return;
    }

    updateConversations((prev) =>
      prev.map((c) => {
        if (c.id !== activeConversationId) return c;
        return {
          ...c,
          turns: kept,
          sessionId: null,
          updatedAt: Date.now(),
          title: editingFirst ? titleFromMessage(trimmed) : c.title,
        };
      }),
    );

    const lastKept = kept.length > 0 ? kept[kept.length - 1]! : null;
    if (lastKept) {
      setSelectedTurnId(lastKept.id);
      setSearchEventId(lastKept.searchEventId ?? null);
      applyTurnToPanel(lastKept, setResults);
    } else {
      setSelectedTurnId(null);
      setSearchEventId(null);
      setResults([]);
    }

    void runSearch(trimmed, topK, minMatchPercent, { resetSession: true });
  };

  const showLoadingScreen = useMinDurationLoading(!appReady, 3000);

  return (
    <div className="flex h-dvh min-h-screen flex-col overflow-hidden">
      <AtlasLoadingScreen visible={showLoadingScreen} />
      <div className="shrink-0">
        <Header
          indexedCount={indexedCount}
          statusError={statusError}
        />
      </div>

      {error && (
        <div className="shrink-0 px-6 py-2">
          <div
            role="alert"
            className="rounded-lg bg-red-50 px-4 py-2 text-sm text-red-700 ring-1 ring-red-100"
          >
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
                  {createMode ? "Create image" : "Search"}
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
                    mode={createMode ? "create" : "search"}
                  />
                ) : (
                  <ChatMessageList
                    turns={turns}
                    loading={loading}
                    selectedTurnId={selectedTurnId}
                    onSelectTurn={handleSelectTurn}
                    onFollowUpClick={handleFollowUp}
                    onEditResubmit={handleEditResubmit}
                    onAddCreated={(turn) => void handleAddCreated(turn)}
                    onDownloadCreated={(turn) => void handleDownloadCreated(turn)}
                    addingCreatedId={addingCreatedId}
                  />
                )}
              </div>
              <Composer
                value={input}
                topK={topK}
                minMatchPercent={minMatchPercent}
                similarityAxis={similarityAxis}
                loading={loading}
                createMode={createMode}
                onChange={setInput}
                onTopKChange={setTopK}
                onMinMatchPercentChange={setMinMatchPercent}
                onSimilarityAxisChange={setSimilarityAxis}
                onCreateModeChange={handleCreateModeChange}
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
              {createMode ? "Created image" : "Results"}
            </span>
            {!createMode && results.length > 0 && (
              <span className="text-xs text-navy-500">
                {results.length} image{results.length !== 1 ? "s" : ""}
              </span>
            )}
            {!createMode && (
            <div className="ml-auto">
              <SortSelect
                value={searchSortBy}
                onChange={setSearchSortBy}
                disabled={loading}
              />
            </div>
            )}
          </div>
          {createMode ? (
            <CreatedImagePanel
              turn={displayedCreateTurn}
              loading={loading}
              adding={addingCreatedId === displayedCreateTurn?.createSessionId}
              onDownload={(turn) => void handleDownloadCreated(turn)}
              onAdd={(turn) => void handleAddCreated(turn)}
            />
          ) : (
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
          )}
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
    </div>
  );
}
