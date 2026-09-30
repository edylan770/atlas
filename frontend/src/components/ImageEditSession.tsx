import { useCallback, useEffect, useRef, useState } from "react";

import {
  createEditSession,
  discardEditSession,
  fetchEditStatus,
  postEditTurn,
  submitEditSession,
  type EditSessionState,
  type EditStatusResponse,
} from "../api/client";
import { downloadCardImage, downloadImageUrl } from "../imageDownload";
import type { ResultCard as ResultCardType } from "../types";

interface ImageEditSessionProps {
  card: ResultCardType;
  onClose: () => void;
  /** Return to the lightbox preview without fully dismissing. */
  onBack?: () => void;
}

function DownloadIcon({ className = "h-4 w-4" }: { className?: string }) {
  return (
    <svg
      className={className}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
      aria-hidden
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2M7 10l5 5m0 0l5-5m-5 5V4"
      />
    </svg>
  );
}

/**
 * Full-screen iterative Nano Banana edit chat launched from the lightbox.
 * When Gemini is unavailable, still shows the chrome so the flow can be previewed.
 */
export function ImageEditSession({ card, onClose, onBack }: ImageEditSessionProps) {
  const [session, setSession] = useState<EditSessionState | null>(null);
  const [prompt, setPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [bootError, setBootError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [editStatus, setEditStatus] = useState<EditStatusResponse | null>(null);
  const [originalLoadFailed, setOriginalLoadFailed] = useState(false);
  const [failedTurnImages, setFailedTurnImages] = useState<Set<string>>(
    () => new Set(),
  );
  const [submitOk, setSubmitOk] = useState(false);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  // Unsubmitted server session to release on close; cleared after submit.
  const liveSessionIdRef = useRef<string | null>(null);
  const turnAbortRef = useRef<AbortController | null>(null);

  const displayName =
    card.image_name || card.provenance.source_name || "Image";
  const editingUnavailable = Boolean(bootError) && !session;

  const originalSrc =
    session?.original_image_url ?? card.image_url ?? card.thumb_url ?? "";

  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    setBusy(true);
    setBootError(null);
    setSession(null);
    setEditStatus(null);
    setOriginalLoadFailed(false);
    setFailedTurnImages(new Set());
    void (async () => {
      try {
        const status = await fetchEditStatus();
        if (cancelled) return;
        setEditStatus(status);
        if (!status.available) {
          throw new Error(status.error || "Gemini configuration is unavailable");
        }
        const nextSession = await createEditSession(card.image_id, {
          signal: controller.signal,
        });
        if (cancelled) {
          void discardEditSession(nextSession.session_id);
          return;
        }
        liveSessionIdRef.current = nextSession.session_id;
        setSession(nextSession);
      } catch (err: unknown) {
        if (!cancelled) {
          setBootError(err instanceof Error ? err.message : String(err));
        }
      } finally {
        if (!cancelled) setBusy(false);
      }
    })();
    return () => {
      cancelled = true;
      controller.abort();
      turnAbortRef.current?.abort();
      const liveId = liveSessionIdRef.current;
      liveSessionIdRef.current = null;
      if (liveId) void discardEditSession(liveId);
    };
  }, [card.image_id]);

  useEffect(() => {
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus();
    return () => {
      document.body.style.overflow = previous;
    };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [session?.turns.length, busy, actionError, submitOk]);

  const runTurn = useCallback(async () => {
    if (!session || busy || submitOk || editingUnavailable) return;
    const text = prompt.trim();
    if (!text) return;
    setBusy(true);
    setActionError(null);
    const controller = new AbortController();
    turnAbortRef.current = controller;
    try {
      const next = await postEditTurn(session.session_id, text, {
        signal: controller.signal,
      });
      setSession(next);
      setFailedTurnImages(new Set());
      setPrompt("");
      inputRef.current?.focus();
    } catch (err: unknown) {
      if (controller.signal.aborted) return;
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      if (turnAbortRef.current === controller) turnAbortRef.current = null;
      if (!controller.signal.aborted) setBusy(false);
    }
  }, [session, busy, submitOk, editingUnavailable, prompt]);

  const handleDownloadOriginal = async () => {
    setActionError(null);
    try {
      if (session?.original_image_url) {
        await downloadImageUrl(
          session.original_image_url,
          `original-${session.source_image_id}.png`,
        );
      } else {
        await downloadCardImage(card);
      }
    } catch (err: unknown) {
      setActionError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleDownloadTurn = async (turnIndex: number, imageUrl: string) => {
    if (!session) return;
    setActionError(null);
    try {
      await downloadImageUrl(
        imageUrl,
        `edit-${session.source_image_id}-turn-${turnIndex + 1}.png`,
      );
    } catch (err: unknown) {
      setActionError(err instanceof Error ? err.message : String(err));
    }
  };

  const handleSubmit = async () => {
    if (!session || busy || submitOk || editingUnavailable) return;
    if (session.turn_count < 1) {
      setActionError("Edit the image at least once before adding to the database.");
      return;
    }
    setBusy(true);
    setActionError(null);
    try {
      await submitEditSession(session.session_id);
      liveSessionIdRef.current = null;
      setSubmitOk(true);
    } catch (err: unknown) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      data-testid="image-edit-session"
      role="dialog"
      aria-modal="true"
      aria-label={`Edit: ${displayName}`}
      className="fixed inset-0 z-[60] flex flex-col bg-navy-950/90 backdrop-blur-sm"
    >
      <div className="mx-auto flex h-full w-full max-w-6xl flex-col overflow-hidden bg-white shadow-2xl sm:my-3 sm:max-h-[calc(100vh-1.5rem)] sm:rounded-xl">
        <header className="flex shrink-0 items-center justify-between gap-3 border-b border-navy-100 bg-white px-4 py-2.5">
          <div className="flex min-w-0 items-center gap-1.5">
            {onBack && (
              <button
                type="button"
                data-testid="edit-session-back"
                onClick={onBack}
                aria-label="Back to preview"
                className="shrink-0 rounded-md p-1 text-navy-500 transition hover:bg-navy-100 hover:text-navy-900"
              >
                <svg
                  className="h-5 w-5"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                  aria-hidden
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M15 19l-7-7 7-7"
                  />
                </svg>
              </button>
            )}
            <div className="min-w-0">
              <p className="text-sm font-semibold text-navy-900">Edit image</p>
              <p className="truncate text-xs text-navy-500">{displayName}</p>
            </div>
          </div>
          <button
            ref={closeRef}
            type="button"
            data-testid="edit-session-close"
            onClick={onClose}
            aria-label="Close editor"
            className="rounded-md p-1 text-navy-500 transition hover:bg-navy-100 hover:text-navy-900"
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </header>

        <div className="grid min-h-0 flex-1 grid-cols-1 md:grid-cols-2">
          <div className="flex min-h-[36vh] flex-col items-center justify-center gap-3 bg-navy-50 p-4 md:min-h-0">
            <p className="text-[10px] font-semibold uppercase tracking-wide text-navy-500">
              Original
            </p>
            {originalSrc && !originalLoadFailed ? (
              <img
                src={originalSrc}
                alt={`Original ${displayName}`}
                onError={() => setOriginalLoadFailed(true)}
                className="max-h-[min(52vh,520px)] max-w-full object-contain"
              />
            ) : (
              <p
                data-testid="edit-original-image-error"
                className="text-sm text-red-700"
                role="alert"
              >
                Original image could not be loaded.
              </p>
            )}
            <button
              type="button"
              data-testid="edit-download-original"
              onClick={() => void handleDownloadOriginal()}
              disabled={busy || (!session && !card.has_image_file) || !originalSrc}
              className="inline-flex items-center gap-1.5 rounded-md border border-navy-200 bg-white px-3 py-1.5 text-xs font-medium text-navy-800 transition hover:bg-navy-50 disabled:opacity-50"
            >
              <DownloadIcon className="h-3.5 w-3.5" />
              Download original
            </button>
          </div>

          <div className="flex min-h-0 flex-col border-t border-navy-100 md:border-l md:border-t-0">
            <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-4">
              {editingUnavailable && (
                <div
                  data-testid="edit-unavailable-banner"
                  className="rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-950 ring-1 ring-amber-200"
                  role="status"
                >
                  <p className="font-semibold">Nano Banana editing is unavailable</p>
                  <p className="mt-1 text-xs text-amber-900/90">
                    {bootError}. You can still preview this screen; Generate and Add to
                    database stay disabled until a Gemini API key is configured (env or
                    Secrets Manager).
                  </p>
                </div>
              )}
              <p className="text-xs text-navy-500">
                Describe changes below. Edits appear in the chat; the original stays on
                the left.
              </p>
              {editStatus?.available && (
                <p
                  data-testid="edit-provider-status"
                  className="text-[10px] text-navy-400"
                >
                  {editStatus.model} via {editStatus.backend ?? "Gemini"}
                  {editStatus.location ? ` (${editStatus.location})` : ""}
                </p>
              )}
              {(session?.turns ?? []).map((turn, index) => (
                <div key={`${index}-${turn.prompt}`} className="space-y-2">
                  <div className="rounded-lg bg-brand-50 px-3 py-2 text-sm text-navy-800 ring-1 ring-brand-100">
                    <span className="mb-0.5 block text-[10px] font-semibold uppercase tracking-wide text-brand-700">
                      You
                    </span>
                    {turn.prompt}
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="min-w-0 flex-1">
                      <span className="mb-1 block text-[10px] font-semibold uppercase tracking-wide text-navy-500">
                        Edit {index + 1}
                      </span>
                      {failedTurnImages.has(turn.image_url) ? (
                        <div
                          data-testid={`edit-turn-image-error-${index}`}
                          className="rounded-lg bg-red-50 px-3 py-4 text-xs text-red-800 ring-1 ring-red-200"
                          role="alert"
                        >
                          The generated image could not be loaded. Retry the edit or
                          check the session/image request in server logs.
                        </div>
                      ) : (
                        <img
                          src={`${turn.image_url}${turn.image_url.includes("?") ? "&" : "?"}v=${index + 1}`}
                          alt={`Edit ${index + 1} of ${displayName}`}
                          onError={() =>
                            setFailedTurnImages((current) => {
                              const next = new Set(current);
                              next.add(turn.image_url);
                              return next;
                            })
                          }
                          className="max-w-full rounded-lg object-contain ring-1 ring-navy-100"
                        />
                      )}
                    </div>
                    <button
                      type="button"
                      data-testid={`edit-download-turn-${index}`}
                      onClick={() => void handleDownloadTurn(index, turn.image_url)}
                      title={`Download edit ${index + 1}`}
                      aria-label={`Download edit ${index + 1}`}
                      className="mt-5 shrink-0 rounded-md p-1.5 text-navy-600 transition hover:bg-navy-100 hover:text-brand-700"
                    >
                      <DownloadIcon />
                    </button>
                  </div>
                </div>
              ))}
              {busy && !editingUnavailable && (
                <p
                  role="status"
                  className="text-xs font-medium text-navy-500"
                >
                  Working…
                </p>
              )}
              {submitOk && (
                <div className="rounded-lg bg-emerald-50 px-3 py-2 text-sm text-emerald-900 ring-1 ring-emerald-200">
                  Submitted for admin review. An admin must accept it before it joins the corpus.
                </div>
              )}
              {actionError && (
                <div
                  data-testid="edit-action-error-banner"
                  className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-900 ring-1 ring-red-200"
                  role="alert"
                >
                  <p className="font-semibold">Edit failed</p>
                  <p className="mt-1 text-xs text-red-900/90">{actionError}</p>
                </div>
              )}
              <div ref={chatEndRef} />
            </div>

            <div className="shrink-0 border-t border-navy-100 p-3">
              <textarea
                ref={inputRef}
                data-testid="edit-prompt"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                disabled={busy || submitOk || editingUnavailable || !session}
                rows={3}
                placeholder={
                  editingUnavailable
                    ? "Editing disabled until Gemini is configured"
                    : "e.g. Make the background sky blue and remove the logo"
                }
                className="w-full resize-none rounded-lg border border-navy-200 px-3 py-2 text-sm text-navy-900 outline-none focus:border-brand-400 focus:ring-2 focus:ring-brand-200 disabled:bg-navy-50"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void runTurn();
                  }
                }}
              />
              <div className="mt-2 flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  data-testid="edit-send"
                  onClick={() => void runTurn()}
                  disabled={
                    busy ||
                    submitOk ||
                    editingUnavailable ||
                    !session ||
                    !prompt.trim()
                  }
                  className="rounded-md bg-brand-500 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-brand-600 disabled:opacity-50"
                >
                  Generate
                </button>
                <button
                  type="button"
                  data-testid="edit-submit"
                  onClick={() => void handleSubmit()}
                  disabled={
                    busy ||
                    submitOk ||
                    editingUnavailable ||
                    !session ||
                    (session?.turn_count ?? 0) < 1
                  }
                  className="rounded-md border border-brand-200 bg-brand-50 px-3 py-1.5 text-xs font-semibold text-brand-800 transition hover:bg-brand-100 disabled:opacity-50"
                >
                  Add to database
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
