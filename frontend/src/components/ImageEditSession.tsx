import { useCallback, useEffect, useRef, useState } from "react";

import {
  createEditSession,
  postEditTurn,
  submitEditSession,
  type EditSessionState,
} from "../api/client";
import { downloadCardImage } from "../imageDownload";
import type { ResultCard as ResultCardType } from "../types";

interface ImageEditSessionProps {
  card: ResultCardType;
  onClose: () => void;
  /** Return to the lightbox preview without fully dismissing. */
  onBack?: () => void;
}

async function downloadPng(url: string, filename: string): Promise<void> {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error("Download failed");
  const blob = await res.blob();
  const objectUrl = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = objectUrl;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(objectUrl);
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
  const [submitOk, setSubmitOk] = useState(false);
  const [imageBump, setImageBump] = useState(0);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  const displayName =
    card.image_name || card.provenance.source_name || "Image";
  const editingUnavailable = Boolean(bootError) && !session;

  useEffect(() => {
    let cancelled = false;
    setBusy(true);
    setBootError(null);
    void createEditSession(card.image_id)
      .then((s) => {
        if (!cancelled) setSession(s);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setBootError(err instanceof Error ? err.message : String(err));
        }
      })
      .finally(() => {
        if (!cancelled) setBusy(false);
      });
    return () => {
      cancelled = true;
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

  const imageSrc = session
    ? `${session.image_url}?t=${imageBump}`
    : card.image_url || card.thumb_url || "";

  const runTurn = useCallback(async () => {
    if (!session || busy || submitOk || editingUnavailable) return;
    const text = prompt.trim();
    if (!text) return;
    setBusy(true);
    setActionError(null);
    try {
      const next = await postEditTurn(session.session_id, text);
      setSession(next);
      setPrompt("");
      setImageBump((n) => n + 1);
      inputRef.current?.focus();
    } catch (err: unknown) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }, [session, busy, submitOk, editingUnavailable, prompt]);

  const handleDownload = async () => {
    setActionError(null);
    try {
      if (session) {
        await downloadPng(
          `${session.image_url}?t=${imageBump}`,
          `edited-${session.source_image_id}.png`,
        );
      } else {
        await downloadCardImage(card);
      }
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
          <div className="relative flex min-h-[36vh] items-center justify-center overflow-hidden bg-navy-50 p-4 md:min-h-0">
            {imageSrc ? (
              <img
                src={imageSrc}
                alt={`Working edit of ${displayName}`}
                className="max-h-full max-w-full object-contain"
              />
            ) : (
              <p className="text-sm text-navy-500">Image unavailable</p>
            )}
            {busy && !editingUnavailable && (
              <span
                role="status"
                className="absolute bottom-3 right-3 rounded-full bg-navy-900/80 px-2.5 py-1 text-[10px] font-medium text-white"
              >
                Working…
              </span>
            )}
          </div>

          <div className="flex min-h-0 flex-col border-t border-navy-100 md:border-l md:border-t-0">
            <div className="min-h-0 flex-1 space-y-2 overflow-y-auto p-4">
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
                Describe changes in plain language. Each reply updates the image on the left.
              </p>
              {(session?.turns ?? []).map((t, i) => (
                <div
                  key={`${i}-${t.prompt}`}
                  className="rounded-lg bg-brand-50 px-3 py-2 text-sm text-navy-800 ring-1 ring-brand-100"
                >
                  <span className="mb-0.5 block text-[10px] font-semibold uppercase tracking-wide text-brand-700">
                    You
                  </span>
                  {t.prompt}
                </div>
              ))}
              {submitOk && (
                <div className="rounded-lg bg-emerald-50 px-3 py-2 text-sm text-emerald-900 ring-1 ring-emerald-200">
                  Submitted for admin review. An admin must accept it before it joins the corpus.
                </div>
              )}
              {actionError && (
                <p className="text-sm text-red-700" role="alert">
                  {actionError}
                </p>
              )}
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
                  data-testid="edit-download"
                  onClick={() => void handleDownload()}
                  disabled={busy || (!session && !card.has_image_file)}
                  className="rounded-md border border-navy-200 bg-white px-3 py-1.5 text-xs font-medium text-navy-800 transition hover:bg-navy-50 disabled:opacity-50"
                >
                  Download
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
