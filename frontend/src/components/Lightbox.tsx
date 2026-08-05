import { useCallback, useEffect, useRef, useState } from "react";

import { downloadCardImage } from "../imageDownload";
import type { ResultCard as ResultCardType } from "../types";

interface LightboxProps {
  cards: ResultCardType[];
  index: number;
  onClose: () => void;
  onNavigate: (index: number) => void;
  onEdit?: (card: ResultCardType) => void;
}

/**
 * In-app image preview: opens instantly on the already-cached thumbnail and
 * swaps to the full-resolution image when it finishes loading. Esc / backdrop
 * click closes; arrow keys or on-screen buttons move through the result set.
 *
 * Layout: header / stage / footer are locked rows. The stage uses min-h-0 +
 * overflow-hidden so images never paint over the title or action buttons.
 */
export function Lightbox({ cards, index, onClose, onNavigate, onEdit }: LightboxProps) {
  const card = cards[index];
  const [fullLoaded, setFullLoaded] = useState(false);
  const closeRef = useRef<HTMLButtonElement>(null);

  // Reset the swap state whenever the visible card changes.
  useEffect(() => {
    setFullLoaded(false);
  }, [index]);

  const goTo = useCallback(
    (next: number) => {
      if (next >= 0 && next < cards.length) {
        onNavigate(next);
      }
    },
    [cards.length, onNavigate],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        goTo(index + 1);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        goTo(index - 1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [index, goTo, onClose]);

  // Lock page scroll while open and focus the dialog for keyboard users.
  useEffect(() => {
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus();
    return () => {
      document.body.style.overflow = previous;
    };
  }, []);

  if (!card) {
    return null;
  }

  const displayName =
    card.image_name || card.provenance.source_name || "Image preview";
  const assetTypeLabel = card.asset_type
    ? card.asset_type.charAt(0).toUpperCase() + card.asset_type.slice(1)
    : "";

  return (
    <div
      data-testid="lightbox"
      role="dialog"
      aria-modal="true"
      aria-label={`Preview: ${displayName}`}
      className="fixed inset-0 z-50 flex items-center justify-center bg-navy-950/85 p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="flex h-full max-h-[min(100%,56rem)] w-full max-w-4xl flex-col overflow-hidden rounded-xl bg-white shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="relative z-10 flex shrink-0 items-center justify-between gap-3 border-b border-navy-100 bg-white px-4 py-2.5">
          <p className="min-w-0 flex-1 truncate text-sm font-semibold text-navy-900">
            {displayName}
          </p>
          <div className="flex shrink-0 items-center gap-3">
            <span
              data-testid="lightbox-counter"
              className="text-xs tabular-nums text-navy-500"
            >
              {index + 1} / {cards.length}
            </span>
            <button
              ref={closeRef}
              type="button"
              data-testid="lightbox-close"
              onClick={onClose}
              aria-label="Close preview"
              className="rounded-md p-1 text-navy-500 transition hover:bg-navy-100 hover:text-navy-900"
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
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>
        </div>

        <div className="relative min-h-0 flex-1 overflow-hidden bg-navy-50">
          {/* Thumbnail base layer: on screen instantly from browser cache. */}
          <img
            data-testid="lightbox-thumb"
            src={card.thumb_url || card.image_url}
            alt=""
            aria-hidden
            className={`absolute inset-0 m-auto max-h-full max-w-full object-contain p-2 transition-opacity duration-200 ${
              fullLoaded ? "opacity-0" : "opacity-100 blur-[1px]"
            }`}
          />
          {/* Full-resolution layer fades in over the thumb when ready. */}
          <img
            data-testid="lightbox-full"
            src={card.image_url}
            alt={card.caption || displayName}
            onLoad={() => setFullLoaded(true)}
            className={`absolute inset-0 m-auto max-h-full max-w-full object-contain p-2 transition-opacity duration-200 ${
              fullLoaded ? "opacity-100" : "opacity-0"
            }`}
          />
          {!fullLoaded && (
            <span
              data-testid="lightbox-loading"
              role="status"
              aria-label="Loading full image"
              className="absolute bottom-3 right-3 z-10 flex items-center gap-1.5 rounded-full bg-navy-900/80 px-2.5 py-1 text-[10px] font-medium text-white"
            >
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-white [animation-delay:-200ms]" />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-white [animation-delay:-100ms]" />
              <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-white" />
            </span>
          )}

          {index > 0 && (
            <button
              type="button"
              data-testid="lightbox-prev"
              onClick={() => goTo(index - 1)}
              aria-label="Previous image"
              className="absolute left-2 top-1/2 z-10 -translate-y-1/2 rounded-full bg-white/90 p-2 text-navy-700 shadow ring-1 ring-navy-200 transition hover:bg-white"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden>
                <path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" />
              </svg>
            </button>
          )}
          {index < cards.length - 1 && (
            <button
              type="button"
              data-testid="lightbox-next"
              onClick={() => goTo(index + 1)}
              aria-label="Next image"
              className="absolute right-2 top-1/2 z-10 -translate-y-1/2 rounded-full bg-white/90 p-2 text-navy-700 shadow ring-1 ring-navy-200 transition hover:bg-white"
            >
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} aria-hidden>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
              </svg>
            </button>
          )}
        </div>

        <div className="relative z-10 flex shrink-0 flex-wrap items-center gap-2 border-t border-navy-100 bg-white px-4 py-2.5">
          <span className="rounded bg-brand-500 px-1.5 py-px text-[10px] font-semibold text-white">
            {card.match_percent}%
          </span>
          {assetTypeLabel && (
            <span className="rounded bg-navy-200 px-1.5 py-px text-[10px] font-semibold text-navy-800">
              {assetTypeLabel}
            </span>
          )}
          {card.provenance.chips.slice(0, 2).map((chip) => (
            <span
              key={chip}
              className="rounded bg-navy-100 px-1.5 py-px text-[10px] font-medium text-navy-700"
            >
              {chip}
            </span>
          ))}
          {card.caption && (
            <p className="min-w-0 flex-1 truncate text-xs leading-snug text-navy-700">
              {card.caption}
            </p>
          )}
          {card.has_image_file && (
            <div className="ml-auto flex shrink-0 items-center gap-2">
              {onEdit && (
                <button
                  type="button"
                  data-testid="lightbox-edit"
                  onClick={() => onEdit(card)}
                  className="rounded border border-navy-200 bg-white px-2.5 py-1 text-xs font-medium text-navy-800 transition hover:bg-navy-50"
                >
                  Edit
                </button>
              )}
              <button
                type="button"
                data-testid="lightbox-download"
                onClick={() => void downloadCardImage(card)}
                className="rounded border border-brand-200 bg-brand-50 px-2.5 py-1 text-xs font-medium text-brand-800 transition hover:bg-brand-100"
              >
                Download
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
