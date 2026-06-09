import { recordInteraction } from "../api/telemetry";
import { sendSimilar, type SimilarityAxis } from "../api/client";
import type { ResultCard as ResultCardType } from "../types";

interface ResultCardProps {
  card: ResultCardType;
  onFindSimilar?: (imageId: string, imageName: string) => void;
  findSimilarDisabled?: boolean;
  searchEventId?: string | null;
  sessionId?: string | null;
  topK?: number;
  minMatchPercent?: number;
  similarityAxis?: SimilarityAxis;
  onSimilarResults?: (results: ResultCardType[], searchEventId?: string | null) => void;
}

export function ResultCard({
  card,
  onFindSimilar,
  findSimilarDisabled = false,
  searchEventId,
  sessionId,
  topK = 10,
  minMatchPercent = 0,
  similarityAxis = "balanced",
  onSimilarResults,
}: ResultCardProps) {
  const displayName =
    card.image_name || card.provenance.source_name || "this image";

  const downloadFileName = () => {
    const base = (card.image_name || card.provenance.source_name || card.image_id)
      .replace(/[^\w.-]+/g, "_")
      .slice(0, 80);
    if (/\.(png|jpe?g|webp|gif|bmp|tiff?)$/i.test(base)) return base;
    return `${base}.png`;
  };

  const track = (type: "view" | "download" | "similar") => {
    if (searchEventId) {
      void recordInteraction(searchEventId, card.image_id, type, card.rank);
    }
  };

  const handleView = () => track("view");

  const handleDownloadImage = async (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!card.has_image_file) return;
    track("download");
    try {
      const res = await fetch(card.image_url);
      if (!res.ok) throw new Error("Download failed");
      const blob = await res.blob();
      const ext =
        blob.type === "image/jpeg"
          ? ".jpg"
          : blob.type === "image/webp"
            ? ".webp"
            : ".png";
      let name = downloadFileName();
      if (!/\.(png|jpe?g|webp|gif|bmp|tiff?)$/i.test(name)) name += ext;
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = name;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch {
      window.open(card.image_url, "_blank", "noopener");
    }
  };

  const handleOpenSource = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (card.source_url) {
      window.open(card.source_url, "_blank", "noopener");
    }
  };

  const handleSimilar = async () => {
    track("similar");
    try {
      const res = await sendSimilar(
        card.image_id,
        sessionId ?? null,
        topK,
        minMatchPercent,
        similarityAxis,
      );
      onSimilarResults?.(res.results, res.search_event_id ?? null);
    } catch {
      /* parent may show error */
    }
  };

  const showPrimarySimilar = Boolean(onFindSimilar && card.has_image_file);
  const showInlineSimilar = Boolean(onSimilarResults && !onFindSimilar);

  return (
    <article
      className="flex flex-col overflow-hidden rounded-lg bg-white shadow-sm ring-1 ring-navy-200 transition hover:shadow-md hover:ring-brand-300"
      onClick={handleView}
      role="presentation"
    >
      <div className="relative h-28 bg-navy-50 sm:h-32">
        {card.has_image_file ? (
          <img
            src={card.image_url}
            alt={card.caption || card.provenance.source_name}
            className="h-full w-full object-contain"
            loading="lazy"
          />
        ) : (
          <div className="flex h-full items-center justify-center text-[10px] text-navy-500">
            Image unavailable
          </div>
        )}
        <span className="absolute left-1.5 top-1.5 rounded bg-navy-900/90 px-1.5 py-px text-[10px] font-medium text-white">
          #{card.rank}
        </span>
        <span
          className="absolute right-1.5 top-1.5 rounded bg-brand-500 px-1.5 py-px text-[10px] font-semibold text-white"
          title="Calibrated relevance (display only); ranking uses raw model scores"
        >
          {card.match_percent}%
        </span>
        {card.has_image_file && (
          <button
            type="button"
            onClick={(e) => void handleDownloadImage(e)}
            title="Download image"
            aria-label="Download image"
            className="absolute bottom-1.5 right-1.5 flex items-center gap-1 rounded-md bg-white/95 px-2 py-1 text-[10px] font-semibold text-navy-800 shadow ring-1 ring-navy-200 transition hover:bg-brand-50 hover:text-brand-700 hover:ring-brand-300"
          >
            <svg
              className="h-3.5 w-3.5"
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
            Download
          </button>
        )}
      </div>
      {card.has_image_file && (
        <button
          type="button"
          onClick={(e) => void handleDownloadImage(e)}
          className="flex w-full items-center justify-center gap-1.5 border-b border-navy-100 bg-brand-50 px-2 py-1.5 text-[11px] font-semibold text-brand-800 transition hover:bg-brand-100"
        >
          <svg
            className="h-3.5 w-3.5"
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
          Download image
        </button>
      )}
      <div className="flex flex-1 flex-col gap-1 p-2">
        {card.image_name && (
          <p className="line-clamp-1 text-xs font-semibold leading-tight text-navy-900">
            {card.image_name}
          </p>
        )}
        <div className="flex flex-wrap gap-0.5">
          {card.provenance.chips.slice(0, 3).map((chip) => (
            <span
              key={chip}
              className="rounded bg-navy-100 px-1.5 py-px text-[9px] font-medium text-navy-700"
            >
              {chip}
            </span>
          ))}
        </div>
        {card.tags && card.tags.length > 0 && (
          <div className="flex flex-wrap gap-0.5">
            {card.tags.slice(0, 4).map((tag) => (
              <span
                key={tag}
                className="rounded bg-brand-50 px-1 py-px text-[9px] text-brand-800"
              >
                {tag}
              </span>
            ))}
          </div>
        )}
        {card.use_case && (
          <p className="line-clamp-1 text-[9px] italic text-navy-600">{card.use_case}</p>
        )}
        {card.caption && (
          <p className="line-clamp-2 text-[10px] leading-snug text-navy-800">
            {card.caption}
          </p>
        )}
        {card.recommended_cases && card.recommended_cases.length > 0 && (
          <p className="line-clamp-1 text-[9px] text-navy-500" title={card.recommended_cases.join("\n")}>
            Try: {card.recommended_cases[0]}
          </p>
        )}
        {card.match_hint && (
          <p
            className="line-clamp-1 text-[9px] text-navy-500"
            title={card.match_hint}
          >
            {card.match_hint}
          </p>
        )}
        {showPrimarySimilar && (
          <button
            type="button"
            disabled={findSimilarDisabled}
            onClick={() => onFindSimilar!(card.image_id, displayName)}
            className="mt-0.5 w-full rounded border border-brand-200 bg-brand-50 py-1 text-[10px] font-medium text-brand-800 transition hover:bg-brand-100 disabled:opacity-50"
          >
            Find similar
          </button>
        )}
        {(card.source_url || showInlineSimilar) && (
          <div
            className="mt-0.5 flex flex-wrap gap-2 border-t border-navy-100 pt-1"
            onClick={(e) => e.stopPropagation()}
          >
            {card.source_url && (
              <button
                type="button"
                onClick={handleOpenSource}
                className="text-[9px] font-medium text-brand-600 hover:underline"
              >
                Open source file
              </button>
            )}
            {showInlineSimilar && (
              <button
                type="button"
                onClick={() => void handleSimilar()}
                className="text-[9px] font-medium text-brand-600 hover:underline"
              >
                Similar
              </button>
            )}
          </div>
        )}
      </div>
    </article>
  );
}
