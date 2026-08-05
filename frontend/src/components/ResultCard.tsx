import { recordInteraction } from "../api/telemetry";
import { sendSimilar, type SimilarityAxis } from "../api/client";
import { downloadCardImage } from "../imageDownload";
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
  onOpenPreview?: () => void;
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
  onOpenPreview,
}: ResultCardProps) {
  const displayName =
    card.image_name || card.provenance.source_name || "this image";

  const assetTypeLabel = card.asset_type
    ? card.asset_type.charAt(0).toUpperCase() + card.asset_type.slice(1)
    : "";

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
    await downloadCardImage(card);
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

  const hasDetails =
    Boolean(assetTypeLabel) ||
    card.provenance.chips.length > 0 ||
    (card.tags && card.tags.length > 0) ||
    Boolean(card.use_case) ||
    Boolean(card.caption) ||
    (card.recommended_cases && card.recommended_cases.length > 0) ||
    Boolean(card.match_hint);

  return (
    <article
      className="relative flex h-56 flex-col overflow-hidden rounded-lg bg-white shadow-sm ring-1 ring-navy-200 transition hover:shadow-md hover:ring-brand-300 sm:h-60"
      onClick={handleView}
      role="presentation"
    >
      <div
        className={`relative min-h-0 flex-1 bg-navy-50 ${card.has_image_file && onOpenPreview ? "cursor-zoom-in" : ""}`}
        onClick={
          card.has_image_file && onOpenPreview
            ? (e) => {
                e.stopPropagation();
                track("view");
                onOpenPreview();
              }
            : undefined
        }
        role="presentation"
      >
        {card.has_image_file ? (
          <img
            src={card.thumb_url || card.image_url}
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
            title="Download Image"
            aria-label="Download Image"
            className="absolute bottom-1.5 inset-x-1.5 flex items-center justify-center gap-1 rounded-md bg-white/95 px-2 py-1 text-[10px] font-semibold text-navy-800 shadow ring-1 ring-navy-200 transition hover:bg-brand-50 hover:text-brand-700 hover:ring-brand-300"
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
            Download Image
          </button>
        )}
      </div>
      <div className="relative z-10 flex shrink-0 flex-col gap-1 border-t border-navy-100 bg-white p-2">
        {card.image_name && (
          <p className="line-clamp-1 text-xs font-semibold leading-tight text-navy-900">
            {card.image_name}
          </p>
        )}
        {hasDetails && (
          <details
            className="group relative"
            onClick={(e) => e.stopPropagation()}
          >
            <summary className="flex cursor-pointer list-none items-center gap-1 text-[10px] font-medium text-navy-600 marker:content-none [&::-webkit-details-marker]:hidden">
              <span
                className="inline-block text-[9px] text-navy-400 transition group-open:rotate-90"
                aria-hidden
              >
                ▸
              </span>
              <span>Details</span>
            </summary>
            <div className="absolute bottom-full left-0 right-0 z-20 mb-0 max-h-36 overflow-y-auto border-t border-navy-100 bg-white/95 p-2 shadow-md backdrop-blur-sm">
              <div className="flex flex-col gap-1">
                <div className="flex flex-wrap gap-0.5">
                  {assetTypeLabel && (
                    <span className="rounded bg-navy-200 px-1.5 py-px text-[9px] font-semibold text-navy-800">
                      {assetTypeLabel}
                    </span>
                  )}
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
                  <p className="line-clamp-1 text-[9px] italic text-navy-600">
                    {card.use_case}
                  </p>
                )}
                {card.caption && (
                  <p className="line-clamp-2 text-[10px] leading-snug text-navy-800">
                    {card.caption}
                  </p>
                )}
                {card.recommended_cases && card.recommended_cases.length > 0 && (
                  <p
                    className="line-clamp-1 text-[9px] text-navy-500"
                    title={card.recommended_cases.join("\n")}
                  >
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
              </div>
            </div>
          </details>
        )}
        {showPrimarySimilar && (
          <button
            type="button"
            disabled={findSimilarDisabled}
            onClick={() => onFindSimilar!(card.image_id, displayName)}
            className="w-full rounded border border-brand-200 bg-brand-50 py-1 text-[10px] font-medium text-brand-800 transition hover:bg-brand-100 disabled:opacity-50"
          >
            Find similar
          </button>
        )}
        {showInlineSimilar && (
          <div
            className="flex flex-wrap gap-2 border-t border-navy-100 pt-1"
            onClick={(e) => e.stopPropagation()}
          >
            <button
              type="button"
              onClick={() => void handleSimilar()}
              className="text-[9px] font-medium text-brand-600 hover:underline"
            >
              Similar
            </button>
          </div>
        )}
      </div>
    </article>
  );
}
