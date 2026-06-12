import { useState } from "react";
import { forceDeckSlide, saveSlideDecision } from "../api/deckClient";
import type { ResultSort, SlideDecision, SlideSuggestion } from "../types";
import { ResultCard } from "./ResultCard";

interface DeckSlideCardProps {
  slide: SlideSuggestion;
  deckHash: string;
  decision?: SlideDecision;
  topK: number;
  minMatchPercent: number;
  sort?: ResultSort;
  onDecision: (slideIndex: number, decision: SlideDecision) => void;
  onSlideUpdate: (slide: SlideSuggestion) => void;
}

export function DeckSlideCard({
  slide,
  deckHash,
  decision,
  topK,
  minMatchPercent,
  sort,
  onDecision,
  onSlideUpdate,
}: DeckSlideCardProps) {
  const [textOpen, setTextOpen] = useState(false);
  const [forcing, setForcing] = useState(false);
  const [forceError, setForceError] = useState<string | null>(null);

  const handleForce = async () => {
    setForcing(true);
    setForceError(null);
    try {
      const res = await forceDeckSlide(deckHash, slide.slide_index, {
        topK,
        minMatchPercent,
        sort,
      });
      onSlideUpdate(res.slide);
    } catch (e) {
      setForceError(e instanceof Error ? e.message : "Force failed");
    } finally {
      setForcing(false);
    }
  };

  const accept = () => {
    saveSlideDecision(deckHash, slide.slide_index, "accepted");
    onDecision(slide.slide_index, "accepted");
  };

  const dismiss = () => {
    saveSlideDecision(deckHash, slide.slide_index, "dismissed");
    onDecision(slide.slide_index, "dismissed");
  };

  const descriptionLine =
    slide.status === "image_needed"
      ? slide.description
      : slide.reason || "No image needed";

  return (
    <section
      className={`rounded-xl border bg-white shadow-sm ${
        decision === "accepted"
          ? "border-emerald-300 ring-1 ring-emerald-200"
          : decision === "dismissed"
            ? "border-navy-200 opacity-75"
            : "border-navy-200"
      }`}
    >
      <div className="border-b border-navy-100 px-4 py-3">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div className="min-w-0 flex-1">
            <h2 className="text-sm font-semibold text-navy-900">
              Slide {slide.slide_index}
              {slide.title ? (
                <span className="font-normal text-navy-600"> — {slide.title}</span>
              ) : null}
            </h2>
            <div className="mt-1 flex flex-wrap gap-1.5">
              {slide.llm_cached && (
                <span className="rounded bg-navy-100 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-navy-600">
                  LLM cached
                </span>
              )}
              {slide.search_cached && (
                <span className="rounded bg-navy-100 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-navy-600">
                  Search cached
                </span>
              )}
              {slide.status === "no_image_needed" && (
                <span className="rounded bg-amber-100 px-1.5 py-0.5 text-[10px] font-medium text-amber-800">
                  No image needed
                </span>
              )}
            </div>
          </div>
          <div className="flex shrink-0 flex-wrap gap-2">
            {slide.status === "no_image_needed" && (
              <button
                type="button"
                onClick={() => void handleForce()}
                disabled={forcing}
                className="rounded-lg border border-brand-300 bg-brand-50 px-3 py-1 text-xs font-semibold text-brand-700 hover:bg-brand-100 disabled:opacity-50"
              >
                {forcing ? "Forcing…" : "Force image"}
              </button>
            )}
            <button
              type="button"
              onClick={accept}
              className="rounded-lg bg-emerald-600 px-3 py-1 text-xs font-semibold text-white hover:bg-emerald-500"
            >
              Accept
            </button>
            <button
              type="button"
              onClick={dismiss}
              className="rounded-lg border border-navy-300 px-3 py-1 text-xs font-semibold text-navy-700 hover:bg-navy-50"
            >
              Dismiss
            </button>
          </div>
        </div>
        {forceError && (
          <p className="mt-2 text-xs text-red-600">{forceError}</p>
        )}
        <button
          type="button"
          onClick={() => setTextOpen((o) => !o)}
          className="mt-2 text-xs font-medium text-brand-600 hover:text-brand-500"
        >
          {textOpen ? "Hide extracted text" : "Show extracted text"}
        </button>
        {textOpen && (
          <div className="mt-2 space-y-2 text-xs text-navy-600">
            {slide.body_preview && (
              <pre className="whitespace-pre-wrap rounded bg-navy-50 p-2 font-sans">
                {slide.body_preview}
              </pre>
            )}
            {slide.notes_preview && (
              <div>
                <p className="font-semibold text-navy-700">Speaker notes</p>
                <pre className="whitespace-pre-wrap rounded bg-navy-50 p-2 font-sans">
                  {slide.notes_preview}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="border-b border-navy-50 bg-navy-50/50 px-4 py-2">
        <p className="text-[10px] font-semibold uppercase tracking-wide text-navy-500">
          Generated search description
        </p>
        <p className="mt-0.5 text-sm text-navy-800">{descriptionLine || "—"}</p>
      </div>

      {slide.results.length > 0 ? (
        <div className="flex gap-3 overflow-x-auto p-4">
          {slide.results.map((card) => (
            <div key={card.image_id} className="w-64 shrink-0">
              <ResultCard card={card} topK={topK} minMatchPercent={minMatchPercent} />
            </div>
          ))}
        </div>
      ) : (
        <p className="px-4 py-6 text-center text-sm text-navy-500">
          {slide.status === "no_image_needed"
            ? "No image suggestions for this slide."
            : "No matching images in the corpus."}
        </p>
      )}
    </section>
  );
}
