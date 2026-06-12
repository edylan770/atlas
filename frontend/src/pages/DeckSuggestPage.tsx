import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { fetchStatus } from "../api/client";
import {
  loadSlideDecisions,
  suggestDeck,
} from "../api/deckClient";
import { AtlasAcronymLine, AtlasWordmark } from "../components/AtlasBranding";
import { DeckSlideCard } from "../components/DeckSlideCard";
import { SortSelect } from "../components/SortSelect";
import type { DeckSuggestResponse, ResultSort, SlideDecision, SlideSuggestion } from "../types";
import { defaultSearchSort } from "../sortResults";

export default function DeckSuggestPage() {
  const [indexedCount, setIndexedCount] = useState(0);
  const [file, setFile] = useState<File | null>(null);
  const [topK, setTopK] = useState(10);
  const [minMatchPercent, setMinMatchPercent] = useState(0);
  const [sortBy, setSortBy] = useState<ResultSort>(defaultSearchSort());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [response, setResponse] = useState<DeckSuggestResponse | null>(null);
  const [decisions, setDecisions] = useState<Record<number, SlideDecision>>({});
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    void fetchStatus()
      .then((s) => setIndexedCount(s.indexed_count))
      .catch(() => setIndexedCount(0));
  }, []);

  const handleFile = (f: File | null) => {
    if (!f) return;
    if (!f.name.toLowerCase().endsWith(".pptx")) {
      setError("Please upload a .pptx file.");
      return;
    }
    setFile(f);
    setError(null);
  };

  const runSuggest = useCallback(async () => {
    if (!file) {
      setError("Choose a PowerPoint file first.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await suggestDeck(file, { topK, minMatchPercent, sort: sortBy });
      setResponse(res);
      setDecisions(loadSlideDecisions(res.deck_hash));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed");
      setResponse(null);
    } finally {
      setLoading(false);
    }
  }, [file, topK, minMatchPercent, sortBy]);

  const updateSlide = (updated: SlideSuggestion) => {
    setResponse((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        slides: prev.slides.map((s) =>
          s.slide_index === updated.slide_index ? updated : s,
        ),
      };
    });
  };

  const setDecision = (slideIndex: number, decision: SlideDecision) => {
    setDecisions((d) => ({ ...d, [slideIndex]: decision }));
  };

  return (
    <div className="flex min-h-screen flex-col bg-navy-50">
      <header className="border-b border-navy-800 bg-navy-900 text-white shadow-md">
        <div className="flex items-center justify-between gap-4 px-5 py-2.5">
          <div className="flex min-w-0 flex-wrap items-baseline gap-x-3 gap-y-1">
            <AtlasWordmark />
            <AtlasAcronymLine />
            <span className="text-sm font-medium text-white/70">Deck suggest</span>
          </div>
          <div className="flex shrink-0 items-center gap-2">
            <span className="rounded-full bg-white/10 px-2.5 py-0.5 text-xs font-medium text-white ring-1 ring-white/20">
              {indexedCount} indexed
            </span>
            <Link
              to="/"
              className="rounded-lg border border-white/25 px-3.5 py-1.5 text-sm font-semibold text-white transition hover:bg-white/10"
            >
              Search
            </Link>
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-5xl flex-1 px-4 py-6">
        <p className="mb-4 text-sm text-navy-600">
          Upload a content-filled PowerPoint deck. Each slide&apos;s text is translated into
          a concrete image search description, then matched against your indexed corpus.
        </p>

        <div className="rounded-xl border border-navy-200 bg-white p-4 shadow-sm">
          <div
            className="flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-navy-200 bg-navy-50/50 px-6 py-10"
            onDragOver={(e) => {
              e.preventDefault();
            }}
            onDrop={(e) => {
              e.preventDefault();
              const f = e.dataTransfer.files[0];
              handleFile(f ?? null);
            }}
          >
            <p className="text-sm font-medium text-navy-800">
              {file ? file.name : "Drop a .pptx file here"}
            </p>
            <input
              ref={inputRef}
              type="file"
              accept=".pptx,application/vnd.openxmlformats-officedocument.presentationml.presentation"
              className="hidden"
              onChange={(e) => handleFile(e.target.files?.[0] ?? null)}
            />
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              className="mt-3 rounded-lg bg-brand-500 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-400"
            >
              Choose file
            </button>
          </div>

          <div className="mt-4 flex flex-wrap items-end gap-4">
            <label className="text-sm text-navy-700">
              Top K
              <input
                type="number"
                min={1}
                max={30}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="ml-2 w-16 rounded border border-navy-200 px-2 py-1"
              />
            </label>
            <label className="text-sm text-navy-700">
              Min match %
              <input
                type="number"
                min={0}
                max={100}
                value={minMatchPercent}
                onChange={(e) => setMinMatchPercent(Number(e.target.value))}
                className="ml-2 w-16 rounded border border-navy-200 px-2 py-1"
              />
            </label>
            <SortSelect value={sortBy} onChange={setSortBy} disabled={loading} />
            <button
              type="button"
              onClick={() => void runSuggest()}
              disabled={loading || !file}
              className="rounded-lg bg-navy-800 px-5 py-2 text-sm font-semibold text-white hover:bg-navy-700 disabled:opacity-50"
            >
              {loading ? "Processing…" : "Suggest images"}
            </button>
          </div>
          {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
        </div>

        {response && (
          <div className="mt-6 space-y-4">
            <div className="flex flex-wrap items-center gap-2 text-sm text-navy-600">
              <span className="font-semibold text-navy-900">{response.filename}</span>
              <span>· {response.slides.length} slides</span>
              {response.deck_cached && (
                <span className="rounded bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-800">
                  Full deck cache hit
                </span>
              )}
              {response.llm_batches > 0 && (
                <span className="text-xs">LLM batches: {response.llm_batches}</span>
              )}
            </div>
            {response.slides.map((slide) => (
              <DeckSlideCard
                key={slide.slide_index}
                slide={slide}
                deckHash={response.deck_hash}
                decision={decisions[slide.slide_index]}
                topK={topK}
                minMatchPercent={minMatchPercent}
                sort={sortBy}
                onDecision={setDecision}
                onSlideUpdate={updateSlide}
              />
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
