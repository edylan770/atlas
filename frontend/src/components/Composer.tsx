import { useRef, type KeyboardEvent } from "react";
import { Link } from "react-router-dom";
import type { SimilarityAxis } from "../api/client";

const IMAGE_ACCEPT = "image/png,image/jpeg,image/webp,image/gif,image/bmp,image/tiff";

function DeckSuggestIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="h-5 w-5"
      aria-hidden
    >
      <rect x="3" y="4" width="18" height="12" rx="1" />
      <path d="M7 8h10M7 12h6" />
      <path d="M8 20h8" />
    </svg>
  );
}

const SIMILARITY_AXES: { id: SimilarityAxis; label: string }[] = [
  { id: "balanced", label: "Balanced" },
  { id: "subject", label: "Subject" },
  { id: "style", label: "Style" },
  { id: "layout", label: "Layout" },
];

interface ComposerProps {
  value: string;
  topK: number;
  minMatchPercent: number;
  similarityAxis: SimilarityAxis;
  loading: boolean;
  onChange: (value: string) => void;
  onTopKChange: (value: number) => void;
  onMinMatchPercentChange: (value: number) => void;
  onSimilarityAxisChange: (value: SimilarityAxis) => void;
  onSend: () => void;
  onSimilarImageSearch: (file: File) => void;
}

function CameraIcon() {
  return (
    <svg
      className="h-5 w-5"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={1.75}
      aria-hidden
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z"
      />
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M15 13a3 3 0 11-6 0 3 3 0 016 0z"
      />
    </svg>
  );
}

export function Composer({
  value,
  topK,
  minMatchPercent,
  similarityAxis,
  loading,
  onChange,
  onTopKChange,
  onMinMatchPercentChange,
  onSimilarityAxisChange,
  onSend,
  onSimilarImageSearch,
}: ComposerProps) {
  const imageInputRef = useRef<HTMLInputElement>(null);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  };

  return (
    <div className="shrink-0 border-t border-navy-200 bg-white p-3">
      <div className="flex gap-2">
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={2}
          disabled={loading}
          placeholder='e.g. "dashboard screenshots from Q3_Review.pptx"'
          className="flex-1 resize-none rounded-xl border border-navy-200 px-3 py-2 text-sm text-navy-900 shadow-inner focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100 disabled:opacity-60"
        />
        <div className="flex shrink-0 items-end gap-2 self-end">
          <button
            type="button"
            onClick={() => imageInputRef.current?.click()}
            disabled={loading}
            title="Search by image"
            aria-label="Search by image"
            className="rounded-xl border border-navy-200 bg-white p-2.5 text-navy-700 shadow-sm transition hover:border-brand-400 hover:bg-brand-50 hover:text-brand-700 disabled:opacity-50"
          >
            <CameraIcon />
          </button>
          <Link
            to="/deck"
            title="Deck suggest"
            aria-label="Open deck suggest"
            className="rounded-xl border border-navy-200 p-2.5 text-navy-600 transition hover:border-brand-400 hover:bg-brand-50 hover:text-brand-600"
          >
            <DeckSuggestIcon />
          </Link>
          <button
            type="button"
            onClick={onSend}
            disabled={loading || !value.trim()}
            className="rounded-xl bg-brand-500 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-brand-400 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading ? "…" : "Send"}
          </button>
        </div>
      </div>
      <input
        ref={imageInputRef}
        type="file"
        accept={IMAGE_ACCEPT}
        className="hidden"
        disabled={loading}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (file?.type.startsWith("image/")) {
            onSimilarImageSearch(file);
          }
          e.target.value = "";
        }}
      />
      <div className="mt-2 flex flex-wrap items-center gap-1">
        <span className="mr-1 text-xs text-navy-600">Similarity axis</span>
        {SIMILARITY_AXES.map(({ id, label }) => (
          <button
            key={id}
            type="button"
            disabled={loading}
            onClick={() => onSimilarityAxisChange(id)}
            className={`rounded-lg px-2.5 py-1 text-xs font-medium transition disabled:opacity-50 ${
              similarityAxis === id
                ? "bg-brand-500 text-white shadow-sm"
                : "border border-navy-200 bg-white text-navy-700 hover:border-brand-400 hover:bg-brand-50"
            }`}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
        <label className="flex items-center gap-2 text-xs text-navy-600">
          <span>Max results</span>
          <input
            type="range"
            min={1}
            max={50}
            value={topK}
            onChange={(e) => onTopKChange(Number(e.target.value))}
            className="accent-brand-600"
          />
          <span className="w-7 font-medium text-navy-800">{topK}</span>
        </label>
        <label className="flex items-center gap-2 text-xs text-navy-600">
          <span>Min match %</span>
          <input
            type="range"
            min={0}
            max={100}
            value={minMatchPercent}
            onChange={(e) => onMinMatchPercentChange(Number(e.target.value))}
            className="accent-brand-600"
          />
          <span className="w-8 font-medium text-navy-800">{minMatchPercent}</span>
        </label>
      </div>
    </div>
  );
}
