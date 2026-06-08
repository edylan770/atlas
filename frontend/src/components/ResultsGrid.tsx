import type { SimilarityAxis } from "../api/client";
import type { ResultCard as ResultCardType } from "../types";
import { ResultCard } from "./ResultCard";

interface ResultsGridProps {
  results: ResultCardType[];
  loading?: boolean;
  onFindSimilar?: (imageId: string, imageName: string) => void;
  searchEventId?: string | null;
  sessionId?: string | null;
  topK?: number;
  minMatchPercent?: number;
  similarityAxis?: SimilarityAxis;
  onSimilarResults?: (results: ResultCardType[], searchEventId?: string | null) => void;
}

export function ResultsGrid({
  results,
  loading = false,
  onFindSimilar,
  searchEventId,
  sessionId,
  topK,
  minMatchPercent,
  similarityAxis,
  onSimilarResults,
}: ResultsGridProps) {
  if (results.length === 0) {
    return (
      <div className="flex min-h-0 flex-1 flex-col items-center justify-center p-8 text-center text-navy-500">
        <svg
          className="mb-3 h-12 w-12 opacity-40"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={1}
            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
          />
        </svg>
        <p className="text-sm">Text or image search results appear here</p>
      </div>
    );
  }

  return (
    <div className="grid min-h-0 flex-1 auto-rows-min grid-cols-2 gap-3 overflow-y-auto p-3">
      {results.map((card) => (
        <ResultCard
          key={card.image_id}
          card={card}
          onFindSimilar={onFindSimilar}
          findSimilarDisabled={loading}
          searchEventId={searchEventId}
          sessionId={sessionId}
          topK={topK}
          minMatchPercent={minMatchPercent}
          similarityAxis={similarityAxis}
          onSimilarResults={onSimilarResults}
        />
      ))}
    </div>
  );
}
