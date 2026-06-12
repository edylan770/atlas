import { SuggestionChips, SuggestionChipsSkeleton } from "./SuggestionChips";

interface EmptyStateProps {
  suggestions: string[];
  loading: boolean;
  onPickExample: (text: string) => void;
}

export function EmptyState({
  suggestions,
  loading,
  onPickExample,
}: EmptyStateProps) {
  const hasSuggestions = !loading && suggestions.length > 0;

  return (
    <div className="flex flex-col items-center justify-center px-5 py-8 text-center">
      <div className="mb-3 rounded-2xl bg-navy-900 p-3 text-brand-300 ring-1 ring-navy-700">
        <svg
          className="mx-auto h-8 w-8"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={1.5}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M21 21l-4.35-4.35M11 18a7 7 0 100-14 7 7 0 000 14z"
          />
        </svg>
      </div>
      <h2 className="text-lg font-semibold text-navy-900">
        Search your asset library
      </h2>
      <p className="mt-2 max-w-md text-sm text-navy-600">
        Describe what you are looking for in plain language. Refine across turns
        — results appear on the right.
      </p>
      <div className="mt-6 flex justify-center">
        {loading ? (
          <SuggestionChipsSkeleton />
        ) : hasSuggestions ? (
          <SuggestionChips
            suggestions={suggestions}
            onPick={onPickExample}
            className="justify-center"
          />
        ) : (
          <p className="text-xs text-navy-500">
            Suggestions will appear once your corpus is indexed.
          </p>
        )}
      </div>
    </div>
  );
}
