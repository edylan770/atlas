interface SuggestionChipsProps {
  suggestions: string[];
  onPick: (text: string) => void;
  disabled?: boolean;
  className?: string;
}

export function SuggestionChips({
  suggestions,
  onPick,
  disabled = false,
  className = "",
}: SuggestionChipsProps) {
  if (suggestions.length === 0) return null;

  return (
    <div className={`flex flex-wrap gap-2 ${className}`}>
      {suggestions.map((text) => (
        <button
          key={text}
          type="button"
          disabled={disabled}
          onClick={() => onPick(text)}
          className="rounded-full border border-navy-200 bg-white px-3 py-1.5 text-xs text-navy-700 shadow-sm transition hover:border-brand-400 hover:bg-brand-50 hover:text-brand-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {text}
        </button>
      ))}
    </div>
  );
}

interface SuggestionChipsSkeletonProps {
  count?: number;
  className?: string;
}

export function SuggestionChipsSkeleton({
  count = 4,
  className = "",
}: SuggestionChipsSkeletonProps) {
  return (
    <div className={`flex flex-wrap gap-2 ${className}`}>
      {Array.from({ length: count }, (_, i) => (
        <span
          key={i}
          className="h-8 w-36 animate-pulse rounded-full bg-navy-100"
          aria-hidden
        />
      ))}
    </div>
  );
}
