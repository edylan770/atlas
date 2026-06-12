import type { ResultSort } from "../types";

const SEARCH_OPTIONS: { value: ResultSort; label: string }[] = [
  { value: "relevance", label: "Relevance" },
  { value: "newest", label: "Newest" },
  { value: "oldest", label: "Oldest" },
  { value: "name", label: "Name" },
  { value: "source", label: "Source" },
];

const CATALOG_OPTIONS = SEARCH_OPTIONS.filter((o) => o.value !== "relevance");

interface SortSelectProps {
  value: ResultSort;
  onChange: (value: ResultSort) => void;
  includeRelevance?: boolean;
  disabled?: boolean;
  className?: string;
}

export function SortSelect({
  value,
  onChange,
  includeRelevance = true,
  disabled = false,
  className = "",
}: SortSelectProps) {
  const options = includeRelevance ? SEARCH_OPTIONS : CATALOG_OPTIONS;
  return (
    <label className={`inline-flex items-center gap-1.5 text-xs text-navy-600 ${className}`}>
      <span className="shrink-0 font-medium">Sort</span>
      <select
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.value as ResultSort)}
        className="rounded-md border border-navy-200 bg-white px-2 py-1 text-xs text-navy-800 disabled:opacity-50"
      >
        {options.map((opt) => (
          <option key={opt.value} value={opt.value}>
            {opt.label}
          </option>
        ))}
      </select>
    </label>
  );
}
