import tistaLogoUrl from "../assets/tista-logo.png";
import { AtlasAcronymLine, AtlasWordmark } from "./AtlasBranding";

interface HeaderProps {
  indexedCount: number;
  onOpenCorpus: () => void;
}

export function Header({ indexedCount, onOpenCorpus }: HeaderProps) {
  return (
    <header className="border-b border-navy-800 bg-navy-900 text-white shadow-md">
      <div className="flex flex-col gap-2.5 px-5 py-2.5 sm:flex-row sm:items-center sm:justify-between sm:gap-4">
        <div className="flex min-w-0 flex-wrap items-baseline gap-x-3 gap-y-1">
          <AtlasWordmark />
          <AtlasAcronymLine className="hidden min-[480px]:block" />
        </div>
        <div className="flex min-w-0 flex-wrap items-center justify-start gap-2 sm:justify-end">
          <span
            className="rounded-full bg-white/10 px-2.5 py-0.5 text-xs font-medium text-white ring-1 ring-white/20"
            title={`${indexedCount} indexed`}
          >
            <span className="sm:hidden">{indexedCount}</span>
            <span className="hidden sm:inline">{indexedCount} indexed</span>
          </span>
          <button
            type="button"
            onClick={onOpenCorpus}
            className="rounded-lg bg-brand-500 px-3.5 py-1.5 text-sm font-semibold text-white shadow-sm transition hover:bg-brand-400"
          >
            Add to Database
          </button>
          <img
            src={tistaLogoUrl}
            alt="Tista — science and technology corporation"
            className="h-9 w-auto max-w-[160px] rounded bg-white px-2 py-0.5 object-contain object-right"
          />
        </div>
      </div>
    </header>
  );
}
