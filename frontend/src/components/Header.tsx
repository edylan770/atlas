import tistaLogoUrl from "../assets/tista-logo.png";
import { AtlasAcronymLine, AtlasWordmark } from "./AtlasBranding";

interface HeaderProps {
  indexedCount: number;
  /** When set, status fetch failed — do not treat count as authoritative. */
  statusError?: string | null;
}

export function Header({ indexedCount, statusError }: HeaderProps) {
  const statusUnavailable = Boolean(statusError);
  return (
    <header className="border-b border-navy-800 bg-navy-900 text-white shadow-md">
      <div className="flex flex-col gap-2.5 px-5 py-2.5 sm:flex-row sm:items-center sm:justify-between sm:gap-4">
        <div className="flex min-w-0 flex-wrap items-baseline gap-x-3 gap-y-1">
          <AtlasWordmark />
          <AtlasAcronymLine className="hidden min-[480px]:block" />
        </div>
        <div className="flex min-w-0 flex-wrap items-center justify-start gap-2 sm:justify-end">
          <span
            className={
              statusUnavailable
                ? "rounded-full bg-amber-500/20 px-2.5 py-0.5 text-xs font-medium text-amber-100 ring-1 ring-amber-300/40"
                : "rounded-full bg-white/10 px-2.5 py-0.5 text-xs font-medium text-white ring-1 ring-white/20"
            }
            title={
              statusUnavailable
                ? `Index status unavailable: ${statusError}`
                : `${indexedCount} indexed`
            }
          >
            {statusUnavailable ? (
              <>
                <span className="sm:hidden">?</span>
                <span className="hidden sm:inline">Index unavailable</span>
              </>
            ) : (
              <>
                <span className="sm:hidden">{indexedCount}</span>
                <span className="hidden sm:inline">{indexedCount} indexed</span>
              </>
            )}
          </span>
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
