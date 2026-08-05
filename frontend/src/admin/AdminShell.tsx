import { useState } from "react";
import { Link, NavLink } from "react-router-dom";
import tistaLogoUrl from "../assets/tista-logo.png";
import { AtlasAcronymLine, AtlasWordmark } from "../components/AtlasBranding";
import {
  clearAdminApiKey,
  getAdminApiKey,
  setAdminApiKey,
  verifyAdminApiKey,
} from "../api/adminClient";
import { CorpusIngestHost } from "./CorpusIngestHost";

const navLinkClass = ({ isActive }: { isActive: boolean }) =>
  isActive
    ? "rounded-md bg-white/10 px-2.5 py-1 text-white"
    : "rounded-md px-2.5 py-1 text-white/70 transition hover:bg-white/5 hover:text-white";

const navLinkEndClass = ({ isActive }: { isActive: boolean }) =>
  isActive
    ? "rounded-md bg-white/10 px-2.5 py-1 text-white"
    : "rounded-md px-2.5 py-1 text-white/70 transition hover:bg-white/5 hover:text-white";

export function AdminKeyGate({ children }: { children: React.ReactNode }) {
  const [key, setKey] = useState(getAdminApiKey() ?? "");
  const [unlocked, setUnlocked] = useState(!!getAdminApiKey());
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);

  async function handleContinue() {
    const trimmed = key.trim();
    if (!trimmed || checking) return;
    setChecking(true);
    setError(null);
    try {
      await verifyAdminApiKey(trimmed);
      setAdminApiKey(trimmed);
      setUnlocked(true);
    } catch (e) {
      clearAdminApiKey();
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setChecking(false);
    }
  }

  if (unlocked) return <>{children}</>;

  return (
    <div className="flex min-h-screen flex-col bg-navy-50">
      <div className="mx-auto flex w-full max-w-md flex-1 flex-col justify-center gap-4 p-8">
        <div className="rounded-xl bg-white p-6 shadow-md ring-1 ring-navy-200">
          <div className="mb-4 flex flex-col gap-1">
            <AtlasWordmark light />
            <h1 className="text-lg font-semibold text-navy-900">Admin Dashboard</h1>
            <p className="text-xs text-navy-500">
              Sign in to manage corpus, analytics, and audit logs.
            </p>
          </div>
          <p className="mb-3 text-sm text-navy-600">
            Enter the server{" "}
            <code className="rounded bg-navy-100 px-1 text-navy-800">ADMIN_API_KEY</code>.
          </p>
          <input
            type="password"
            value={key}
            onChange={(e) => {
              setKey(e.target.value);
              if (error) setError(null);
            }}
            onKeyDown={(e) => {
              if (e.key === "Enter") void handleContinue();
            }}
            disabled={checking}
            className="mb-3 w-full rounded-lg border border-navy-200 px-3 py-2 text-sm text-navy-900 focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-200 disabled:opacity-60"
            placeholder="Admin API key"
            autoComplete="current-password"
          />
          {error ? (
            <p className="mb-3 -mt-1 text-sm text-red-600" role="alert">
              {error}
            </p>
          ) : null}
          <button
            type="button"
            className="w-full rounded-lg bg-brand-500 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-brand-400 disabled:cursor-not-allowed disabled:opacity-60"
            disabled={checking || !key.trim()}
            onClick={() => void handleContinue()}
          >
            {checking ? "Checking…" : "Continue"}
          </button>
          <Link
            to="/"
            className="mt-4 block text-center text-sm font-medium text-brand-600 hover:underline"
          >
            Back to search
          </Link>
        </div>
      </div>
    </div>
  );
}

function AdminHeader({ onOpenCorpus }: { onOpenCorpus: () => void }) {
  return (
    <header className="border-b border-navy-800 bg-navy-900 text-white shadow-md">
      <div className="mx-auto flex max-w-6xl flex-col gap-2.5 px-5 py-2.5 sm:flex-row sm:items-center sm:justify-between sm:gap-4">
        <div className="flex min-w-0 flex-wrap items-baseline gap-x-3 gap-y-1">
          <AtlasWordmark />
          <AtlasAcronymLine className="hidden min-[480px]:block" />
          <span className="w-full text-sm font-semibold text-brand-300 sm:w-auto">
            Admin Dashboard
          </span>
        </div>
        <div className="flex min-w-0 flex-wrap items-center justify-start gap-2 sm:justify-end">
          <nav className="flex flex-wrap items-center gap-1 text-sm font-medium">
            <NavLink to="/admin" end className={navLinkEndClass}>
              Dashboard
            </NavLink>
            <NavLink to="/admin/quality" className={navLinkClass}>
              Search quality
            </NavLink>
            <NavLink to="/admin/corpus" className={navLinkClass}>
              Corpus
            </NavLink>
            <NavLink to="/admin/ingestions" className={navLinkClass}>
              Ingestions
            </NavLink>
            <NavLink to="/admin/pending" className={navLinkClass}>
              Pending additions
            </NavLink>
            <NavLink to="/admin/audit" className={navLinkClass}>
              Audit log
            </NavLink>
            <NavLink
              to="/"
              className="ml-1 rounded-md px-2.5 py-1 text-white/50 transition hover:text-white"
            >
              Back to search
            </NavLink>
          </nav>
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

function AdminFooter() {
  return (
    <footer className="border-t border-navy-800 bg-navy-950 px-5 py-2 text-[11px] text-white/50">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-2">
        <span>
          <span className="font-semibold text-white/80">ATLAS Admin</span>
          {" · "}
          Asset Library management
        </span>
        <Link to="/" className="text-brand-300 hover:text-brand-200 hover:underline">
          Return to search
        </Link>
      </div>
    </footer>
  );
}

export function AdminLayout({ children }: { children: React.ReactNode }) {
  const [corpusOpen, setCorpusOpen] = useState(false);

  return (
    <AdminKeyGate>
      <div className="flex min-h-screen flex-col bg-navy-50">
        <AdminHeader onOpenCorpus={() => setCorpusOpen(true)} />
        <main className="mx-auto w-full max-w-6xl flex-1 p-6">{children}</main>
        <AdminFooter />
        <CorpusIngestHost open={corpusOpen} onOpenChange={setCorpusOpen} />
      </div>
    </AdminKeyGate>
  );
}
