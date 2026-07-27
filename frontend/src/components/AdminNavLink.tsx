import { Link } from "react-router-dom";
import { getAdminApiKey } from "../api/adminClient";

type Variant = "header" | "headerDark" | "sidebar" | "sidebarCollapsed" | "footer";

interface AdminNavLinkProps {
  variant: Variant;
}

function LockIcon({ className }: { className?: string }) {
  return (
    <svg
      className={className}
      viewBox="0 0 20 20"
      fill="currentColor"
      aria-hidden
    >
      <path
        fillRule="evenodd"
        d="M10 1a4 4 0 00-4 4v2H5a2 2 0 00-2 2v7a2 2 0 002 2h10a2 2 0 002-2v-7a2 2 0 00-2-2h-1V5a4 4 0 00-4-4zm-2 4V5a2 2 0 114 0v2H8z"
        clipRule="evenodd"
      />
    </svg>
  );
}

export function AdminNavLink({ variant }: AdminNavLinkProps) {
  const hasKey = !!getAdminApiKey();
  // No public admin affordance: admins navigate to /admin directly and unlock
  // there. Once a key is present this session, show the convenience link.
  if (!hasKey) {
    return null;
  }
  const label = "Admin";
  const title = "Open ATLAS admin dashboard";

  if (variant === "footer") {
    return (
      <Link
        to="/admin"
        title={title}
        className={`inline-flex items-center gap-1.5 transition hover:underline ${
          hasKey ? "text-brand-300" : "text-white/60 hover:text-white/80"
        }`}
      >
        <LockIcon className="h-3.5 w-3.5 shrink-0" />
        {label}
      </Link>
    );
  }

  if (variant === "headerDark") {
    return (
      <Link
        to="/admin"
        title={title}
        className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition ${
          hasKey
            ? "text-white/90 ring-1 ring-white/25 hover:bg-white/10"
            : "text-white/80 ring-1 ring-white/20 hover:bg-white/10 hover:text-white"
        }`}
      >
        <LockIcon className="h-4 w-4 shrink-0" />
        {label}
      </Link>
    );
  }

  if (variant === "sidebarCollapsed") {
    return (
      <Link
        to="/admin"
        title={title}
        className="mt-2 flex rounded-lg p-2 text-navy-500 ring-1 ring-transparent transition hover:bg-navy-200 hover:text-brand-600 hover:ring-navy-200"
        aria-label={label}
      >
        <LockIcon className="h-5 w-5" />
      </Link>
    );
  }

  if (variant === "sidebar") {
    return (
      <Link
        to="/admin"
        title={title}
        className="flex w-full items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-navy-600 ring-1 ring-navy-200 transition hover:bg-white hover:text-brand-600 hover:ring-brand-200"
      >
        <LockIcon className="h-4 w-4 shrink-0" />
        {label}
      </Link>
    );
  }

  return (
    <Link
      to="/admin"
      title={title}
      className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-sm font-medium transition ${
        hasKey
          ? "text-navy-700 ring-1 ring-navy-200 hover:bg-navy-50 hover:text-brand-600"
          : "text-navy-600 ring-1 ring-navy-200 hover:bg-brand-50 hover:text-brand-700 hover:ring-brand-200"
      }`}
    >
      <LockIcon className="h-4 w-4 shrink-0" />
      {label}
    </Link>
  );
}
