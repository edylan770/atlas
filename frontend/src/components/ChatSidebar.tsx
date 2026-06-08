import { useMemo, useState, type ReactNode } from "react";
import { searchConversations } from "../chat/search";
import type { Conversation } from "../types";
import { AdminNavLink } from "./AdminNavLink";
interface ChatSidebarProps {
  conversations: Conversation[];
  activeId: string | null;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  onSelect: (id: string, turnId?: string | null) => void;
  onNewChat: () => void;
  onDelete: (id: string) => void;
}

function formatRelativeTime(ts: number): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(ts).toLocaleDateString();
}

function highlightSnippet(snippet: string, query: string): ReactNode {
  const q = query.trim();
  if (!q) return snippet;
  const lower = snippet.toLowerCase();
  const idx = lower.indexOf(q.toLowerCase());
  if (idx === -1) return snippet;
  return (
    <>
      {snippet.slice(0, idx)}
      <mark className="rounded bg-brand-200 px-0.5 text-navy-900">
        {snippet.slice(idx, idx + q.length)}
      </mark>
      {snippet.slice(idx + q.length)}
    </>
  );
}

export function ChatSidebar({
  conversations,
  activeId,
  collapsed,
  onToggleCollapsed,
  onSelect,
  onNewChat,
  onDelete,
}: ChatSidebarProps) {
  const [chatSearch, setChatSearch] = useState("");
  const sorted = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt);
  const searchResults = useMemo(
    () => searchConversations(conversations, chatSearch),
    [conversations, chatSearch],
  );
  const isSearching = chatSearch.trim().length > 0;

  if (collapsed) {
    return (
      <div className="flex w-10 shrink-0 flex-col items-center border-r border-navy-200 bg-navy-100 py-2">
        <button
          type="button"
          onClick={onToggleCollapsed}
          title="Show chats"
          className="rounded-lg p-2 text-navy-600 hover:bg-navy-200 hover:text-navy-900"
        >
          »
        </button>
        <button
          type="button"
          onClick={onNewChat}
          title="New chat"
          className="mt-2 rounded-lg p-2 text-brand-600 hover:bg-brand-50"
        >
          +
        </button>
      </div>
    );
  }

  return (
    <aside className="flex w-44 shrink-0 flex-col border-r border-navy-200 bg-navy-50">
      <div className="flex items-center justify-between gap-1 border-b border-navy-200 bg-navy-100/80 px-2 py-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-navy-800">
          Chats
        </span>
        <div className="flex gap-0.5">
          <button
            type="button"
            onClick={onNewChat}
            className="rounded-md px-2 py-1 text-xs font-medium text-brand-600 hover:bg-brand-50"
          >
            New
          </button>
          <button
            type="button"
            onClick={onToggleCollapsed}
            title="Hide chats"
            className="rounded-md px-1.5 py-1 text-xs text-navy-500 hover:bg-navy-200"
          >
            «
          </button>
        </div>
      </div>

      <div className="border-b border-navy-200 px-2 py-2">
        <div className="relative">
          <svg
            className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-navy-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
            aria-hidden
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M21 21l-4.35-4.35M11 18a7 7 0 100-14 7 7 0 000 14z"
            />
          </svg>
          <input
            type="search"
            value={chatSearch}
            onChange={(e) => setChatSearch(e.target.value)}
            placeholder="Search chats…"
            className="w-full rounded-lg border border-navy-200 bg-white py-1.5 pl-8 pr-7 text-xs text-navy-900 shadow-inner placeholder:text-navy-400 focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-200"
          />
          {chatSearch && (
            <button
              type="button"
              onClick={() => setChatSearch("")}
              className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded p-0.5 text-navy-400 hover:text-navy-700"
              aria-label="Clear search"
            >
              ×
            </button>
          )}
        </div>
      </div>

      <ul className="flex-1 overflow-y-auto p-1">
        {isSearching ? (
          searchResults.length === 0 ? (
            <li className="px-2 py-3 text-xs text-navy-500">
              No chats match &ldquo;{chatSearch.trim()}&rdquo;
            </li>
          ) : (
            searchResults.map((hit) => {
              const active = hit.conversationId === activeId;
              return (
                <li key={`${hit.conversationId}-${hit.turnId ?? "t"}`}>
                  <button
                    type="button"
                    onClick={() => onSelect(hit.conversationId, hit.turnId)}
                    className={`w-full rounded-lg px-2 py-2 text-left text-sm transition ${
                      active
                        ? "bg-white font-medium text-navy-900 shadow-sm ring-1 ring-brand-300"
                        : "text-navy-700 hover:bg-white/90"
                    }`}
                  >
                    <span className="line-clamp-1 font-medium leading-snug">
                      {hit.conversationTitle}
                    </span>
                    <span className="mt-0.5 line-clamp-2 text-[11px] leading-snug text-navy-500">
                      {highlightSnippet(hit.snippet, chatSearch)}
                    </span>
                    <span className="mt-0.5 flex items-center gap-1.5 text-[10px] text-navy-400">
                      <span className="rounded bg-navy-100 px-1 py-px text-navy-600">
                        {hit.matchLabel}
                      </span>
                      {formatRelativeTime(hit.updatedAt)}
                    </span>
                  </button>
                </li>
              );
            })
          )
        ) : sorted.length === 0 ? (
          <li className="px-2 py-3 text-xs text-navy-500">No chats yet</li>
        ) : (
          sorted.map((c) => {
            const active = c.id === activeId;
            return (
              <li key={c.id} className="group relative">
                <button
                  type="button"
                  onClick={() => onSelect(c.id)}
                  className={`w-full rounded-lg px-2 py-2 text-left text-sm transition ${
                    active
                      ? "bg-white font-medium text-navy-900 shadow-sm ring-1 ring-brand-300"
                      : "text-navy-700 hover:bg-white/90"
                  }`}
                >
                  <span className="line-clamp-2 leading-snug">{c.title}</span>
                  <span className="mt-0.5 block text-[10px] text-navy-400">
                    {formatRelativeTime(c.updatedAt)}
                  </span>
                </button>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(c.id);
                  }}
                  title="Delete chat"
                  className="absolute right-1 top-1 hidden rounded p-1 text-[10px] text-navy-400 hover:bg-red-50 hover:text-red-600 group-hover:block"
                >
                  ×
                </button>
              </li>
            );
          })
        )}
      </ul>
      <div className="mt-auto border-t border-slate-200 p-2">
        <AdminNavLink variant="sidebar" />
      </div>
    </aside>
  );
}
