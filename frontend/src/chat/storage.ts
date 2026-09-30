import type { Conversation, ConversationTurn } from "../types";

const STORAGE_KEY = "imagecb.conversations.v1";
const ACTIVE_KEY = "imagecb.activeConversationId.v1";

export interface StoredState {
  conversations: Conversation[];
  activeConversationId: string | null;
}

function newConversationId(): string {
  return crypto.randomUUID();
}

export function newTurnId(): string {
  return crypto.randomUUID();
}

export function titleFromMessage(text: string, maxLen = 48): string {
  const t = text.trim().replace(/\s+/g, " ");
  if (!t) return "New chat";
  if (t.length <= maxLen) return t;
  return `${t.slice(0, maxLen - 1)}…`;
}

export function createConversation(): Conversation {
  const now = Date.now();
  return {
    id: newConversationId(),
    title: "New chat",
    sessionId: null,
    createdAt: now,
    updatedAt: now,
    turns: [],
  };
}

export function loadStoredState(): StoredState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const active = localStorage.getItem(ACTIVE_KEY);
    if (!raw) {
      return { conversations: [], activeConversationId: active };
    }
    const parsed = JSON.parse(raw) as Conversation[];
    const conversations = Array.isArray(parsed) ? parsed : [];
    return {
      conversations,
      activeConversationId: active,
    };
  } catch {
    return { conversations: [], activeConversationId: null };
  }
}

function isQuotaError(error: unknown): boolean {
  return (
    error instanceof DOMException &&
    (error.name === "QuotaExceededError" ||
      error.name === "NS_ERROR_DOM_QUOTA_REACHED" ||
      error.code === 22)
  );
}

/** Drop result payloads (re-fetchable) from all but the most recent conversations. */
function withoutOldResults(
  conversations: Conversation[],
  keepRecent: number,
): Conversation[] {
  const recent = new Set(
    [...conversations]
      .sort((a, b) => b.updatedAt - a.updatedAt)
      .slice(0, keepRecent)
      .map((c) => c.id),
  );
  return conversations.map((c) =>
    recent.has(c.id)
      ? c
      : { ...c, turns: c.turns.map((t) => ({ ...t, results: [] })) },
  );
}

export function saveStoredState(state: StoredState): void {
  const attempts = [
    () => state.conversations,
    () => withoutOldResults(state.conversations, 10),
    () => withoutOldResults(state.conversations, 1),
    () => withoutOldResults(state.conversations, 0),
  ];
  for (let i = 0; i < attempts.length; i++) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(attempts[i]!()));
      if (i > 0) {
        console.warn("Chat history near storage quota; trimmed saved results.");
      }
      break;
    } catch (error) {
      if (!isQuotaError(error) || i === attempts.length - 1) {
        console.warn("Could not save chat history to localStorage.", error);
        break;
      }
    }
  }
  try {
    if (state.activeConversationId) {
      localStorage.setItem(ACTIVE_KEY, state.activeConversationId);
    } else {
      localStorage.removeItem(ACTIVE_KEY);
    }
  } catch {
    /* private mode */
  }
}

/** Server session ids are in-memory only; stale ids after API restart get a new session on next send. */

export function lastTurn(turns: ConversationTurn[]): ConversationTurn | null {
  return turns.length > 0 ? turns[turns.length - 1]! : null;
}

