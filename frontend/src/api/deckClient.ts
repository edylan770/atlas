import { getUserId } from "./telemetry";
import type { DeckForceResponse, DeckSuggestResponse, ResultSort } from "../types";

function withUserHeaders(init?: RequestInit): RequestInit {
  const headers: Record<string, string> = {
    ...(init?.headers as Record<string, string> | undefined),
  };
  const uid = getUserId();
  if (uid) headers["X-User-Id"] = uid;
  return { ...init, headers };
}

async function deckRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, withUserHeaders(init));
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? body.message ?? detail;
      if (Array.isArray(detail)) {
        detail = detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join("; ");
      }
    } catch {
      /* ignore */
    }
    throw new Error(String(detail));
  }
  return res.json() as Promise<T>;
}

export interface DeckSuggestOptions {
  topK?: number;
  minMatchPercent?: number;
  sort?: ResultSort;
}

export async function suggestDeck(
  file: File,
  options: DeckSuggestOptions = {},
): Promise<DeckSuggestResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("top_k", String(options.topK ?? 10));
  form.append("min_match_percent", String(options.minMatchPercent ?? 0));
  if (options.sort) form.append("sort", options.sort);
  return deckRequest<DeckSuggestResponse>("/api/deck/suggest", {
    method: "POST",
    body: form,
  });
}

export async function forceDeckSlide(
  deckHash: string,
  slideIndex: number,
  options: DeckSuggestOptions = {},
): Promise<DeckForceResponse> {
  return deckRequest<DeckForceResponse>("/api/deck/force", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      deck_hash: deckHash,
      slide_index: slideIndex,
      top_k: options.topK ?? 10,
      min_match_percent: options.minMatchPercent ?? 0,
      sort: options.sort,
    }),
  });
}

const DECISIONS_KEY_PREFIX = "imagecb.deck.decisions.";

export function loadSlideDecisions(
  deckHash: string,
): Record<number, "accepted" | "dismissed"> {
  try {
    const raw = localStorage.getItem(`${DECISIONS_KEY_PREFIX}${deckHash}`);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as Record<string, string>;
    const out: Record<number, "accepted" | "dismissed"> = {};
    for (const [k, v] of Object.entries(parsed)) {
      if (v === "accepted" || v === "dismissed") {
        out[Number(k)] = v;
      }
    }
    return out;
  } catch {
    return {};
  }
}

export function saveSlideDecision(
  deckHash: string,
  slideIndex: number,
  decision: "accepted" | "dismissed",
): void {
  const current = loadSlideDecisions(deckHash);
  current[slideIndex] = decision;
  const serialized: Record<string, string> = {};
  for (const [k, v] of Object.entries(current)) {
    serialized[String(k)] = v;
  }
  localStorage.setItem(
    `${DECISIONS_KEY_PREFIX}${deckHash}`,
    JSON.stringify(serialized),
  );
}
