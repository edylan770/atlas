// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Conversation } from "../types";
import { saveStoredState } from "./storage";

function conversation(id: string, updatedAt: number): Conversation {
  return {
    id,
    title: id,
    sessionId: null,
    createdAt: updatedAt,
    updatedAt,
    turns: [
      {
        id: `${id}-t`,
        userContent: "q",
        assistantContent: "a",
        results: [{ image_id: `${id}-img` } as never],
        parsedQuery: null,
      },
    ],
  };
}

afterEach(() => {
  vi.restoreAllMocks();
  localStorage.clear();
});

describe("saveStoredState", () => {
  it("trims old results instead of silently dropping history on quota errors", () => {
    const realSetItem = Storage.prototype.setItem;
    let calls = 0;
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(function (
      this: Storage,
      key: string,
      value: string,
    ) {
      if (key === "imagecb.conversations.v1" && calls++ === 0) {
        throw new DOMException("full", "QuotaExceededError");
      }
      realSetItem.call(this, key, value);
    });
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    const conversations = Array.from({ length: 12 }, (_, i) =>
      conversation(`c${i}`, i),
    );
    saveStoredState({ conversations, activeConversationId: "c11" });

    const saved = JSON.parse(
      localStorage.getItem("imagecb.conversations.v1") ?? "[]",
    ) as Conversation[];
    expect(saved).toHaveLength(12);
    const withResults = saved.filter((c) => c.turns[0]!.results.length > 0);
    expect(withResults.map((c) => c.id).sort()).toEqual(
      ["c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11"].sort(),
    );
    expect(saved.every((c) => c.turns[0]!.userContent === "q")).toBe(true);
    expect(localStorage.getItem("imagecb.activeConversationId.v1")).toBe("c11");
    expect(warn).toHaveBeenCalled();
  });
});
