// @vitest-environment jsdom
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ConversationTurn } from "../types";
import { CreatedImagePanel } from "./CreatedImagePanel";

const turn: ConversationTurn = {
  id: "turn-1",
  userContent: "a blue dashboard",
  assistantContent: "Created an image from your prompt.",
  results: [],
  parsedQuery: null,
  kind: "create",
  createdImageUrl: "/api/edit/sessions/session-1/turns/0/image",
  createSessionId: "session-1",
  createSubmitted: false,
};

afterEach(cleanup);

describe("CreatedImagePanel", () => {
  it("keeps the image and actions in one panel without a scroll container", () => {
    render(
      <CreatedImagePanel
        turn={turn}
        loading={false}
        adding={false}
        onDownload={vi.fn()}
        onAdd={vi.fn()}
      />,
    );

    const panel = screen.getByTestId("created-image-panel");
    expect(panel.className).not.toContain("overflow-y-auto");
    expect(screen.getByRole("img", { name: "a blue dashboard" }).getAttribute("src")).toBe(
      turn.createdImageUrl,
    );
    const footer = screen.getByTestId("created-image-footer");
    expect(footer.contains(screen.getByTestId("create-panel-download"))).toBe(true);
    expect(footer.contains(screen.getByTestId("create-panel-add-to-database"))).toBe(
      true,
    );
  });
});
