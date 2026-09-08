// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ConversationTurn } from "../types";
import { ChatMessageList } from "./ChatMessageList";

const createTurn: ConversationTurn = {
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

beforeEach(() => {
  Element.prototype.scrollIntoView = vi.fn();
});

afterEach(cleanup);

describe("ChatMessageList create turns", () => {
  it("renders the created image and add to database", () => {
    const onAddCreated = vi.fn();
    render(
      <ChatMessageList
        turns={[createTurn]}
        selectedTurnId="turn-1"
        loading={false}
        onSelectTurn={() => {}}
        onFollowUpClick={() => {}}
        onEditResubmit={() => {}}
        onAddCreated={onAddCreated}
        onDownloadCreated={() => {}}
      />,
    );

    const image = screen.getByRole("img", { name: "a blue dashboard" });
    expect(image.getAttribute("src")).toBe(createTurn.createdImageUrl);
    fireEvent.click(screen.getByTestId("create-add-to-database"));
    expect(onAddCreated).toHaveBeenCalledWith(createTurn);
  });
});
