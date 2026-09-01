// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ResultCard } from "../types";
import { ImageEditSession } from "./ImageEditSession";

const apiMocks = vi.hoisted(() => ({
  fetchEditStatus: vi.fn(),
  createEditSession: vi.fn(),
  postEditTurn: vi.fn(),
  submitEditSession: vi.fn(),
}));

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return { ...actual, ...apiMocks };
});

const card = {
  image_id: "source-1",
  image_name: "Test image",
  image_url: "/api/images/source-1",
  thumb_url: "/api/images/source-1/thumb",
  has_image_file: true,
  provenance: { source_name: "source.png" },
} as ResultCard;

const emptySession = {
  session_id: "session-1",
  source_image_id: "source-1",
  original_image_url: "/api/edit/sessions/session-1/original",
  image_url: "/api/edit/sessions/session-1/image",
  turn_count: 0,
  last_prompt: null,
  submitted: false,
  turns: [],
};

beforeEach(() => {
  vi.clearAllMocks();
  Element.prototype.scrollIntoView = vi.fn();
  apiMocks.fetchEditStatus.mockResolvedValue({
    available: true,
    model: "gemini-2.5-flash-image",
    backend: "vertex_express",
    location: "us-central1",
  });
  apiMocks.createEditSession.mockResolvedValue(emptySession);
});

afterEach(cleanup);

async function renderReady() {
  render(<ImageEditSession card={card} onClose={() => {}} />);
  await waitFor(() =>
    expect((screen.getByTestId("edit-prompt") as HTMLTextAreaElement).disabled).toBe(
      false,
    ),
  );
}

describe("ImageEditSession", () => {
  it("renders a generated turn immediately", async () => {
    apiMocks.postEditTurn.mockResolvedValue({
      ...emptySession,
      turn_count: 1,
      last_prompt: "make it blue",
      turns: [
        {
          prompt: "make it blue",
          image_url: "/api/edit/sessions/session-1/turns/0/image",
        },
      ],
    });
    await renderReady();

    fireEvent.change(screen.getByTestId("edit-prompt"), {
      target: { value: "make it blue" },
    });
    fireEvent.click(screen.getByTestId("edit-send"));

    const image = await screen.findByAltText("Edit 1 of Test image");
    expect(image.getAttribute("src")).toContain("/turns/0/image?v=1");
    expect((screen.getByTestId("edit-prompt") as HTMLTextAreaElement).value).toBe("");
    expect(screen.getByTestId("edit-provider-status").textContent).toContain(
      "vertex_express",
    );
  });

  it("keeps the prompt and shows the backend error", async () => {
    apiMocks.postEditTurn.mockRejectedValue(
      new Error("Image edit failed [permission_denied]: access denied"),
    );
    await renderReady();

    fireEvent.change(screen.getByTestId("edit-prompt"), {
      target: { value: "make it blue" },
    });
    fireEvent.click(screen.getByTestId("edit-send"));

    expect(
      (await screen.findByTestId("edit-action-error-banner")).textContent,
    ).toContain("permission_denied");
    expect((screen.getByTestId("edit-prompt") as HTMLTextAreaElement).value).toBe(
      "make it blue",
    );
  });

  it("shows an explicit error when a generated image cannot load", async () => {
    apiMocks.postEditTurn.mockResolvedValue({
      ...emptySession,
      turn_count: 1,
      last_prompt: "make it blue",
      turns: [
        {
          prompt: "make it blue",
          image_url: "/api/edit/sessions/session-1/turns/0/image",
        },
      ],
    });
    await renderReady();

    fireEvent.change(screen.getByTestId("edit-prompt"), {
      target: { value: "make it blue" },
    });
    fireEvent.click(screen.getByTestId("edit-send"));
    fireEvent.error(await screen.findByAltText("Edit 1 of Test image"));

    expect(await screen.findByTestId("edit-turn-image-error-0")).toBeTruthy();
  });
});
