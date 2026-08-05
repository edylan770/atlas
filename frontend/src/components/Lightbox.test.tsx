// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ResultCard } from "../types";
import { Lightbox } from "./Lightbox";

function card(id: string, overrides: Partial<ResultCard> = {}): ResultCard {
  return {
    rank: 1,
    image_id: id,
    image_url: `/api/images/${id}`,
    thumb_url: `/api/images/${id}/thumb`,
    provenance: {
      source_name: `${id}.png`,
      source_type: "image",
      slide_index: null,
      page_index: null,
      modified: null,
      author: null,
      chips: ["source.pptx"],
    },
    caption: `caption for ${id}`,
    match_hint: null,
    match_percent: 97,
    has_image_file: true,
    image_name: `Image ${id}`,
    use_case: "",
    tags: [],
    recommended_cases: [],
    theme: "",
    aliases: [],
    source_url: null,
    source_location: "",
    source_path: null,
    caption_quality: "ok",
    needs_regeneration: false,
    created_at: null,
    asset_type: "chart",
    ...overrides,
  } as ResultCard;
}

const cards = [card("a"), card("b"), card("c")];

afterEach(cleanup);

describe("Lightbox", () => {
  it("shows the thumbnail immediately and swaps to full-res on load", () => {
    render(
      <Lightbox cards={cards} index={0} onClose={() => {}} onNavigate={() => {}} />,
    );
    const thumb = screen.getByTestId("lightbox-thumb");
    const full = screen.getByTestId("lightbox-full");
    expect(thumb.getAttribute("src")).toBe("/api/images/a/thumb");
    expect(full.getAttribute("src")).toBe("/api/images/a");
    // before full loads: thumb visible, loading dots shown
    expect(thumb.className).toContain("opacity-100");
    expect(full.className).toContain("opacity-0");
    expect(screen.getByTestId("lightbox-loading")).toBeTruthy();

    fireEvent.load(full);

    expect(full.className).toContain("opacity-100");
    expect(thumb.className).toContain("opacity-0");
    expect(screen.queryByTestId("lightbox-loading")).toBeNull();
  });

  it("closes on Escape and on backdrop click, not on content click", () => {
    const onClose = vi.fn();
    render(
      <Lightbox cards={cards} index={0} onClose={onClose} onNavigate={() => {}} />,
    );
    fireEvent.click(screen.getByText("Image a")); // content
    expect(onClose).not.toHaveBeenCalled();
    fireEvent.click(screen.getByTestId("lightbox")); // backdrop
    expect(onClose).toHaveBeenCalledTimes(1);
    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it("navigates with arrow keys and clamps at the ends", () => {
    const onNavigate = vi.fn();
    const { rerender } = render(
      <Lightbox cards={cards} index={0} onClose={() => {}} onNavigate={onNavigate} />,
    );
    fireEvent.keyDown(window, { key: "ArrowLeft" }); // clamped at start
    expect(onNavigate).not.toHaveBeenCalled();
    fireEvent.keyDown(window, { key: "ArrowRight" });
    expect(onNavigate).toHaveBeenCalledWith(1);

    rerender(
      <Lightbox cards={cards} index={2} onClose={() => {}} onNavigate={onNavigate} />,
    );
    fireEvent.keyDown(window, { key: "ArrowRight" }); // clamped at end
    expect(onNavigate).toHaveBeenCalledTimes(1);
  });

  it("hides prev at start, next at end, and shows an honest counter", () => {
    const { rerender } = render(
      <Lightbox cards={cards} index={0} onClose={() => {}} onNavigate={() => {}} />,
    );
    expect(screen.queryByTestId("lightbox-prev")).toBeNull();
    expect(screen.getByTestId("lightbox-next")).toBeTruthy();
    expect(screen.getByTestId("lightbox-counter").textContent).toBe("1 / 3");

    rerender(
      <Lightbox cards={cards} index={2} onClose={() => {}} onNavigate={() => {}} />,
    );
    expect(screen.getByTestId("lightbox-prev")).toBeTruthy();
    expect(screen.queryByTestId("lightbox-next")).toBeNull();
    expect(screen.getByTestId("lightbox-counter").textContent).toBe("3 / 3");
  });

  it("resets the thumb-first state when navigating to another image", () => {
    const { rerender } = render(
      <Lightbox cards={cards} index={0} onClose={() => {}} onNavigate={() => {}} />,
    );
    fireEvent.load(screen.getByTestId("lightbox-full"));
    expect(screen.getByTestId("lightbox-full").className).toContain("opacity-100");

    rerender(
      <Lightbox cards={cards} index={1} onClose={() => {}} onNavigate={() => {}} />,
    );
    expect(screen.getByTestId("lightbox-full").className).toContain("opacity-0");
    expect(screen.getByTestId("lightbox-loading")).toBeTruthy();
  });

  it("locks body scroll while open and restores it on unmount", () => {
    const { unmount } = render(
      <Lightbox cards={cards} index={0} onClose={() => {}} onNavigate={() => {}} />,
    );
    expect(document.body.style.overflow).toBe("hidden");
    unmount();
    expect(document.body.style.overflow).toBe("");
  });

  it("omits the download button when the image file is missing", () => {
    const missing = [card("a", { has_image_file: false })];
    render(
      <Lightbox cards={missing} index={0} onClose={() => {}} onNavigate={() => {}} />,
    );
    expect(screen.queryByTestId("lightbox-download")).toBeNull();
  });

  it("calls onEdit when Edit is clicked", () => {
    const onEdit = vi.fn();
    render(
      <Lightbox
        cards={cards}
        index={0}
        onClose={() => {}}
        onNavigate={() => {}}
        onEdit={onEdit}
      />,
    );
    fireEvent.click(screen.getByTestId("lightbox-edit"));
    expect(onEdit).toHaveBeenCalledTimes(1);
    expect(onEdit.mock.calls[0][0].image_id).toBe("a");
  });

  it("hides Edit when onEdit is not provided", () => {
    render(
      <Lightbox cards={cards} index={0} onClose={() => {}} onNavigate={() => {}} />,
    );
    expect(screen.queryByTestId("lightbox-edit")).toBeNull();
  });
});
