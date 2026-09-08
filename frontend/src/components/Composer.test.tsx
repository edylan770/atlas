// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { MemoryRouter } from "react-router-dom";

import { Composer } from "./Composer";

function renderComposer(createMode: boolean, onCreateModeChange = vi.fn()) {
  return render(
    <MemoryRouter>
      <Composer
        value=""
        topK={10}
        minMatchPercent={0}
        similarityAxis="balanced"
        loading={false}
        createMode={createMode}
        onChange={() => {}}
        onTopKChange={() => {}}
        onMinMatchPercentChange={() => {}}
        onSimilarityAxisChange={() => {}}
        onCreateModeChange={onCreateModeChange}
        onSend={() => {}}
        onSimilarImageSearch={() => {}}
      />
    </MemoryRouter>,
  );
}

afterEach(cleanup);

describe("Composer create image mode", () => {
  it("switches the prompt from search to image creation", () => {
    const onCreateModeChange = vi.fn();
    const { rerender } = renderComposer(false, onCreateModeChange);

    expect(screen.getByPlaceholderText(/dashboard screenshots/i)).toBeTruthy();
    expect(screen.getByRole("button", { name: "Send" })).toBeTruthy();
    expect(screen.getByLabelText("Image to Image Search")).toBeTruthy();
    expect(screen.getByText("Max results")).toBeTruthy();
    expect(screen.getByText("Advanced")).toBeTruthy();

    fireEvent.click(screen.getByTestId("create-image-mode"));
    expect(onCreateModeChange).toHaveBeenCalledWith(true);

    rerender(
      <MemoryRouter>
        <Composer
          value=""
          topK={10}
          minMatchPercent={0}
          similarityAxis="balanced"
          loading={false}
          createMode
          onChange={() => {}}
          onTopKChange={() => {}}
          onMinMatchPercentChange={() => {}}
          onSimilarityAxisChange={() => {}}
          onCreateModeChange={onCreateModeChange}
          onSend={() => {}}
          onSimilarImageSearch={() => {}}
        />
      </MemoryRouter>,
    );

    expect(screen.getByPlaceholderText("Describe the image to create")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Create" })).toBeTruthy();
    expect(screen.queryByLabelText("Image to Image Search")).toBeNull();
    expect(screen.queryByText("Max results")).toBeNull();
    expect(screen.queryByText("Advanced")).toBeNull();
    expect(
      screen.getByText(/creates an image instead of searching the corpus/i),
    ).toBeTruthy();
  });
});
