// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";

import { downloadCardImage } from "./imageDownload";
import type { ResultCard } from "./types";

const card = {
  image_id: "img-1",
  image_url: "/api/images/img-1",
  image_name: "Quarterly Revenue: Chart!",
  provenance: { source_name: "deck.pptx", chips: [] },
} as unknown as ResultCard;

afterEach(() => vi.restoreAllMocks());

describe("downloadCardImage", () => {
  it("downloads with a sanitized friendly filename and revokes the object URL", async () => {
    const blob = new Blob(["x"], { type: "image/jpeg" });
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        blob: () => Promise.resolve(blob),
        headers: new Headers({ "content-type": "image/jpeg" }),
      }),
    );
    const createUrl = vi.fn().mockReturnValue("blob:fake");
    const revokeUrl = vi.fn();
    vi.stubGlobal("URL", Object.assign(URL, { createObjectURL: createUrl, revokeObjectURL: revokeUrl }));
    const clicked: string[] = [];
    vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(function (this: HTMLAnchorElement) {
      clicked.push(this.download);
    });

    await downloadCardImage(card);

    expect(clicked).toHaveLength(1);
    expect(clicked[0]).toBe("Quarterly Revenue Chart.jpg");
    expect(revokeUrl).toHaveBeenCalledWith("blob:fake");
  });

  it("throws on a failed fetch without touching the DOM", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 500 }));
    await expect(downloadCardImage(card)).rejects.toThrow("500");
  });
});
