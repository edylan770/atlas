import type { ResultCard as ResultCardType } from "./types";

function inferExtension(contentType: string | null): string {
  switch ((contentType || "").split(";")[0].trim()) {
    case "image/jpeg":
      return ".jpg";
    case "image/webp":
      return ".webp";
    case "image/gif":
      return ".gif";
    default:
      return ".png";
  }
}

function safeBaseName(card: ResultCardType): string {
  const raw = card.image_name || card.provenance.source_name || card.image_id;
  return raw.replace(/[^\w\- ]+/g, "").trim().slice(0, 80) || card.image_id;
}

/** Fetch the full-size image and trigger a browser download with a friendly name. */
export async function downloadCardImage(card: ResultCardType): Promise<void> {
  const res = await fetch(card.image_url);
  if (!res.ok) {
    throw new Error(`Image download failed (${res.status})`);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  try {
    const a = document.createElement("a");
    a.href = url;
    a.download = safeBaseName(card) + inferExtension(res.headers.get("content-type"));
    document.body.appendChild(a);
    a.click();
    a.remove();
  } finally {
    URL.revokeObjectURL(url);
  }
}
