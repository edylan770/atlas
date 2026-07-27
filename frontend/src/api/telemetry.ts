const USER_ID_KEY = "imagecb.userId";

export function getUserId(): string | null {
  return localStorage.getItem(USER_ID_KEY);
}


function telemetryHeaders(): HeadersInit {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  const uid = getUserId();
  if (uid) headers["X-User-Id"] = uid;
  return headers;
}

export async function recordInteraction(
  searchEventId: string,
  imageId: string,
  interactionType: "view" | "download" | "similar",
  rank?: number,
): Promise<void> {
  try {
    await fetch("/api/telemetry/interaction", {
      method: "POST",
      headers: telemetryHeaders(),
      body: JSON.stringify({
        search_event_id: searchEventId,
        image_id: imageId,
        interaction_type: interactionType,
        rank: rank ?? null,
      }),
    });
  } catch {
    /* telemetry must not break UX */
  }
}
