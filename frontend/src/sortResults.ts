import type { CatalogItem, ResultCard, ResultSort } from "./types";

function recordName(item: {
  image_name?: string | null;
  source_name?: string;
  source_file?: string;
}) {
  const name = (item.image_name ?? "").trim();
  if (name) return name.toLowerCase();
  const source = item.source_file ?? item.source_name ?? "";
  const base = source.split(/[/\\]/).pop() ?? source;
  return base.toLowerCase();
}

function recordSource(item: { source_path?: string | null; source_file?: string; provenance?: { source_name?: string } }) {
  const path = item.source_path ?? item.source_file ?? item.provenance?.source_name ?? "";
  return path.toLowerCase();
}

function recordCreated(item: { created_at?: string | null }) {
  return item.created_at ?? "";
}

function withRanks<T extends { rank: number }>(items: T[]): T[] {
  return items.map((item, index) => ({ ...item, rank: index + 1 }));
}

export function sortResultCards(results: ResultCard[], sort: ResultSort): ResultCard[] {
  if (sort === "relevance" || results.length === 0) {
    return results;
  }
  const copy = [...results];
  if (sort === "newest") {
    copy.sort((a, b) => {
      const cmp = recordCreated(b).localeCompare(recordCreated(a));
      return cmp !== 0 ? cmp : a.image_id.localeCompare(b.image_id);
    });
  } else if (sort === "oldest") {
    copy.sort((a, b) => {
      const cmp = recordCreated(a).localeCompare(recordCreated(b));
      return cmp !== 0 ? cmp : a.image_id.localeCompare(b.image_id);
    });
  } else if (sort === "name") {
    copy.sort((a, b) => {
      const cmp = recordName(a).localeCompare(recordName(b));
      return cmp !== 0 ? cmp : a.image_id.localeCompare(b.image_id);
    });
  } else if (sort === "source") {
    copy.sort((a, b) => {
      const cmp = recordSource(a).localeCompare(recordSource(b));
      return cmp !== 0 ? cmp : a.image_id.localeCompare(b.image_id);
    });
  }
  return withRanks(copy);
}

export function sortCatalogItems(items: CatalogItem[], sort: ResultSort): CatalogItem[] {
  if (sort === "relevance") {
    sort = "newest";
  }
  if (items.length === 0) {
    return items;
  }
  const copy = [...items];
  if (sort === "newest") {
    copy.sort((a, b) => {
      const cmp = recordCreated(b).localeCompare(recordCreated(a));
      return cmp !== 0 ? cmp : a.image_id.localeCompare(b.image_id);
    });
  } else if (sort === "oldest") {
    copy.sort((a, b) => {
      const cmp = recordCreated(a).localeCompare(recordCreated(b));
      return cmp !== 0 ? cmp : a.image_id.localeCompare(b.image_id);
    });
  } else if (sort === "name") {
    copy.sort((a, b) => {
      const cmp = recordName(a).localeCompare(recordName(b));
      return cmp !== 0 ? cmp : a.image_id.localeCompare(b.image_id);
    });
  } else if (sort === "source") {
    copy.sort((a, b) => {
      const cmp = recordSource(a).localeCompare(recordSource(b));
      return cmp !== 0 ? cmp : a.image_id.localeCompare(b.image_id);
    });
  }
  return copy;
}

export interface CorpusSortable {
  image_id: string;
  image_name?: string | null;
  source_file?: string;
  created_at?: string | null;
}

export function sortCorpusImages<T extends CorpusSortable>(items: T[], sort: ResultSort): T[] {
  if (sort === "relevance") {
    sort = "newest";
  }
  if (items.length === 0) {
    return items;
  }
  const copy = [...items];
  if (sort === "newest") {
    copy.sort((a, b) => {
      const cmp = recordCreated(b).localeCompare(recordCreated(a));
      return cmp !== 0 ? cmp : a.image_id.localeCompare(b.image_id);
    });
  } else if (sort === "oldest") {
    copy.sort((a, b) => {
      const cmp = recordCreated(a).localeCompare(recordCreated(b));
      return cmp !== 0 ? cmp : a.image_id.localeCompare(b.image_id);
    });
  } else if (sort === "name") {
    copy.sort((a, b) => {
      const cmp = recordName(a).localeCompare(recordName(b));
      return cmp !== 0 ? cmp : a.image_id.localeCompare(b.image_id);
    });
  } else if (sort === "source") {
    copy.sort((a, b) => {
      const cmp = (a.source_file ?? "").toLowerCase().localeCompare((b.source_file ?? "").toLowerCase());
      return cmp !== 0 ? cmp : a.image_id.localeCompare(b.image_id);
    });
  }
  return copy;
}

export function defaultSearchSort(): ResultSort {
  return "relevance";
}

export function defaultCatalogSort(): ResultSort {
  return "newest";
}
