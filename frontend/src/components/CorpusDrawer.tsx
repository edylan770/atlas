import { useCallback, useRef, useState, type DragEvent } from "react";
import { SortSelect } from "./SortSelect";
import type { CatalogItem, ResultSort } from "../types";

const ACCEPT =
  ".png,.jpg,.jpeg,.webp,.gif,.bmp,.tif,.tiff,.pdf,.pptx";

interface CorpusDrawerProps {
  open: boolean;
  onClose: () => void;
  skipCaption: boolean;
  skipOcr: boolean;
  force: boolean;
  ingestWorkers: number;
  onSkipCaptionChange: (v: boolean) => void;
  onSkipOcrChange: (v: boolean) => void;
  onForceChange: (v: boolean) => void;
  onIngestWorkersChange: (v: number) => void;
  onIngest: (files: File[]) => void;
  ingestMessage: string | null;
  ingesting: boolean;
  ingestProgress: { filesDone: number; filesTotal: number; batchLabel: string } | null;
  catalog: CatalogItem[];
  catalogLoading: boolean;
  catalogSortBy: ResultSort;
  onCatalogSortChange: (value: ResultSort) => void;
}

export function CorpusDrawer({
  open,
  onClose,
  skipCaption,
  skipOcr,
  force,
  ingestWorkers,
  onSkipCaptionChange,
  onSkipOcrChange,
  onForceChange,
  onIngestWorkersChange,
  onIngest,
  ingestMessage,
  ingesting,
  ingestProgress,
  catalog,
  catalogLoading,
  catalogSortBy,
  onCatalogSortChange,
}: CorpusDrawerProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [selectedCount, setSelectedCount] = useState(0);

  const handleFiles = useCallback(
    (list: FileList | File[] | null) => {
      if (!list?.length) return;
      const files = Array.from(list);
      setSelectedCount(files.length);
      onIngest(files);
    },
    [onIngest],
  );

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (ingesting) return;
    handleFiles(e.dataTransfer.files);
  };

  if (!open) return null;

  return (
    <>
      <button
        type="button"
        className="fixed inset-0 z-40 bg-navy-950/50"
        aria-label="Close drawer"
        onClick={onClose}
      />
      <aside className="fixed right-0 top-0 z-50 flex h-full w-full max-w-lg flex-col bg-white shadow-xl">
        <div className="flex items-center justify-between border-b border-navy-200 bg-navy-50 px-5 py-4">
          <h2 className="text-lg font-semibold text-navy-900">Add to Database</h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg p-1 text-navy-500 hover:bg-navy-100"
          >
            ✕
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-5">
          <p className="mb-4 text-sm text-navy-500">
            Upload many images, PDFs, or decks at once. Each image is captioned in
            parallel (Bedrock) to generate a name, tags, use case, and suggested
            searches.
          </p>

          <div
            onDragOver={(e) => {
              e.preventDefault();
              if (!ingesting) setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            className={`mb-4 rounded-xl border-2 border-dashed px-4 py-8 text-center transition ${
              dragOver
                ? "border-brand-400 bg-brand-50"
                : "border-navy-200 bg-navy-50/50"
            } ${ingesting ? "opacity-60" : ""}`}
          >
            <p className="text-sm font-medium text-navy-700">
              Drag and drop files or folders here
            </p>
            <p className="mt-1 text-xs text-navy-500">
              Images, PDF, PPTX — large batches are uploaded in chunks
            </p>
            <div className="mt-4 flex flex-wrap justify-center gap-2">
              <button
                type="button"
                disabled={ingesting}
                onClick={() => fileInputRef.current?.click()}
                className="rounded-lg bg-brand-500 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-400 disabled:opacity-50"
              >
                Choose files
              </button>
              <button
                type="button"
                disabled={ingesting}
                onClick={() => folderInputRef.current?.click()}
                className="rounded-lg border border-navy-200 bg-white px-4 py-2 text-sm font-medium text-navy-700 hover:bg-navy-50 disabled:opacity-50"
              >
                Choose folder
              </button>
            </div>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept={ACCEPT}
            disabled={ingesting}
            className="hidden"
            onChange={(e) => {
              handleFiles(e.target.files);
              e.target.value = "";
            }}
          />
          <input
            ref={folderInputRef}
            type="file"
            multiple
            // @ts-expect-error webkitdirectory is non-standard but widely supported
            webkitdirectory=""
            disabled={ingesting}
            className="hidden"
            onChange={(e) => {
              handleFiles(e.target.files);
              e.target.value = "";
            }}
          />

          {selectedCount > 0 && !ingesting && (
            <p className="mb-3 text-xs text-navy-500">
              Last selection: {selectedCount} file{selectedCount !== 1 ? "s" : ""}
            </p>
          )}

          <label className="mb-3 flex items-center gap-2 text-xs text-navy-500">
            <span className="shrink-0">Parallel workers</span>
            <input
              type="range"
              min={1}
              max={16}
              value={ingestWorkers}
              onChange={(e) => onIngestWorkersChange(Number(e.target.value))}
              disabled={ingesting}
              className="flex-1 accent-brand-600"
            />
            <span className="w-6 font-medium text-navy-700">{ingestWorkers}</span>
          </label>

          <div className="space-y-2 text-sm">
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={skipCaption}
                onChange={(e) => onSkipCaptionChange(e.target.checked)}
                disabled={ingesting}
              />
              Skip captions (faster, no tags/use cases)
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={skipOcr}
                onChange={(e) => onSkipOcrChange(e.target.checked)}
                disabled={ingesting}
              />
              Skip OCR
            </label>
            <label className="flex items-center gap-2">
              <input
                type="checkbox"
                checked={force}
                onChange={(e) => onForceChange(e.target.checked)}
                disabled={ingesting}
              />
              Force re-ingest duplicates
            </label>
          </div>

          {ingesting && ingestProgress && (
            <div className="mt-4">
              <div className="mb-1 flex justify-between text-xs text-navy-600">
                <span>{ingestProgress.batchLabel}</span>
                <span>
                  {ingestProgress.filesDone} / {ingestProgress.filesTotal} files
                </span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-navy-100">
                <div
                  className="h-full bg-brand-500 transition-all"
                  style={{
                    width: `${
                      ingestProgress.filesTotal > 0
                        ? (100 * ingestProgress.filesDone) / ingestProgress.filesTotal
                        : 0
                    }%`,
                  }}
                />
              </div>
              <p className="mt-2 text-sm text-brand-600">Ingesting…</p>
            </div>
          )}

          {ingestMessage && (
            <pre className="mt-4 max-h-40 overflow-y-auto whitespace-pre-wrap rounded-lg bg-navy-50 p-3 text-xs text-navy-700">
              {ingestMessage}
            </pre>
          )}

          <div className="mt-8 border-t border-navy-100 pt-6">
            <div className="flex items-center justify-between gap-2">
              <h3 className="text-sm font-semibold text-navy-800">Corpus catalog</h3>
              <SortSelect
                value={catalogSortBy}
                onChange={onCatalogSortChange}
                includeRelevance={false}
                disabled={catalogLoading}
              />
            </div>
            <p className="mt-1 text-xs text-navy-500">
              Recently indexed images with generated metadata.
            </p>
            {catalogLoading && (
              <p className="mt-3 text-xs text-navy-400">Loading catalog…</p>
            )}
            {!catalogLoading && catalog.length === 0 && (
              <p className="mt-3 text-xs text-navy-400">No images indexed yet.</p>
            )}
            <ul className="mt-3 space-y-3">
              {catalog.map((item) => (
                <li
                  key={item.image_id}
                  className="rounded-lg border border-navy-100 bg-navy-50/80 p-3"
                >
                  <div className="flex gap-3">
                    <img
                      src={item.image_url}
                      alt={item.image_name}
                      className="h-14 w-14 shrink-0 rounded-md object-cover bg-navy-200"
                      loading="lazy"
                    />
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium text-navy-800">
                        {item.image_name}
                      </p>
                      <div className="flex flex-wrap items-center gap-1">
                        <p className="truncate text-[10px] text-navy-400">
                          {item.source_name}
                        </p>
                        {item.asset_type && (
                          <span className="rounded bg-navy-200 px-1 py-px text-[9px] font-semibold text-navy-800">
                            {item.asset_type.charAt(0).toUpperCase() + item.asset_type.slice(1)}
                          </span>
                        )}
                      </div>
                      {item.use_case && (
                        <p className="mt-1 line-clamp-2 text-xs text-navy-600">
                          {item.use_case}
                        </p>
                      )}
                    </div>
                  </div>
                  {item.tags.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {item.tags.slice(0, 8).map((tag) => (
                        <span
                          key={tag}
                          className="rounded-md bg-white px-1.5 py-0.5 text-[10px] text-navy-600 ring-1 ring-navy-200"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                  {item.recommended_cases.length > 0 && (
                    <ul className="mt-2 space-y-0.5 text-[10px] text-brand-700">
                      {item.recommended_cases.slice(0, 3).map((q) => (
                        <li key={q} className="truncate" title={q}>
                          → {q}
                        </li>
                      ))}
                    </ul>
                  )}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </aside>
    </>
  );
}
