import type { ConversationTurn } from "../types";

interface CreatedImagePanelProps {
  turn: ConversationTurn | null;
  loading: boolean;
  adding: boolean;
  onDownload: (turn: ConversationTurn) => void;
  onAdd: (turn: ConversationTurn) => void;
}

export function CreatedImagePanel({
  turn,
  loading,
  adding,
  onDownload,
  onAdd,
}: CreatedImagePanelProps) {
  const canAdd = Boolean(turn?.createSessionId) && !turn?.createSubmitted;

  return (
    <div
      data-testid="created-image-panel"
      className="flex min-h-0 flex-1 flex-col"
    >
      <div className="flex min-h-0 flex-1 items-center justify-center bg-navy-50/70 p-6">
        {turn?.createdImageUrl ? (
          <img
            src={turn.createdImageUrl}
            alt={turn.userContent}
            className="max-h-full max-w-full object-contain"
          />
        ) : (
          <p className="max-w-sm text-center text-sm text-navy-500">
            {loading
              ? "Creating image…"
              : "Your created image will appear here."}
          </p>
        )}
      </div>
      {turn?.createdImageUrl && (
        <footer
          data-testid="created-image-footer"
          className="shrink-0 border-t border-navy-100 bg-white px-4 py-3"
        >
          <p className="line-clamp-2 text-sm text-navy-700" title={turn.userContent}>
            {turn.userContent}
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <button
              type="button"
              data-testid="create-panel-download"
              onClick={() => onDownload(turn)}
              className="rounded-md border border-navy-200 bg-white px-3 py-1.5 text-xs font-semibold text-navy-700 hover:bg-navy-50"
            >
              Download
            </button>
            {canAdd && (
              <button
                type="button"
                data-testid="create-panel-add-to-database"
                disabled={adding || loading}
                onClick={() => onAdd(turn)}
                className="rounded-md border border-brand-200 bg-brand-50 px-3 py-1.5 text-xs font-semibold text-brand-800 hover:bg-brand-100 disabled:opacity-50"
              >
                {adding ? "Submitting…" : "Add to database"}
              </button>
            )}
            {turn.createSubmitted && (
              <span className="text-xs text-emerald-800">
                Submitted for admin review. An admin must accept it before it joins the corpus.
              </span>
            )}
          </div>
        </footer>
      )}
    </div>
  );
}
