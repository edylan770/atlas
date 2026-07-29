import { useEffect, useState, type KeyboardEvent } from "react";
import ReactMarkdown from "react-markdown";
import type { ConversationTurn } from "../types";
import { SuggestionChips } from "./SuggestionChips";

interface ChatMessageListProps {
  turns: ConversationTurn[];
  selectedTurnId: string | null;
  loading: boolean;
  onSelectTurn: (turnId: string) => void;
  onFollowUpClick: (text: string) => void;
  onEditResubmit: (turnId: string, text: string) => void;
}

function PencilIcon() {
  return (
    <svg
      className="h-3.5 w-3.5"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={1.75}
      aria-hidden
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931zm0 0L19.5 7.125"
      />
    </svg>
  );
}

export function ChatMessageList({
  turns,
  selectedTurnId,
  loading,
  onSelectTurn,
  onFollowUpClick,
  onEditResubmit,
}: ChatMessageListProps) {
  const latestTurnId = turns.length > 0 ? turns[turns.length - 1]!.id : null;
  const [editingTurnId, setEditingTurnId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  useEffect(() => {
    if (!selectedTurnId) return;
    document
      .getElementById(`turn-${selectedTurnId}`)
      ?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [selectedTurnId]);

  useEffect(() => {
    if (loading) {
      setEditingTurnId(null);
      setDraft("");
    }
  }, [loading]);

  const startEdit = (turn: ConversationTurn) => {
    if (loading) return;
    setEditingTurnId(turn.id);
    setDraft(turn.userContent);
  };

  const cancelEdit = () => {
    setEditingTurnId(null);
    setDraft("");
  };

  const saveEdit = (turnId: string) => {
    const text = draft.trim();
    if (!text || loading) return;
    setEditingTurnId(null);
    setDraft("");
    onEditResubmit(turnId, text);
  };

  const handleEditKeyDown = (
    e: KeyboardEvent<HTMLTextAreaElement>,
    turnId: string,
  ) => {
    if (e.key === "Escape") {
      e.preventDefault();
      cancelEdit();
      return;
    }
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      saveEdit(turnId);
    }
  };

  return (
    <div className="flex flex-col gap-4 px-5 py-4">
      {turns.map((turn) => {
        const selected = turn.id === selectedTurnId;
        const isLatest = turn.id === latestTurnId;
        const isEditing = editingTurnId === turn.id;
        const showFollowUps =
          !loading &&
          isLatest &&
          turn.followUpSuggestions &&
          turn.followUpSuggestions.length > 0;

        return (
          <div
            key={turn.id}
            id={`turn-${turn.id}`}
            className="flex flex-col gap-2"
          >
            {isEditing ? (
              <div className="flex max-w-[88%] flex-col gap-2 self-end">
                <textarea
                  value={draft}
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={(e) => handleEditKeyDown(e, turn.id)}
                  rows={3}
                  autoFocus
                  className="w-full resize-y rounded-2xl border border-brand-300 bg-white px-4 py-2.5 text-sm leading-relaxed text-navy-900 shadow-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100"
                />
                <div className="flex justify-end gap-2">
                  <button
                    type="button"
                    onClick={cancelEdit}
                    className="rounded-lg px-3 py-1.5 text-xs font-medium text-navy-600 transition hover:bg-navy-50"
                  >
                    Cancel
                  </button>
                  <button
                    type="button"
                    onClick={() => saveEdit(turn.id)}
                    disabled={!draft.trim()}
                    className="rounded-lg bg-brand-500 px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-brand-400 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    Save
                  </button>
                </div>
              </div>
            ) : (
              <div className="group relative flex max-w-[88%] items-start justify-end gap-1.5 self-end">
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    startEdit(turn);
                  }}
                  disabled={loading}
                  title="Edit query"
                  aria-label="Edit query"
                  className="mt-1 shrink-0 rounded p-1 text-navy-400 opacity-0 transition hover:bg-navy-100 hover:text-navy-700 group-hover:opacity-100 focus-visible:opacity-100 disabled:pointer-events-none disabled:opacity-0"
                >
                  <PencilIcon />
                </button>
                <button
                  type="button"
                  onClick={() => onSelectTurn(turn.id)}
                  className={`rounded-2xl px-4 py-2.5 text-left text-sm leading-relaxed shadow-sm transition ${
                    selected
                      ? "bg-brand-500 text-white ring-2 ring-brand-300"
                      : "bg-brand-500 text-white hover:ring-2 hover:ring-brand-200"
                  }`}
                >
                  {turn.userContent}
                </button>
              </div>
            )}

            <button
              type="button"
              onClick={() => onSelectTurn(turn.id)}
              className={`max-w-[88%] rounded-2xl px-4 py-2.5 text-left text-sm leading-relaxed shadow-sm transition ${
                selected
                  ? "mr-auto bg-white text-navy-800 ring-2 ring-brand-400"
                  : "mr-auto cursor-pointer bg-white text-navy-800 ring-1 ring-navy-200 hover:ring-brand-200"
              }`}
            >
              <div className="prose-chat">
                <ReactMarkdown>{turn.assistantContent}</ReactMarkdown>
              </div>
            </button>

            {showFollowUps && (
              <SuggestionChips
                suggestions={turn.followUpSuggestions!}
                onPick={onFollowUpClick}
                disabled={loading}
                className="mr-auto max-w-[88%] pl-1"
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
