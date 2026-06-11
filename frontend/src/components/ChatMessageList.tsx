import { useEffect } from "react";
import ReactMarkdown from "react-markdown";
import type { ConversationTurn } from "../types";
import { SuggestionChips } from "./SuggestionChips";

interface ChatMessageListProps {
  turns: ConversationTurn[];
  selectedTurnId: string | null;
  loading: boolean;
  onSelectTurn: (turnId: string) => void;
  onFollowUpClick: (text: string) => void;
}

export function ChatMessageList({
  turns,
  selectedTurnId,
  loading,
  onSelectTurn,
  onFollowUpClick,
}: ChatMessageListProps) {
  const latestTurnId = turns.length > 0 ? turns[turns.length - 1]!.id : null;

  useEffect(() => {
    if (!selectedTurnId) return;
    document
      .getElementById(`turn-${selectedTurnId}`)
      ?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [selectedTurnId]);

  return (
    <div className="flex flex-col gap-4 px-5 py-4">
      {turns.map((turn) => {
        const selected = turn.id === selectedTurnId;
        const isLatest = turn.id === latestTurnId;
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
            <button
              type="button"
              onClick={() => onSelectTurn(turn.id)}
              className={`max-w-[88%] self-end rounded-2xl px-4 py-2.5 text-left text-sm leading-relaxed shadow-sm transition ${
                selected
                  ? "bg-brand-500 text-white ring-2 ring-brand-300"
                  : "bg-brand-500 text-white hover:ring-2 hover:ring-brand-200"
              }`}
            >
              {turn.userContent}
            </button>

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
