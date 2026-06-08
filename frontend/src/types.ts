export interface Provenance {
  source_name: string;
  source_type: string;
  slide_index?: number | null;
  page_index?: number | null;
  modified?: string | null;
  author?: string | null;
  chips: string[];
}

export interface ResultCard {
  rank: number;
  image_id: string;
  image_url: string;
  provenance: Provenance;
  caption: string;
  match_hint?: string | null;
  match_percent: number;
  has_image_file: boolean;
  image_name?: string;
  use_case?: string;
  tags?: string[];
  recommended_cases?: string[];
  source_url?: string | null;
  source_location?: string;
  source_path?: string | null;
}

export interface CatalogItem {
  image_id: string;
  image_url: string;
  image_name: string;
  use_case: string;
  tags: string[];
  recommended_cases: string[];
  caption: string;
  source_name: string;
}

export interface CorpusCatalogResponse {
  items: CatalogItem[];
  indexed_count: number;
  source_url?: string | null;
  source_location?: string;
  source_path?: string | null;
}

export interface ParsedQuery {
  semantic_query: string;
  must_have_keywords: string[];
  must_avoid_keywords: string[];
  source_filters: {
    file_types: string[];
    filename_contains: string[];
    authors: string[];
  };
  time_filter: { after?: string | null; before?: string | null };
  is_refinement: boolean;
  top_k: number;
  interpretation_notes?: string[];
}

export interface ChatResponse {
  session_id: string;
  assistant_message: string;
  results: ResultCard[];
  parsed_query?: ParsedQuery | null;
  search_event_id?: string | null;
}

export interface SimilarResponse {
  session_id: string | null;
  assistant_message: string;
  results: ResultCard[];
  parsed_query?: ParsedQuery | null;
  search_event_id?: string | null;
}

export interface ChatStreamMetadata {
  session_id: string;
  search_event_id?: string | null;
  results: ResultCard[];
  parsed_query?: ParsedQuery | null;
}

export interface ChatStreamCallbacks {
  onMetadata: (data: ChatStreamMetadata) => void;
  onToken: (text: string) => void;
  onDone: (assistantMessage: string) => void;
  onError: (detail: string) => void;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  turnId?: string;
}

export interface ConversationTurn {
  id: string;
  userContent: string;
  assistantContent: string;
  results: ResultCard[];
  parsedQuery: ParsedQuery | null;
  searchEventId?: string | null;
}

export interface Conversation {
  id: string;
  title: string;
  sessionId: string | null;
  createdAt: number;
  updatedAt: number;
  turns: ConversationTurn[];
}

export interface StatusResponse {
  indexed_count: number;
}

export interface SuggestionsResponse {
  suggestions: string[];
  cached: boolean;
}

export interface IngestResponse {
  message: string;
  indexed_count: number;
  stats: Record<string, number>;
}

export interface SlideSuggestion {
  slide_index: number;
  title?: string | null;
  body_preview: string;
  notes_preview: string;
  content_hash: string;
  status: "image_needed" | "no_image_needed";
  description: string;
  reason: string;
  results: ResultCard[];
  llm_cached: boolean;
  search_cached: boolean;
}

export interface DeckSuggestResponse {
  deck_hash: string;
  filename: string;
  slides: SlideSuggestion[];
  deck_cached: boolean;
  llm_batches: number;
}

export interface DeckForceResponse {
  slide: SlideSuggestion;
}

export type SlideDecision = "accepted" | "dismissed";
