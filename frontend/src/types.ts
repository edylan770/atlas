export interface Provenance {
  source_name: string;
  source_type: string;
  slide_index?: number | null;
  page_index?: number | null;
  modified?: string | null;
  author?: string | null;
  chips: string[];
}

export type ResultSort = "relevance" | "newest" | "oldest" | "name" | "source";

export interface ResultCard {
  rank: number;
  image_id: string;
  image_url: string;
  thumb_url?: string;
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
  created_at?: string | null;
  asset_type?: string;
}

export interface CatalogItem {
  image_id: string;
  image_url: string;
  thumb_url?: string;
  image_name: string;
  use_case: string;
  tags: string[];
  recommended_cases: string[];
  caption: string;
  source_name: string;
  source_file?: string;
  created_at?: string | null;
  asset_type?: string;
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
    asset_types: string[];
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
  follow_up_suggestions?: string[];
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
  onDone: (assistantMessage: string, followUpSuggestions: string[]) => void;
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
  followUpSuggestions?: string[];
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
  total_records?: number;
  chroma_vectors?: number;
  text_vector_count?: number;
  bm25_doc_count?: number;
  is_healthy?: boolean;
  stores_in_sync?: boolean;
}

export interface SuggestionsResponse {
  suggestions: string[];
  cached: boolean;
}

export interface IngestResponse {
  message: string;
  indexed_count: number;
  chroma_vectors: number;
  stats: Record<string, number>;
}

export type IngestJobStatus =
  | "staging"
  | "queued"
  | "running"
  | "cancel_requested"
  | "cancelled"
  | "succeeded"
  | "failed";

export interface IngestJob {
  job_id: string;
  status: IngestJobStatus;
  files: string[];
  files_total: number;
  files_done: number;
  images_seen: number;
  images_processed: number;
  options: Record<string, unknown>;
  stats: Record<string, unknown>;
  stage_errors: string[];
  uploads_total?: number;
  upload_bytes_total?: number;
  error?: string | null;
  phase?: string | null;
  status_detail?: string | null;
  runner_id?: string | null;
  created_at?: string | null;
  started_at?: string | null;
  finished_at?: string | null;
  heartbeat_at?: string | null;
  cancel_requested_at?: string | null;
  cancellable: boolean;
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
