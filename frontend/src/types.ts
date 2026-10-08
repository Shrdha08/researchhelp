// Mirrors the backend's Pydantic schemas (backend/src/researchhelp/api/schemas.py).

export type PaperStatus = "processing" | "ready" | "failed";
export type Mode = "auto" | "evidence" | "research";
export type Intent = "evidence" | "research";

export interface Paper {
  id: string;
  filename: string;
  status: PaperStatus;
  title: string | null;
  num_pages: number | null;
  num_chunks: number | null;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface UploadedPaper {
  paper: Paper;
  duplicate: boolean;
}

export interface Citation {
  source_id: string; // "S1"
  paper_id: string;
  paper_title: string;
  page: number;
  section: string;
  chunk_id: string;
  snippet: string;
}

export interface Source {
  paper_id: string;
  paper_title: string;
  page: number;
  section: string;
  chunk_id: string;
  text: string;
  score: number | null;
}

export interface EvidenceResult {
  answer: string;
  citations: Citation[];
  sources: Source[];
}

export interface ResearchEvidence {
  id: string; // "E1"
  claim: string;
  paper_title: string;
  citations: Citation[];
}

export interface AnalysisItem {
  statement: string;
  based_on: string[];
  confidence: string;
}

export interface DirectionItem {
  title: string;
  rationale: string;
  based_on: string[];
  validation_experiment: string;
}

export interface ResearchResult {
  status: "ok" | "no_evidence" | "unstructured";
  message: string;
  evidence: ResearchEvidence[];
  analysis: AnalysisItem[];
  directions: DirectionItem[];
  sources: Source[];
  raw_text: string;
}

export interface QueryResponse {
  conversation_id: string | null;
  intent: Intent;
  route_source: "llm" | "keyword" | "forced";
  route_reason: string;
  evidence: EvidenceResult | null;
  research: ResearchResult | null;
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  intent: Intent | null;
  mode: Mode | null;
  paper_ids: string[];
  payload: QueryResponse | null;
}

export interface Conversation extends ConversationSummary {
  messages: Message[];
}
