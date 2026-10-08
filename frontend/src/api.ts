import type {
  Conversation,
  ConversationSummary,
  Mode,
  Paper,
  QueryResponse,
  UploadedPaper,
} from "./types";

const BASE = "/api";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

/** FastAPI errors are `{detail: string}` (our own) or `{detail: [{loc, msg}, ...]}` (validation). */
function describe(body: unknown, fallback: string): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d: { loc?: unknown[]; msg?: string }) => `${d.loc?.slice(1).join(".") ?? ""}: ${d.msg}`)
      .join("; ");
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new ApiError("Cannot reach the server. Is the backend running?", 0);
  }
  if (response.status === 204) return undefined as T;
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(describe(body, `Request failed (${response.status})`), response.status);
  }
  return body as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export interface Health {
  status: "ok" | "degraded";
  qdrant: boolean;
  database: boolean;
  papers: number;
  llm_configured: boolean;
}

export const api = {
  health: () => request<Health>("/health"),
  listPapers: () => request<Paper[]>("/papers"),

  uploadPapers: async (files: File[]): Promise<UploadedPaper[]> => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    const res = await request<{ papers: UploadedPaper[] }>("/papers/upload", {
      method: "POST",
      body: form,
    });
    return res.papers;
  },

  deletePaper: (id: string) => request<void>(`/papers/${id}`, { method: "DELETE" }),

  query: (question: string, paperIds: string[], mode: Mode, conversationId: string | null) =>
    request<QueryResponse>(
      "/query",
      json({ question, paper_ids: paperIds, mode, conversation_id: conversationId }),
    ),

  listConversations: () => request<ConversationSummary[]>("/conversations"),
  getConversation: (id: string) => request<Conversation>(`/conversations/${id}`),
  deleteConversation: (id: string) => request<void>(`/conversations/${id}`, { method: "DELETE" }),
};
