import type { Paper, QueryResponse, Source } from "../types";

export const paper = (over: Partial<Paper> = {}): Paper => ({
  id: "p1",
  filename: "dpr.pdf",
  status: "ready",
  title: "Dense Passage Retrieval",
  num_pages: 13,
  num_chunks: 89,
  error: null,
  created_at: "2026-10-08T10:00:00+00:00",
  updated_at: "2026-10-08T10:00:00+00:00",
  ...over,
});

export const sources: Source[] = [
  {
    paper_id: "p1",
    paper_title: "Dense Passage Retrieval",
    page: 5,
    section: "Experiments",
    chunk_id: "p1:p5:c1",
    text: "We trained with a batch size of 128 and a learning rate of 1e-5.",
    score: 0.9,
  },
  {
    paper_id: "p1",
    paper_title: "Dense Passage Retrieval",
    page: 7,
    section: "Results",
    chunk_id: "p1:p7:c0",
    text: "All experiments were done on eight 32GB GPUs.",
    score: 0.8,
  },
];

const citation = (id: string, page: number) => ({
  source_id: id,
  paper_id: "p1",
  paper_title: "Dense Passage Retrieval",
  page,
  section: "Experiments",
  chunk_id: `p1:p${page}:c1`,
  snippet: "…",
});

export const evidenceResponse = (over: Partial<QueryResponse> = {}): QueryResponse => ({
  conversation_id: "c1",
  intent: "evidence",
  route_source: "llm",
  route_reason: "factual lookup",
  evidence: {
    answer: "Batch size **128** and learning rate 1e-5 [S1].\n\n| Metric | Value |\n|---|---|\n| GPUs | 8 [S2] |",
    citations: [citation("S1", 5), citation("S2", 7)],
    sources,
  },
  research: null,
  ...over,
});

export const researchResponse = (over: Partial<QueryResponse> = {}): QueryResponse => ({
  conversation_id: "c1",
  intent: "research",
  route_source: "forced",
  route_reason: "selected by the user",
  evidence: null,
  research: {
    status: "ok",
    message: "",
    evidence: [
      { id: "E1", claim: "DPR trains with batch size 128.", paper_title: "Dense Passage Retrieval", citations: [citation("S1", 5)] },
      { id: "E2", claim: "Experiments ran on eight GPUs.", paper_title: "Dense Passage Retrieval", citations: [citation("S2", 7)] },
    ],
    analysis: [{ statement: "Training needs substantial hardware.", based_on: ["E2"], confidence: "high" }],
    directions: [
      {
        title: "Lightweight retriever training",
        rationale: "Reduce the hardware needed.",
        based_on: ["E1", "E2"],
        validation_experiment: "Train on one GPU and compare top-20 accuracy.",
      },
    ],
    sources,
    raw_text: "",
  },
  ...over,
});
