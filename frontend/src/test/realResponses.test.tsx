/**
 * Contract test: real responses captured from the running backend (real Groq output, real
 * citations, unusual characters such as non-breaking hyphens and superscripts) must render
 * through the UI components. If the backend's schema drifts from src/types.ts, this fails.
 * Regenerate the JSON by running a query against a live server (see frontend/README).
 */
import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { EvidenceAnswer } from "../components/EvidenceAnswer";
import { ResearchAnswer } from "../components/ResearchAnswer";
import conversation from "./real/conversation.json";
import evidence from "./real/evidence_response.json";
import research from "./real/research_response.json";
import type { Conversation, QueryResponse } from "../types";

const evidenceResponse = evidence as unknown as QueryResponse;
const researchResponse = research as unknown as QueryResponse;

describe("real backend responses", () => {
  it("has the shape the TypeScript types describe", () => {
    const conv = conversation as unknown as Conversation;
    expect(conv.messages.map((m) => m.role)).toEqual(["user", "assistant", "user", "assistant"]);
    expect(conv.messages[1].payload?.intent).toBe("evidence");
    expect(conv.messages[3].payload?.intent).toBe("research");
    for (const r of [evidenceResponse, researchResponse]) {
      expect(["llm", "keyword", "forced"]).toContain(r.route_source);
      expect(typeof r.route_reason).toBe("string");
    }
  });

  it("renders a real evidence answer with its citations resolved to retrieved passages", () => {
    render(<EvidenceAnswer uid="real-ev" response={evidenceResponse} />);
    const result = evidenceResponse.evidence!;
    expect(screen.getByRole("article", { name: "Evidence answer" })).toBeInTheDocument();
    expect(result.citations.length).toBeGreaterThan(0);
    for (const c of result.citations) {
      // every cited source id points at a passage that exists in the retrieved list
      const index = Number(c.source_id.slice(1)) - 1;
      expect(result.sources[index]?.chunk_id).toBe(c.chunk_id);
      expect(document.getElementById(`real-ev-src-${c.source_id}`)).not.toBeNull();
    }
    // no raw "[S2]" marker is left unrendered in the answer text
    expect(screen.getByRole("article").querySelector(".markdown")?.textContent).not.toMatch(/\[S\d+\]/);
  });

  it("renders a real research answer as three sections with working evidence links", () => {
    render(<ResearchAnswer uid="real-rs" response={researchResponse} />);
    const result = researchResponse.research!;
    expect(result.status).toBe("ok");
    const evidenceSection = screen.getByRole("region", { name: /^Evidence/ });
    expect(within(evidenceSection).getAllByRole("listitem").length).toBe(result.evidence.length);
    expect(screen.getByRole("region", { name: /^Analysis/ })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: /^Proposed research directions/ })).toBeInTheDocument();

    // every "based on" link resolves to an evidence item that is actually on the page
    const ids = new Set(result.evidence.map((e) => e.id));
    for (const item of [...result.analysis, ...result.directions]) {
      expect(item.based_on.length).toBeGreaterThan(0);
      for (const id of item.based_on) {
        expect(ids.has(id)).toBe(true);
        expect(document.getElementById(`real-rs-ev-${id}`)).not.toBeNull();
      }
    }
    // every evidence item cites a real page
    for (const e of result.evidence) expect(e.citations.every((c) => c.page >= 1)).toBe(true);
  });
});
