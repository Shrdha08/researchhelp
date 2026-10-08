import { act, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { evidenceResponse, paper, researchResponse } from "./test/fixtures";
import { fakeBackend, okHealth } from "./test/fakeBackend";
import type { Conversation, ConversationSummary } from "./types";

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

const summary: ConversationSummary = {
  id: "c1",
  title: "What batch size?",
  created_at: "2026-10-08T10:00:00+00:00",
  updated_at: "2026-10-08T10:05:00+00:00",
  message_count: 2,
};

const ask = async (text: string) => {
  await userEvent.type(screen.getByRole("textbox", { name: "Your question" }), text);
  await userEvent.click(screen.getByRole("button", { name: "Ask" }));
};

describe("App", () => {
  it("walks the main journey: select papers, ask, read a cited answer, follow up", async () => {
    const calls = fakeBackend({
      "GET /health": () => ({ body: okHealth }),
      "GET /papers": () => ({
        body: [paper({ id: "p1", title: "Dense Passage Retrieval" }), paper({ id: "p2", title: "FiD" })],
      }),
      "GET /conversations": () => ({ body: [] }),
      "POST /query": () => ({ body: evidenceResponse() }),
    });
    render(<App />);
    await screen.findByText("Dense Passage Retrieval");

    // Nothing selected: the form explains why it cannot be sent.
    expect(screen.getByRole("button", { name: "Ask" })).toBeDisabled();
    expect(screen.getByText("Select at least one paper to ask about.")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("checkbox", { name: "Select Dense Passage Retrieval" }));
    await userEvent.click(screen.getByRole("checkbox", { name: "Select FiD" }));
    await ask("What batch size was used?");

    const answer = await screen.findByRole("article", { name: "Evidence answer" });
    expect(within(answer).getByText("128", { selector: "strong" })).toBeInTheDocument();
    expect(screen.getByText("What batch size was used?")).toBeInTheDocument(); // the user's bubble
    expect(screen.getByRole("textbox", { name: "Your question" })).toHaveValue(""); // cleared on success

    const first = JSON.parse(calls.find((c) => c.key === "POST /query")!.init!.body as string);
    expect(first).toEqual({
      question: "What batch size was used?",
      paper_ids: ["p1", "p2"],
      mode: "auto",
      conversation_id: null,
    });

    // The follow-up continues the same conversation.
    await ask("And the learning rate?");
    await waitFor(() => expect(calls.filter((c) => c.key === "POST /query")).toHaveLength(2));
    const second = JSON.parse(calls.filter((c) => c.key === "POST /query")[1].init!.body as string);
    expect(second.conversation_id).toBe("c1");
  });

  it("sends the chosen mode, and shows a research answer in three sections", async () => {
    const calls = fakeBackend({
      "GET /health": () => ({ body: okHealth }),
      "GET /papers": () => ({ body: [paper()] }),
      "GET /conversations": () => ({ body: [] }),
      "POST /query": () => ({ body: researchResponse() }),
    });
    render(<App />);
    await screen.findByText("Dense Passage Retrieval");
    await userEvent.click(screen.getByRole("checkbox", { name: /Select Dense/ }));
    await userEvent.click(screen.getByRole("radio", { name: "Research" }));
    await ask("What could be improved?");

    await screen.findByRole("article", { name: "Research answer" });
    expect(screen.getByRole("region", { name: /^Evidence/ })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: /^Analysis/ })).toBeInTheDocument();
    expect(screen.getByRole("region", { name: /^Proposed research directions/ })).toBeInTheDocument();
    expect(JSON.parse(calls.find((c) => c.key === "POST /query")!.init!.body as string).mode).toBe("research");
  });

  it("shows the server's error and keeps the question for a retry", async () => {
    fakeBackend({
      "GET /health": () => ({ body: okHealth }),
      "GET /papers": () => ({ body: [paper()] }),
      "GET /conversations": () => ({ body: [] }),
      "POST /query": () => ({ status: 503, body: { detail: "GROQ_API_KEY is not set" } }),
    });
    render(<App />);
    await screen.findByText("Dense Passage Retrieval");
    await userEvent.click(screen.getByRole("checkbox", { name: /Select Dense/ }));
    await ask("Which datasets?");

    expect(await screen.findByText("GROQ_API_KEY is not set")).toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Your question" })).toHaveValue("Which datasets?");
    expect(screen.getByRole("button", { name: "Ask" })).toBeEnabled();
  });

  it("polls while a paper is processing and enables it once ready", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let status: "processing" | "ready" = "processing";
    const calls = fakeBackend({
      "GET /health": () => ({ body: okHealth }),
      "GET /papers": () => ({ body: [paper({ status, num_chunks: status === "ready" ? 34 : null })] }),
      "GET /conversations": () => ({ body: [] }),
    });
    render(<App />);
    expect(await screen.findByText("Indexing…")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /Select Dense/ })).toBeDisabled();

    status = "ready";
    await act(() => vi.advanceTimersByTimeAsync(2100));
    expect(await screen.findByText("13 pages · 34 chunks")).toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: /Select Dense/ })).toBeEnabled();

    const before = calls.filter((c) => c.key === "GET /papers").length;
    await act(() => vi.advanceTimersByTimeAsync(6000)); // polling stops once nothing is processing
    expect(calls.filter((c) => c.key === "GET /papers").length).toBe(before);
  });

  it("reopens a saved conversation, re-rendering its answers and re-selecting its papers", async () => {
    const conversation: Conversation = {
      ...summary,
      message_count: 4,
      messages: [
        { id: 1, role: "user", content: "What batch size?", created_at: "", intent: null, mode: "auto", paper_ids: ["p1"], payload: null },
        { id: 2, role: "assistant", content: "128", created_at: "", intent: "evidence", mode: null, paper_ids: ["p1"], payload: evidenceResponse() },
        { id: 3, role: "user", content: "What could be improved?", created_at: "", intent: null, mode: "research", paper_ids: ["p1", "gone"], payload: null },
        { id: 4, role: "assistant", content: "Research…", created_at: "", intent: "research", mode: null, paper_ids: ["p1"], payload: researchResponse() },
      ],
    };
    fakeBackend({
      "GET /health": () => ({ body: okHealth }),
      "GET /papers": () => ({ body: [paper()] }),
      "GET /conversations": () => ({ body: [summary] }),
      "GET /conversations/c1": () => ({ body: conversation }),
    });
    render(<App />);
    await userEvent.click(await screen.findByRole("button", { name: /^What batch size\?/ }));

    expect(await screen.findByRole("article", { name: "Evidence answer" })).toBeInTheDocument();
    expect(screen.getByRole("article", { name: "Research answer" })).toBeInTheDocument();
    expect(screen.getByText("What could be improved?")).toBeInTheDocument();
    // Selection restored from the last question; the deleted paper ("gone") is skipped.
    expect(screen.getByRole("checkbox", { name: /Select Dense/ })).toBeChecked();
    expect(screen.getByText("1 of 1 selected")).toBeInTheDocument();
  });

  it("deletes a conversation after confirmation and resets the view if it was open", async () => {
    const calls = fakeBackend({
      "GET /health": () => ({ body: okHealth }),
      "GET /papers": () => ({ body: [paper()] }),
      "GET /conversations": () => ({ body: calls.some((c) => c.key === "DELETE /conversations/c1") ? [] : [summary] }),
      "DELETE /conversations/c1": () => ({ status: 204 }),
    });
    render(<App />);
    await screen.findByText("What batch size?");
    await userEvent.click(screen.getByRole("button", { name: /Delete conversation/ }));
    await userEvent.click(screen.getByRole("button", { name: "Confirm" }));
    await screen.findByText(/Your questions will be saved here/);
  });

  it("warns when the backend is degraded, and when an upload is a duplicate", async () => {
    fakeBackend({
      "GET /health": () => ({ body: { ...okHealth, status: "degraded", llm_configured: false } }),
      "GET /papers": () => ({ body: [paper()] }),
      "GET /conversations": () => ({ body: [] }),
      "POST /papers/upload": () => ({ status: 202, body: { papers: [{ paper: paper(), duplicate: true }] } }),
    });
    render(<App />);
    expect(await screen.findByText(/no LLM API key is configured/)).toBeInTheDocument();

    const input = document.getElementById("pdf-input") as HTMLInputElement;
    await userEvent.upload(input, new File(["%PDF"], "dpr.pdf", { type: "application/pdf" }));
    expect(await screen.findByText(/Already uploaded: Dense Passage Retrieval/)).toBeInTheDocument();
  });
});
