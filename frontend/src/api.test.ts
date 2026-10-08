import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "./api";
import { fakeBackend } from "./test/fakeBackend";

afterEach(() => vi.unstubAllGlobals());

describe("api client", () => {
  it("sends the query as JSON, with the conversation id", async () => {
    const calls = fakeBackend({ "POST /query": () => ({ body: { intent: "evidence" } }) });
    await api.query("What data?", ["p1", "p2"], "research", "c9");
    expect(JSON.parse(calls[0].init!.body as string)).toEqual({
      question: "What data?",
      paper_ids: ["p1", "p2"],
      mode: "research",
      conversation_id: "c9",
    });
  });

  it("uploads every file under the 'files' field", async () => {
    const calls = fakeBackend({ "POST /papers/upload": () => ({ status: 202, body: { papers: [] } }) });
    await api.uploadPapers([new File(["a"], "a.pdf"), new File(["b"], "b.pdf")]);
    const form = calls[0].init!.body as FormData;
    expect(form.getAll("files").map((f) => (f as File).name)).toEqual(["a.pdf", "b.pdf"]);
  });

  it("returns undefined for 204 responses", async () => {
    fakeBackend({ "DELETE /papers/p1": () => ({ status: 204 }) });
    await expect(api.deletePaper("p1")).resolves.toBeUndefined();
  });

  it("surfaces the backend's error message", async () => {
    fakeBackend({ "POST /query": () => ({ status: 409, body: { detail: "paper p1 is processing, not ready" } }) });
    await expect(api.query("q?", ["p1"], "auto", null)).rejects.toMatchObject({
      message: "paper p1 is processing, not ready",
      status: 409,
    });
  });

  it("formats validation errors (422)", async () => {
    fakeBackend({
      "POST /query": () => ({
        status: 422,
        body: { detail: [{ loc: ["body", "paper_ids"], msg: "List should have at least 1 item" }] },
      }),
    });
    await expect(api.query("q?", [], "auto", null)).rejects.toThrow("paper_ids: List should have at least 1 item");
  });

  it("explains an unreachable server", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
    const error = await api.listPapers().catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.message).toMatch(/Cannot reach the server/);
  });
});
