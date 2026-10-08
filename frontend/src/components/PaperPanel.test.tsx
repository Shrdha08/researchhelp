import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { paper } from "../test/fixtures";
import { PaperPanel } from "./PaperPanel";

function setup(papers = [paper()], selected = new Set<string>()) {
  const handlers = {
    onToggle: vi.fn(),
    onSelectAllReady: vi.fn(),
    onClearSelection: vi.fn(),
    onUpload: vi.fn(),
    onDelete: vi.fn(),
  };
  render(<PaperPanel papers={papers} selected={selected} uploading={false} {...handlers} />);
  return handlers;
}

describe("PaperPanel", () => {
  it("lets only ready papers be selected, and shows status", () => {
    setup([
      paper({ id: "a", title: "Ready paper" }),
      paper({ id: "b", title: "Indexing paper", status: "processing", num_chunks: null }),
      paper({ id: "c", title: "Broken paper", status: "failed", error: "No extractable text" }),
    ]);
    expect(screen.getByRole("checkbox", { name: "Select Ready paper" })).toBeEnabled();
    expect(screen.getByRole("checkbox", { name: "Select Indexing paper" })).toBeDisabled();
    expect(screen.getByRole("checkbox", { name: "Select Broken paper" })).toBeDisabled();
    expect(screen.getByText("No extractable text")).toBeInTheDocument();
    expect(screen.getByText("Processing…")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Delete Indexing paper" })).toBeDisabled();
  });

  it("reports toggles and the selection count", async () => {
    const h = setup([paper({ id: "a", title: "A" }), paper({ id: "b", title: "B" })], new Set(["a"]));
    expect(screen.getByText("1 of 2 selected")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("checkbox", { name: "Select B" }));
    expect(h.onToggle).toHaveBeenCalledWith("b");
    await userEvent.click(screen.getByRole("button", { name: "Select all ready" }));
    expect(h.onSelectAllReady).toHaveBeenCalled();
  });

  it("asks for confirmation before deleting", async () => {
    const h = setup();
    await userEvent.click(screen.getByRole("button", { name: "Delete Dense Passage Retrieval" }));
    expect(h.onDelete).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await userEvent.click(screen.getByRole("button", { name: "Delete Dense Passage Retrieval" }));
    await userEvent.click(screen.getByRole("button", { name: "Confirm" }));
    expect(h.onDelete).toHaveBeenCalledWith("p1");
  });

  it("uploads only the PDFs among the chosen files", async () => {
    const h = setup([]);
    const input = document.getElementById("pdf-input") as HTMLInputElement;
    await userEvent.upload(input, [
      new File(["x"], "paper.pdf", { type: "application/pdf" }),
      new File(["y"], "notes.txt", { type: "text/plain" }),
    ], { applyAccept: false });
    expect(h.onUpload).toHaveBeenCalledTimes(1);
    expect(h.onUpload.mock.calls[0][0].map((f: File) => f.name)).toEqual(["paper.pdf"]);
    expect(screen.getByText(/No papers yet/)).toBeInTheDocument();
  });
});
