import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { evidenceResponse, researchResponse } from "../test/fixtures";
import { textWithCitations } from "./citations";
import { EvidenceAnswer } from "./EvidenceAnswer";
import { ResearchAnswer } from "./ResearchAnswer";

describe("citation chips", () => {
  it("turns [S1] and [S2, S3] into clickable chips and keeps the surrounding text", async () => {
    const onCite = vi.fn();
    render(<p>{textWithCitations("Batch 128 [S1] on 8 GPUs [S2, S3].", onCite)}</p>);
    expect(screen.getAllByRole("button").map((b) => b.textContent)).toEqual(["S1", "S2", "S3"]);
    expect(screen.getByText(/Batch 128/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Show source S3" }));
    expect(onCite).toHaveBeenCalledWith("S3");
  });
});

describe("EvidenceAnswer", () => {
  it("renders markdown with inline citation chips and a citations list", () => {
    render(<EvidenceAnswer uid="m1" response={evidenceResponse()} />);
    expect(screen.getByText("128", { selector: "strong" })).toBeInTheDocument();
    expect(screen.getByRole("table")).toBeInTheDocument();
    // chips inside the text AND in the citations list
    expect(screen.getAllByRole("button", { name: "Show source S1" }).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/page 5 · Experiments/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/chosen by the router: factual lookup/)).toBeInTheDocument();
  });

  it("opens the retrieved passages and highlights the clicked source", async () => {
    render(<EvidenceAnswer uid="m1" response={evidenceResponse()} />);
    const details = screen.getByText(/Retrieved evidence \(2 passages, 2 cited\)/).closest("details")!;
    expect(details.open).toBe(false);
    await userEvent.click(screen.getAllByRole("button", { name: "Show source S2" })[0]);
    expect(details.open).toBe(true);
    const item = document.getElementById("m1-src-S2")!;
    expect(item).toHaveClass("active");
    expect(within(item).getByText(/eight 32GB GPUs/)).toBeInTheDocument();
  });
});

describe("ResearchAnswer", () => {
  it("shows evidence, analysis and proposed directions as three separate, labelled sections", () => {
    render(<ResearchAnswer uid="m1" response={researchResponse()} />);
    const evidence = screen.getByRole("region", { name: /^Evidence/ });
    const analysis = screen.getByRole("region", { name: /^Analysis/ });
    const directions = screen.getByRole("region", { name: /^Proposed research directions/ });

    expect(within(evidence).getByText("DPR trains with batch size 128.")).toBeInTheDocument();
    expect(within(evidence).getByText(/p\.5/)).toBeInTheDocument(); // page citation
    expect(within(analysis).getByText(/inferred from the evidence, not stated by the authors/)).toBeInTheDocument();
    expect(within(analysis).getByText("high confidence")).toBeInTheDocument();
    expect(within(directions).getByText(/hypotheses to test, not established gaps/)).toBeInTheDocument();
    expect(within(directions).getByText(/Train on one GPU/)).toBeInTheDocument();
    // the same claim text never leaks into the inferred sections
    expect(within(analysis).queryByText("DPR trains with batch size 128.")).not.toBeInTheDocument();
  });

  it("links analysis and directions back to their evidence items", async () => {
    render(<ResearchAnswer uid="m1" response={researchResponse()} />);
    const directions = screen.getByRole("region", { name: /^Proposed research directions/ });
    expect(within(directions).getAllByRole("button").map((b) => b.textContent)).toEqual(["E1", "E2"]);
    await userEvent.click(within(directions).getByRole("button", { name: "Show evidence E1" }));
    expect(document.getElementById("m1-ev-E1")).toHaveClass("active");
    expect(document.getElementById("m1-ev-E2")).not.toHaveClass("active");
  });

  it("opens the cited passage when a page citation is clicked", async () => {
    render(<ResearchAnswer uid="m1" response={researchResponse()} />);
    await userEvent.click(screen.getAllByRole("button", { name: "Show source S2" })[0]);
    expect(document.getElementById("m1-src-S2")).toHaveClass("active");
  });

  it("explains an answer with no evidence instead of showing empty sections", () => {
    const empty = researchResponse();
    empty.research = { ...empty.research!, status: "no_evidence", message: "No relevant evidence was found.", evidence: [], analysis: [], directions: [] };
    render(<ResearchAnswer uid="m1" response={empty} />);
    expect(screen.getByText("No relevant evidence was found.")).toBeInTheDocument();
    expect(screen.queryByRole("region")).not.toBeInTheDocument();
  });

  it("shows raw model output when the answer could not be parsed", () => {
    const raw = researchResponse();
    raw.research = { ...raw.research!, status: "unstructured", message: "Evidence is shown; the analysis output could not be parsed.", analysis: [], directions: [], raw_text: "{broken json" };
    render(<ResearchAnswer uid="m1" response={raw} />);
    expect(screen.getByText("Raw model output")).toBeInTheDocument();
    expect(screen.getByText("{broken json")).toBeInTheDocument();
  });
});
