import type { QueryResponse } from "../types";

const HOW = { llm: "chosen by the router", keyword: "chosen by keyword fallback", forced: "chosen by you" };

export function RouteBadge({ response }: { response: QueryResponse }) {
  const label = response.intent === "evidence" ? "Evidence Q&A" : "Research assistant";
  return (
    <p className="route">
      <span className={`badge badge-${response.intent}`}>{label}</span>
      <span className="muted small">
        {HOW[response.route_source]}
        {response.route_source !== "forced" && response.route_reason ? `: ${response.route_reason}` : ""}
      </span>
    </p>
  );
}
