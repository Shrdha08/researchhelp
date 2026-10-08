import { useCallback, useMemo, useState, type JSX, type ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import type { QueryResponse } from "../types";
import { withCitations } from "./citations";
import { RouteBadge } from "./RouteBadge";
import { SourcesPanel } from "./SourcesPanel";

type TextTag = "p" | "li" | "td" | "th" | "strong" | "em";

export function EvidenceAnswer({ uid, response }: { uid: string; response: QueryResponse }) {
  const result = response.evidence!;
  const [active, setActive] = useState<string | null>(null);
  const [open, setOpen] = useState(false);
  const cited = new Set(result.citations.map((c) => c.source_id));

  const onCite = useCallback((id: string) => {
    setActive(id);
    setOpen(true);
  }, []);

  // Every text-bearing markdown element routes its strings through the citation-chip splitter.
  // Memoised so the component types keep their identity across renders (otherwise React would
  // remount the whole answer on every state change).
  const components = useMemo<Components>(() => {
    const cite = (Tag: TextTag) =>
      function Cited({ children }: { children?: ReactNode }) {
        const El = Tag as keyof JSX.IntrinsicElements as "p";
        return <El>{withCitations(children, onCite)}</El>;
      };
    return { p: cite("p"), li: cite("li"), td: cite("td"), th: cite("th"), strong: cite("strong"), em: cite("em") };
  }, [onCite]);

  return (
    <article className="answer answer-evidence" aria-label="Evidence answer">
      <RouteBadge response={response} />
      <div className="markdown">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={components}>
          {result.answer}
        </ReactMarkdown>
      </div>

      {result.citations.length > 0 && (
        <div className="cited-list">
          <span className="muted small">Sources</span>
          <ul>
            {result.citations.map((c) => (
              <li key={c.source_id}>
                <button type="button" className="chip chip-cite" onClick={() => onCite(c.source_id)}>
                  {c.source_id}
                </button>
                <span>
                  {c.paper_title}{" "}
                  <span className="muted">
                    · page {c.page} · {c.section}
                  </span>
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <SourcesPanel
        uid={uid}
        sources={result.sources}
        cited={cited}
        active={active}
        open={open}
        onToggle={setOpen}
      />
    </article>
  );
}
