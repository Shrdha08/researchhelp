import { useEffect, useRef } from "react";
import type { Source } from "../types";

interface Props {
  uid: string;
  sources: Source[];
  /** Source IDs ("S1") the answer actually cites; the rest were retrieved but unused. */
  cited: Set<string>;
  /** Source to highlight and scroll to (set when the user clicks a citation chip). */
  active: string | null;
  open: boolean;
  onToggle: (open: boolean) => void;
}

/** The retrieved chunks, numbered S1..Sn exactly as the model saw them. */
export function SourcesPanel({ uid, sources, cited, active, open, onToggle }: Props) {
  const activeRef = useRef<HTMLLIElement>(null);

  useEffect(() => {
    if (active && open) activeRef.current?.scrollIntoView?.({ block: "nearest", behavior: "smooth" });
  }, [active, open]);

  if (sources.length === 0) return null;
  return (
    <details className="sources" open={open} onToggle={(e) => onToggle(e.currentTarget.open)}>
      <summary>
        Retrieved evidence ({sources.length} passages, {cited.size} cited)
      </summary>
      <ul className="source-list">
        {sources.map((s, i) => {
          const id = `S${i + 1}`;
          return (
            <li
              key={s.chunk_id}
              id={`${uid}-src-${id}`}
              ref={id === active ? activeRef : undefined}
              className={`source${id === active ? " active" : ""}${cited.has(id) ? " cited" : ""}`}
            >
              <div className="source-head">
                <span className="chip">{id}</span>
                <strong>{s.paper_title}</strong>
                <span className="muted small">
                  page {s.page} · {s.section}
                </span>
                {cited.has(id) && <span className="badge badge-ready">cited</span>}
              </div>
              <p className="source-text">{s.text}</p>
            </li>
          );
        })}
      </ul>
    </details>
  );
}
