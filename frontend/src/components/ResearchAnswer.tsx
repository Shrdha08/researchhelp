import { useState } from "react";
import type { QueryResponse } from "../types";
import { CiteChip } from "./citations";
import { RouteBadge } from "./RouteBadge";
import { SourcesPanel } from "./SourcesPanel";

/** "based on" links: each evidence ID scrolls to and highlights that evidence item. */
function BasedOn({ ids, onJump }: { ids: string[]; onJump: (id: string) => void }) {
  return (
    <span className="based-on">
      <span className="muted small">based on</span>
      {ids.map((id) => (
        <button
          key={id}
          type="button"
          className="chip chip-evidence"
          aria-label={`Show evidence ${id}`}
          onClick={() => onJump(id)}
        >
          {id}
        </button>
      ))}
    </span>
  );
}

export function ResearchAnswer({ uid, response }: { uid: string; response: QueryResponse }) {
  const result = response.research!;
  const [activeEvidence, setActiveEvidence] = useState<string | null>(null);
  const [activeSource, setActiveSource] = useState<string | null>(null);
  const [sourcesOpen, setSourcesOpen] = useState(false);

  const cited = new Set(result.evidence.flatMap((e) => e.citations.map((c) => c.source_id)));

  function jumpToEvidence(id: string) {
    setActiveEvidence(id);
    document.getElementById(`${uid}-ev-${id}`)?.scrollIntoView?.({ block: "center", behavior: "smooth" });
  }
  function showSource(id: string) {
    setActiveSource(id);
    setSourcesOpen(true);
  }

  return (
    <article className="answer answer-research" aria-label="Research answer">
      <RouteBadge response={response} />
      {result.message && <p className="notice">{result.message}</p>}

      {result.evidence.length > 0 && (
        <section className="rs rs-evidence" aria-labelledby={`${uid}-h-ev`}>
          <h3 id={`${uid}-h-ev`}>
            Evidence <span className="rs-sub">what the papers state</span>
          </h3>
          <ol className="rs-list">
            {result.evidence.map((e) => (
              <li
                key={e.id}
                id={`${uid}-ev-${e.id}`}
                className={activeEvidence === e.id ? "active" : ""}
              >
                <span className="chip chip-evidence">{e.id}</span>
                <div>
                  <p>{e.claim}</p>
                  <p className="muted small">
                    {e.paper_title}
                    {e.citations.map((c) => (
                      <span key={c.source_id}>
                        {" · "}
                        <CiteChip id={c.source_id} onCite={showSource} />
                        <span> p.{c.page}</span>
                      </span>
                    ))}
                  </p>
                </div>
              </li>
            ))}
          </ol>
        </section>
      )}

      {result.analysis.length > 0 && (
        <section className="rs rs-analysis" aria-labelledby={`${uid}-h-an`}>
          <h3 id={`${uid}-h-an`}>
            Analysis <span className="rs-sub">inferred from the evidence, not stated by the authors</span>
          </h3>
          <ul className="rs-list">
            {result.analysis.map((a, i) => (
              <li key={i}>
                <div>
                  <p>{a.statement}</p>
                  <p className="meta">
                    <BasedOn ids={a.based_on} onJump={jumpToEvidence} />
                    <span className={`badge badge-conf-${a.confidence}`}>{a.confidence} confidence</span>
                  </p>
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}

      {result.directions.length > 0 && (
        <section className="rs rs-directions" aria-labelledby={`${uid}-h-di`}>
          <h3 id={`${uid}-h-di`}>
            Proposed research directions <span className="rs-sub">hypotheses to test, not established gaps</span>
          </h3>
          <ul className="rs-list">
            {result.directions.map((d, i) => (
              <li key={i}>
                <div>
                  <p className="direction-title">{d.title}</p>
                  <p>{d.rationale}</p>
                  {d.validation_experiment && (
                    <p>
                      <strong>How to test:</strong> {d.validation_experiment}
                    </p>
                  )}
                  <p className="meta">
                    <BasedOn ids={d.based_on} onJump={jumpToEvidence} />
                  </p>
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}

      {result.status === "unstructured" && result.raw_text && (
        <details className="raw">
          <summary>Raw model output</summary>
          <pre>{result.raw_text}</pre>
        </details>
      )}

      <SourcesPanel
        uid={uid}
        sources={result.sources}
        cited={cited}
        active={activeSource}
        open={sourcesOpen}
        onToggle={setSourcesOpen}
      />
    </article>
  );
}
