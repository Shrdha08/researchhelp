import { Children, Fragment, type ReactNode } from "react";

// The backend normalises every citation marker to "[S1]" or "[S1, S3]" before it reaches us.
const MARKER = /\[(S\d+(?:\s*,\s*S\d+)*)\]/g;

export function CiteChip({ id, onCite }: { id: string; onCite: (id: string) => void }) {
  return (
    <button
      type="button"
      className="chip chip-cite"
      aria-label={`Show source ${id}`}
      title={`Show source ${id}`}
      onClick={() => onCite(id)}
    >
      {id}
    </button>
  );
}

/** Splits text on citation markers, replacing each referenced ID with a clickable chip. */
export function textWithCitations(text: string, onCite: (id: string) => void): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  for (const match of text.matchAll(MARKER)) {
    const start = match.index ?? 0;
    if (start > last) out.push(text.slice(last, start));
    match[1]
      .split(",")
      .map((s) => s.trim())
      .forEach((id, i) => {
        if (i > 0) out.push(" ");
        out.push(<CiteChip key={`${start}-${id}`} id={id} onCite={onCite} />);
      });
    last = start + match[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

/** Applies ``textWithCitations`` to every string among a markdown element's children. */
export function withCitations(children: ReactNode, onCite: (id: string) => void): ReactNode {
  return Children.map(children, (child, i) => {
    if (typeof child === "string") {
      return <Fragment key={i}>{textWithCitations(child, onCite)}</Fragment>;
    }
    return child;
  });
}
