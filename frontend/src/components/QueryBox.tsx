import { useState, type FormEvent, type KeyboardEvent } from "react";
import type { Mode } from "../types";

interface Props {
  selectedCount: number;
  busy: boolean;
  onAsk: (question: string, mode: Mode) => Promise<boolean>;
}

const MODES: { value: Mode; label: string; hint: string }[] = [
  { value: "auto", label: "Auto", hint: "The router decides: evidence Q&A or research assistant" },
  { value: "evidence", label: "Evidence", hint: "Answer only from what the papers state, with citations" },
  { value: "research", label: "Research", hint: "Limitations, gaps and proposed directions" },
];

export function QueryBox({ selectedCount, busy, onAsk }: Props) {
  const [text, setText] = useState("");
  const [mode, setMode] = useState<Mode>("auto");
  const question = text.trim();
  const blocker =
    selectedCount === 0
      ? "Select at least one paper to ask about."
      : question.length < 3
        ? "Type a question (at least 3 characters)."
        : null;

  async function submit(e?: FormEvent) {
    e?.preventDefault();
    if (blocker || busy) return;
    const ok = await onAsk(question, mode);
    if (ok) setText(""); // keep the text when the request failed, so it can be retried
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) void submit();
  }

  return (
    <form className="query-box" onSubmit={submit}>
      <div className="modes" role="radiogroup" aria-label="Answer mode">
        {MODES.map((m) => (
          <label key={m.value} className={`mode${mode === m.value ? " on" : ""}`} title={m.hint}>
            <input
              type="radio"
              name="mode"
              value={m.value}
              checked={mode === m.value}
              onChange={() => setMode(m.value)}
            />
            {m.label}
          </label>
        ))}
      </div>
      <textarea
        aria-label="Your question"
        placeholder="Ask about the selected papers, e.g. “Compare the datasets they use” or “What research gaps exist?”"
        value={text}
        rows={3}
        maxLength={2000}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
      />
      <div className="query-foot">
        <span className="muted small">
          {blocker ?? `${selectedCount} paper${selectedCount === 1 ? "" : "s"} selected · Ctrl+Enter to send`}
        </span>
        <button type="submit" className="btn primary" disabled={busy || blocker !== null}>
          {busy ? "Thinking…" : "Ask"}
        </button>
      </div>
    </form>
  );
}
