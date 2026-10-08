import { useState } from "react";
import type { ConversationSummary } from "../types";

interface Props {
  conversations: ConversationSummary[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
}

export function ConversationList({ conversations, activeId, onSelect, onNew, onDelete }: Props) {
  const [confirming, setConfirming] = useState<string | null>(null);
  return (
    <section className="panel" aria-labelledby="conv-heading">
      <div className="panel-head">
        <h2 id="conv-heading">Conversations</h2>
        <button type="button" className="link" onClick={onNew}>
          New
        </button>
      </div>
      {conversations.length === 0 ? (
        <p className="muted empty">Your questions will be saved here.</p>
      ) : (
        <ul className="conv-list">
          {conversations.map((c) => (
            <li key={c.id} className={c.id === activeId ? "active" : ""}>
              <button type="button" className="conv-main" onClick={() => onSelect(c.id)}>
                <span className="conv-title">{c.title}</span>
                <span className="muted small">
                  {c.message_count / 2} question{c.message_count === 2 ? "" : "s"}
                </span>
              </button>
              {confirming === c.id ? (
                <span className="confirm">
                  <button
                    type="button"
                    className="link danger"
                    onClick={() => {
                      setConfirming(null);
                      onDelete(c.id);
                    }}
                  >
                    Confirm
                  </button>
                  <button type="button" className="link" onClick={() => setConfirming(null)}>
                    Cancel
                  </button>
                </span>
              ) : (
                <button
                  type="button"
                  className="link"
                  aria-label={`Delete conversation ${c.title}`}
                  onClick={() => setConfirming(c.id)}
                >
                  Delete
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
