import { useRef, useState, type DragEvent } from "react";
import type { Paper } from "../types";

interface Props {
  papers: Paper[];
  selected: Set<string>;
  uploading: boolean;
  onToggle: (id: string) => void;
  onSelectAllReady: () => void;
  onClearSelection: () => void;
  onUpload: (files: File[]) => void;
  onDelete: (id: string) => void;
}

const STATUS_LABEL = { processing: "Processing…", ready: "Ready", failed: "Failed" } as const;

export function PaperPanel(props: Props) {
  const { papers, selected, uploading } = props;
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [confirming, setConfirming] = useState<string | null>(null);
  const ready = papers.filter((p) => p.status === "ready");

  function pickFiles(list: FileList | null) {
    const files = Array.from(list ?? []).filter((f) => f.name.toLowerCase().endsWith(".pdf"));
    if (files.length) props.onUpload(files);
    if (inputRef.current) inputRef.current.value = "";
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    pickFiles(e.dataTransfer.files);
  }

  return (
    <section className="panel" aria-labelledby="papers-heading">
      <div className="panel-head">
        <h2 id="papers-heading">Papers</h2>
        <span className="muted">
          {selected.size} of {ready.length} selected
        </span>
      </div>

      <div
        className={`dropzone${dragging ? " dragging" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
      >
        <input
          ref={inputRef}
          id="pdf-input"
          type="file"
          accept="application/pdf,.pdf"
          multiple
          hidden
          onChange={(e) => pickFiles(e.target.files)}
        />
        <button
          type="button"
          className="btn"
          disabled={uploading}
          onClick={() => inputRef.current?.click()}
        >
          {uploading ? "Uploading…" : "Upload PDFs"}
        </button>
        <span className="muted">or drop files here</span>
      </div>

      {ready.length > 1 && (
        <div className="row-actions">
          <button type="button" className="link" onClick={props.onSelectAllReady}>
            Select all ready
          </button>
          <button type="button" className="link" onClick={props.onClearSelection}>
            Clear
          </button>
        </div>
      )}

      {papers.length === 0 ? (
        <p className="muted empty">No papers yet. Upload one or more PDFs to begin.</p>
      ) : (
        <ul className="paper-list">
          {papers.map((p) => {
            const label = p.title ?? p.filename;
            return (
              <li key={p.id} className={`paper status-${p.status}`}>
                <label className="paper-main">
                  <input
                    type="checkbox"
                    checked={selected.has(p.id)}
                    disabled={p.status !== "ready"}
                    onChange={() => props.onToggle(p.id)}
                    aria-label={`Select ${label}`}
                  />
                  <span className="paper-text">
                    <span className="paper-title">{label}</span>
                    <span className="muted small">
                      {p.status === "ready"
                        ? `${p.num_pages ?? "?"} pages · ${p.num_chunks ?? "?"} chunks`
                        : p.status === "failed"
                          ? (p.error ?? "Indexing failed")
                          : "Indexing…"}
                    </span>
                  </span>
                </label>
                <span className={`badge badge-${p.status}`}>{STATUS_LABEL[p.status]}</span>
                {confirming === p.id ? (
                  <span className="confirm">
                    <button
                      type="button"
                      className="link danger"
                      onClick={() => {
                        setConfirming(null);
                        props.onDelete(p.id);
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
                    disabled={p.status === "processing"}
                    aria-label={`Delete ${label}`}
                    onClick={() => setConfirming(p.id)}
                  >
                    Delete
                  </button>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}
