import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, type Health } from "./api";
import { ConversationList } from "./components/ConversationList";
import { EvidenceAnswer } from "./components/EvidenceAnswer";
import { PaperPanel } from "./components/PaperPanel";
import { QueryBox } from "./components/QueryBox";
import { ResearchAnswer } from "./components/ResearchAnswer";
import type { ConversationSummary, Mode, Paper, QueryResponse } from "./types";

type ThreadItem =
  | { key: string; kind: "user"; text: string; mode: Mode; paperIds: string[] }
  | { key: string; kind: "answer"; response: QueryResponse }
  | { key: string; kind: "error"; text: string }
  | { key: string; kind: "pending" };

const POLL_MS = 2000;
const message = (e: unknown) => (e instanceof Error ? e.message : "Something went wrong");

function degradedReason(h: Health): string | null {
  if (h.status === "ok") return null;
  const problems = [
    !h.qdrant && "the vector database (Qdrant) is unreachable",
    !h.database && "the database (PostgreSQL) is unreachable",
    !h.llm_configured && "no LLM API key is configured, so questions cannot be answered",
  ].filter(Boolean);
  return `Backend is degraded: ${problems.join("; ")}.`;
}

export default function App() {
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [uploading, setUploading] = useState(false);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [thread, setThread] = useState<ThreadItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [health, setHealth] = useState<Health | null>(null);
  const counter = useRef(0);
  const bottom = useRef<HTMLDivElement>(null);
  const nextKey = () => `m${++counter.current}`;

  const refreshPapers = useCallback(async () => {
    try {
      const list = await api.listPapers();
      setPapers(list);
      // Drop selections of papers that no longer exist or are no longer ready.
      setSelected((prev) => {
        const ready = new Set(list.filter((p) => p.status === "ready").map((p) => p.id));
        const next = new Set([...prev].filter((id) => ready.has(id)));
        return next.size === prev.size ? prev : next;
      });
    } catch (e) {
      setNotice(message(e));
    }
  }, []);

  const refreshConversations = useCallback(async () => {
    try {
      setConversations(await api.listConversations());
    } catch (e) {
      setNotice(message(e));
    }
  }, []);

  useEffect(() => {
    void refreshPapers();
    void refreshConversations();
    api.health().then(setHealth, () => undefined);
  }, [refreshPapers, refreshConversations]);

  // Poll while any paper is still being indexed.
  const anyProcessing = papers.some((p) => p.status === "processing");
  useEffect(() => {
    if (!anyProcessing) return;
    const timer = setTimeout(() => void refreshPapers(), POLL_MS);
    return () => clearTimeout(timer);
  }, [anyProcessing, papers, refreshPapers]);

  useEffect(() => {
    bottom.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [thread.length]);

  async function upload(files: File[]) {
    setUploading(true);
    setNotice(null);
    try {
      const results = await api.uploadPapers(files);
      const dupes = results.filter((r) => r.duplicate).map((r) => r.paper.title ?? r.paper.filename);
      if (dupes.length) setNotice(`Already uploaded: ${dupes.join(", ")}.`);
      await refreshPapers();
    } catch (e) {
      setNotice(message(e));
    } finally {
      setUploading(false);
    }
  }

  async function deletePaper(id: string) {
    try {
      await api.deletePaper(id);
      await refreshPapers();
    } catch (e) {
      setNotice(message(e));
    }
  }

  function toggle(id: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (!next.delete(id)) next.add(id);
      return next;
    });
  }

  async function ask(question: string, mode: Mode): Promise<boolean> {
    const paperIds = [...selected];
    const pendingKey = nextKey();
    setNotice(null);
    setBusy(true);
    setThread((t) => [
      ...t,
      { key: nextKey(), kind: "user", text: question, mode, paperIds },
      { key: pendingKey, kind: "pending" },
    ]);
    try {
      const response = await api.query(question, paperIds, mode, conversationId);
      setThread((t) =>
        t.map((item) => (item.key === pendingKey ? { key: pendingKey, kind: "answer", response } : item)),
      );
      setConversationId(response.conversation_id);
      void refreshConversations();
      return true;
    } catch (e) {
      const text = e instanceof ApiError ? e.message : message(e);
      setThread((t) => t.map((item) => (item.key === pendingKey ? { key: pendingKey, kind: "error", text } : item)));
      return false;
    } finally {
      setBusy(false);
    }
  }

  async function openConversation(id: string) {
    try {
      const c = await api.getConversation(id);
      const items: ThreadItem[] = c.messages.flatMap<ThreadItem>((m) => {
        if (m.role === "user") {
          return [{ key: nextKey(), kind: "user", text: m.content, mode: m.mode ?? "auto", paperIds: m.paper_ids }];
        }
        return m.payload ? [{ key: nextKey(), kind: "answer", response: m.payload }] : [];
      });
      setThread(items);
      setConversationId(id);
      // Re-select the papers the last question was about (those still available).
      const last = [...c.messages].reverse().find((m) => m.role === "user");
      const ready = new Set(papers.filter((p) => p.status === "ready").map((p) => p.id));
      setSelected(new Set((last?.paper_ids ?? []).filter((pid) => ready.has(pid))));
      setNotice(null);
    } catch (e) {
      setNotice(message(e));
    }
  }

  function newConversation() {
    setThread([]);
    setConversationId(null);
  }

  async function deleteConversation(id: string) {
    try {
      await api.deleteConversation(id);
      if (id === conversationId) newConversation();
      await refreshConversations();
    } catch (e) {
      setNotice(message(e));
    }
  }

  const titleOf = (id: string) => {
    const p = papers.find((x) => x.id === id);
    return p ? (p.title ?? p.filename) : `paper ${id.slice(0, 6)}`;
  };
  const degraded = health ? degradedReason(health) : null;

  return (
    <div className="app">
      <header className="top">
        <h1>ResearchHelp</h1>
        <span className="muted">Ask questions about your research papers, with page citations</span>
      </header>

      <div className="banners">
        {degraded && (
          <div className="banner warn" role="status">
            {degraded}
          </div>
        )}
        {notice && (
          <div className="banner" role="alert">
            <span>{notice}</span>
            <button type="button" className="link" onClick={() => setNotice(null)}>
              Dismiss
            </button>
          </div>
        )}
      </div>

      <aside className="side">
        <PaperPanel
          papers={papers}
          selected={selected}
          uploading={uploading}
          onToggle={toggle}
          onSelectAllReady={() => setSelected(new Set(papers.filter((p) => p.status === "ready").map((p) => p.id)))}
          onClearSelection={() => setSelected(new Set())}
          onUpload={(files) => void upload(files)}
          onDelete={(id) => void deletePaper(id)}
        />
        <ConversationList
          conversations={conversations}
          activeId={conversationId}
          onSelect={(id) => void openConversation(id)}
          onNew={newConversation}
          onDelete={(id) => void deleteConversation(id)}
        />
      </aside>

      <main className="main">
        <div className="thread" aria-live="polite">
          {thread.length === 0 && (
            <div className="welcome">
              <h2>Start with your papers</h2>
              <ol>
                <li>Upload one or more PDFs and wait until they are <strong>Ready</strong>.</li>
                <li>Tick the papers a question is about (several for comparisons).</li>
                <li>
                  Ask, for example: <em>“Compare the datasets used by these papers”</em> or{" "}
                  <em>“What research gaps exist and how could they be addressed?”</em>
                </li>
              </ol>
              <p className="muted">
                Factual questions get cited answers. Research questions get three separate sections:
                evidence from the papers, analysis inferred from it, and proposed directions to test.
              </p>
            </div>
          )}
          {thread.map((item) => {
            switch (item.kind) {
              case "user":
                return (
                  <div key={item.key} className="msg-user">
                    <p>{item.text}</p>
                    <span className="muted small">
                      {item.paperIds.map(titleOf).join(" · ")}
                      {item.mode !== "auto" ? ` · ${item.mode} mode` : ""}
                    </span>
                  </div>
                );
              case "pending":
                return (
                  <div key={item.key} className="msg-pending" role="status">
                    Searching the papers and writing an answer…
                  </div>
                );
              case "error":
                return (
                  <div key={item.key} className="msg-error" role="alert">
                    {item.text}
                  </div>
                );
              case "answer":
                return item.response.intent === "research" && item.response.research ? (
                  <ResearchAnswer key={item.key} uid={item.key} response={item.response} />
                ) : item.response.evidence ? (
                  <EvidenceAnswer key={item.key} uid={item.key} response={item.response} />
                ) : null;
            }
          })}
          <div ref={bottom} />
        </div>
        <QueryBox selectedCount={selected.size} busy={busy} onAsk={ask} />
      </main>
    </div>
  );
}
