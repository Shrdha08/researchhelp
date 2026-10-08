import { vi } from "vitest";

type Handler = (init: RequestInit | undefined) => { status?: number; body?: unknown };

/** Installs a fake `fetch` that serves routes keyed "METHOD /path" (the "/api" prefix removed).
 *  Unlisted routes return 404. Returns the call log for assertions. */
export function fakeBackend(routes: Record<string, Handler>) {
  const calls: { key: string; init?: RequestInit }[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      const key = `${init?.method ?? "GET"} ${url.replace(/^\/api/, "")}`;
      calls.push({ key, init });
      const handler = routes[key];
      if (!handler) return new Response(JSON.stringify({ detail: `no route ${key}` }), { status: 404 });
      const { status = 200, body } = handler(init);
      return status === 204
        ? new Response(null, { status })
        : new Response(JSON.stringify(body ?? {}), { status, headers: { "Content-Type": "application/json" } });
    }),
  );
  return calls;
}

export const okHealth = { status: "ok", qdrant: true, database: true, papers: 1, llm_configured: true };
