# ResearchHelp frontend

React 19 + TypeScript + Vite. Plain `fetch`, no state-management library: the app state (papers,
selection, conversations, the current thread) is small enough for `useState` in `App.tsx`.

```bash
npm install
npm run dev        # http://localhost:5173   (needs the backend: `uv run researchhelp serve`)
npm test           # Vitest + Testing Library (28 tests)
npm run build      # type-check + production bundle in dist/
```

The browser only ever calls `/api/...`. In development Vite proxies that to the FastAPI server on
`127.0.0.1:8000` and strips the prefix (override with `VITE_BACKEND_URL`); in production nginx does
the same (Phase 7), so the app needs no backend address and has no CORS configuration.

## Structure

| File | Role |
|---|---|
| `src/types.ts` | TypeScript mirror of the backend's response schemas |
| `src/api.ts` | The only place that calls `fetch`; turns FastAPI errors into readable messages |
| `src/App.tsx` | State, polling while papers are indexing, the ask / open-conversation flows |
| `components/PaperPanel` | Upload (button or drop), status badges, selection (ready papers only), delete |
| `components/QueryBox` | Question, Auto / Evidence / Research mode, why-disabled hint |
| `components/EvidenceAnswer` | Markdown answer with clickable `[S#]` chips that open the cited passage |
| `components/ResearchAnswer` | Evidence / Analysis / Proposed directions as three separate sections |
| `components/SourcesPanel` | The retrieved passages, numbered as the model saw them |
| `components/ConversationList` | Saved conversations: reopen (restores the paper selection), delete |

## Tests

`src/test/fakeBackend.ts` stubs `fetch` so the user journeys in `App.test.tsx` run without a
server. `src/test/real/*.json` are **real responses captured from the running backend**;
`realResponses.test.tsx` renders them, so a schema change in the backend that is not reflected in
`types.ts` fails a test. To refresh them, run a conversation against a live server and save the
messages' `payload` fields (see the capture steps in the test's header comment).
