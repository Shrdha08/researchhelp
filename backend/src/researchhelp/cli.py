"""Command-line driver (before the FastAPI backend exists).

researchhelp ingest data/papers/            # one or more PDFs or directories
researchhelp papers                         # list indexed papers
researchhelp ask "What datasets are used?" -p dpr -p "retrieval-augmented"
researchhelp ask "What gaps exist across these papers?" -p dpr -p rag --mode research
researchhelp inspect data/papers/dpr.pdf    # show parsed chunks (no indexing)
researchhelp delete <paper-id>
"""

import json
from pathlib import Path
from typing import Annotated

import typer

from researchhelp.config import get_settings

app = typer.Typer(add_completion=False, no_args_is_help=True, help="ResearchHelp CLI")


def _pdfs(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(path.glob("*.pdf")))
        elif path.suffix.lower() == ".pdf" and path.exists():
            found.append(path)
        else:
            raise typer.BadParameter(f"Not a PDF or directory: {path}")
    return found


def _resolve_papers(refs: list[str]) -> dict[str, str]:
    """Map user references (paper-id prefix or a piece of the title) to {paper_id: title}."""
    from researchhelp.retrieval.factory import get_store

    papers = get_store().list_papers()
    resolved: dict[str, str] = {}
    for ref in refs:
        matches = [
            p for p in papers if p.paper_id.startswith(ref) or ref.lower() in p.title.lower()
        ]
        if len(matches) != 1:
            names = ", ".join(f"{p.paper_id[:8]} ({p.title[:40]})" for p in matches) or "none"
            raise typer.BadParameter(f"'{ref}' must match exactly one paper; matched: {names}")
        resolved[matches[0].paper_id] = matches[0].title
    return resolved


@app.command()
def ingest(paths: Annotated[list[Path], typer.Argument(help="PDF files or directories")]):
    """Parse, chunk, embed and index PDFs."""
    from researchhelp.ingestion.pipeline import ingest_pdf
    from researchhelp.retrieval.factory import get_store

    settings = get_settings()
    store = get_store()
    for pdf in _pdfs(paths):
        paper = ingest_pdf(pdf, store, settings.chunk_size, settings.chunk_overlap)
        typer.echo(
            f"{paper.paper_id}  {len(paper.chunks):4d} chunks  {paper.num_pages:3d} pages  "
            f"{paper.title[:70]}"
        )


@app.command()
def papers():
    """List indexed papers."""
    from researchhelp.retrieval.factory import get_store

    for p in sorted(get_store().list_papers(), key=lambda p: p.title):
        typer.echo(f"{p.paper_id}  {p.num_chunks:4d} chunks  {p.num_pages:3d} pages  {p.title}")


@app.command()
def delete(paper_id: str):
    """Remove a paper's chunks from the index."""
    from researchhelp.retrieval.factory import get_store

    ((pid, title),) = _resolve_papers([paper_id]).items()
    get_store().delete_paper(pid)
    typer.echo(f"Deleted {pid} ({title})")


@app.command()
def inspect(
    pdf: Path,
    page: Annotated[int | None, typer.Option(help="Only show chunks from this page")] = None,
):
    """Show how a PDF is parsed and chunked, without indexing it."""
    from researchhelp.ingestion.pipeline import parse_pdf

    settings = get_settings()
    paper = parse_pdf(pdf, settings.chunk_size, settings.chunk_overlap)
    typer.echo(
        f"{paper.paper_id}  {paper.title}  ({paper.num_pages} pages, {len(paper.chunks)} chunks)"
    )
    typer.echo("Sections: " + " | ".join(paper.sections))
    for chunk in paper.chunks:
        if page is None or chunk.page == page:
            typer.echo(f"\n--- {chunk.chunk_id}  p{chunk.page}  [{chunk.section}]")
            typer.echo(chunk.text)


def _render_evidence(result, show_context: bool) -> None:
    if show_context:
        typer.echo("Retrieved context:")
        for i, doc in enumerate(result.sources, start=1):
            m = doc.metadata
            typer.echo(
                f"  [S{i}] {m['score']:.3f}  {m['paper_title'][:40]}  p{m['page']}  "
                f"[{m['section']}]  {doc.page_content[:90]!r}"
            )
        typer.echo("")
    typer.echo(result.answer)
    if result.citations:
        typer.echo("\nSources:")
        for c in result.citations:
            typer.echo(f"  [{c.source_id}] {c.paper_title} - page {c.page} ({c.section})")


def _render_research(result) -> None:
    if result.message:
        typer.echo(f"Note: {result.message}\n")
    if result.evidence:
        typer.echo("== Evidence (stated in the papers) ==")
        for e in result.evidence:
            pages = ", ".join(f"p.{c.page}" for c in e.citations)
            typer.echo(f"  {e.id}. {e.claim}")
            typer.echo(f"      [{e.paper_title} - {pages}]")
    if result.analysis:
        typer.echo("\n== Analysis (inferred from the evidence) ==")
        for i, a in enumerate(result.analysis, start=1):
            typer.echo(f"  A{i}. {a.statement}")
            typer.echo(f"      based on {', '.join(a.based_on)}; confidence {a.confidence}")
    if result.directions:
        typer.echo("\n== Proposed research directions (hypotheses, not established gaps) ==")
        for i, d in enumerate(result.directions, start=1):
            typer.echo(f"  D{i}. {d.title}")
            typer.echo(f"      Rationale: {d.rationale}")
            if d.validation_experiment:
                typer.echo(f"      How to test: {d.validation_experiment}")
            typer.echo(f"      based on {', '.join(d.based_on)}")
    if result.status == "unstructured" and result.raw_text:
        typer.echo("\n== Raw model output ==")
        typer.echo(result.raw_text)


@app.command()
def ask(
    question: str,
    paper: Annotated[
        list[str], typer.Option("--paper", "-p", help="Paper ID prefix or title fragment")
    ],
    mode: Annotated[str, typer.Option(help="auto (router decides) | evidence | research")] = "auto",
    strategy: Annotated[
        str | None, typer.Option(help="semantic | hybrid | hybrid_rerank (default: settings)")
    ] = None,
    show_context: Annotated[bool, typer.Option(help="Print the retrieved chunks")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full result as JSON")] = False,
):
    """Ask about the selected papers: evidence Q&A with citations, or research assistance."""
    from researchhelp.graph.factory import get_app
    from researchhelp.rag.common.llm import MissingAPIKeyError

    if mode not in ("auto", "evidence", "research"):
        raise typer.BadParameter("--mode must be auto, evidence or research")
    try:
        graph = get_app(strategy)
    except MissingAPIKeyError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1) from exc
    titles = _resolve_papers(paper)
    state = graph.invoke(
        {"question": question, "paper_ids": list(titles), "paper_titles": titles, "mode": mode}
    )
    result = state["result"]

    if as_json:
        route = {k: state.get(k) for k in ("intent", "route_reason", "route_source")}
        typer.echo(json.dumps({**route, "result": result.to_dict()}, indent=2, ensure_ascii=False))
        return
    typer.echo(f"[route: {state['intent']} ({state['route_source']}): {state['route_reason']}]\n")
    if state["intent"] == "research":
        _render_research(result)
    else:
        _render_evidence(result, show_context)


@app.command()
def migrate(
    revision: Annotated[str, typer.Argument(help="Target revision (default: latest)")] = "head",
):
    """Apply database migrations (the API also does this at startup)."""
    from researchhelp.db.migrate import upgrade

    upgrade(get_settings().database_url, revision)
    typer.echo(f"Database is at revision: {revision}")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface to bind")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port")] = 8000,
    reload: Annotated[bool, typer.Option(help="Auto-reload on code changes (dev)")] = False,
):
    """Run the HTTP API (Swagger UI at /docs)."""
    import uvicorn

    uvicorn.run("researchhelp.api.main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
