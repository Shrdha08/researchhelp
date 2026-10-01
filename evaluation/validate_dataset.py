"""Check every gold quote really appears on its labelled page (after our own PDF parsing), and
that each quote is fully contained in at least one chunk (otherwise no retriever could match it).

    uv run --project backend python -m evaluation.validate_dataset
"""

from collections import Counter

from evaluation.common import PAPERS_DIR, load_questions, norm
from researchhelp.config import get_settings
from researchhelp.ingestion.pipeline import parse_pdf


def main() -> None:
    settings = get_settings()
    questions = load_questions()
    chunks_by_paper = {}
    problems = 0
    for q in questions:
        if q.type != "unanswerable" and not q.evidence:
            print(f"[{q.id}] answerable question without evidence")
            problems += 1
        for ev in q.evidence:
            if ev.paper not in chunks_by_paper:
                parsed = parse_pdf(
                    PAPERS_DIR / f"{ev.paper}.pdf", settings.chunk_size, settings.chunk_overlap
                )
                chunks_by_paper[ev.paper] = parsed.chunks
            on_page = [c for c in chunks_by_paper[ev.paper] if c.page == ev.page]
            page_text = norm(" ".join(c.text for c in on_page))
            quote = norm(ev.quote)
            if quote not in page_text:
                where = sorted({c.page for c in chunks_by_paper[ev.paper] if quote in norm(c.text)})
                print(
                    f"[{q.id}] quote not on {ev.paper} p{ev.page} (found on pages {where}): "
                    f"{ev.quote!r}"
                )
                problems += 1
            elif not any(quote in norm(c.text) for c in on_page):
                print(
                    f"[{q.id}] quote spans a chunk boundary on {ev.paper} p{ev.page}: {ev.quote!r}"
                )
                problems += 1

    types = Counter(q.type for q in questions)
    splits = Counter(q.split for q in questions)
    print(
        f"\n{len(questions)} questions {dict(types)}; splits {dict(splits)}; "
        f"{sum(len(q.evidence) for q in questions)} evidence items; {problems} problems"
    )
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
