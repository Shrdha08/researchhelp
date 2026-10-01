"""Context formatting and citation resolution shared by all RAG pipelines.

The LLM never writes page numbers. Each retrieved chunk is shown to it as ``[S1]``, ``[S2]``...
together with its paper/page/section, and the model cites those IDs. We then map the IDs back to
chunk metadata. IDs that were not in the context are removed, so a citation can never point to
a page the model did not actually see.
"""

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from langchain_core.documents import Document

# "[S1]" / "[S1, S3]". Some models (e.g. gpt-oss) emit full-width brackets "【S1】"; both are
# accepted and normalised to square brackets.
_MARKER = re.compile(r"[\[【]\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*[\]】]")
SNIPPET_CHARS = 300


@dataclass(frozen=True)
class Citation:
    source_id: str
    paper_id: str
    paper_title: str
    page: int
    section: str
    chunk_id: str
    snippet: str

    def to_dict(self) -> dict:
        return asdict(self)


def source_header(doc: Document) -> str:
    m = doc.metadata
    return f"(Paper: {m['paper_title']} | Page {m['page']} | Section: {m['section']})"


def format_context(docs: Sequence[Document]) -> tuple[str, dict[str, Document]]:
    """Return the prompt context and the mapping source_id -> Document."""
    sources = {f"S{i}": doc for i, doc in enumerate(docs, start=1)}
    blocks = [f"[{sid}] {source_header(doc)}\n{doc.page_content}" for sid, doc in sources.items()]
    return "\n\n---\n\n".join(blocks), sources


def format_paper_list(paper_titles: dict[str, str]) -> str:
    return "\n".join(f"- {title}" for title in paper_titles.values())


def _citation(source_id: str, doc: Document) -> Citation:
    m = doc.metadata
    text = doc.page_content
    snippet = (
        text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS].rsplit(" ", 1)[0] + " ..."
    )
    return Citation(
        source_id=source_id,
        paper_id=m["paper_id"],
        paper_title=m["paper_title"],
        page=m["page"],
        section=m["section"],
        chunk_id=m["chunk_id"],
        snippet=snippet,
    )


def resolve_citations(answer: str, sources: dict[str, Document]) -> tuple[str, list[Citation]]:
    """Validate ``[S#]`` markers in ``answer``.

    Returns the answer with unknown IDs removed and the citations in order of first use."""
    used: list[str] = []

    def replace(match: re.Match) -> str:
        ids = [s.strip() for s in re.split(r"[,;]", match.group(1))]
        valid = [s for s in ids if s in sources]
        for s in valid:
            if s not in used:
                used.append(s)
        return f"[{', '.join(valid)}]" if valid else ""

    cleaned = _MARKER.sub(replace, answer)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)  # tidy spaces left by removed markers
    return cleaned.strip(), [_citation(s, sources[s]) for s in used]
