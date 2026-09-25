"""Shared ingestion pipeline: PDF -> text -> cleaning -> sections -> chunks (-> vector store).

``parse_pdf`` is pure (no I/O besides reading the file) so parsing quality can be inspected and
unit-tested without Qdrant or embedding models.
"""

from dataclasses import dataclass
from pathlib import Path

from researchhelp.ingestion.chunking import Chunk, chunk_segments
from researchhelp.ingestion.cleaning import remove_boilerplate
from researchhelp.ingestion.loaders.pdf_loader import load_pdf_bytes
from researchhelp.ingestion.metadata.paper_meta import extract_title, paper_id_from_sha256
from researchhelp.ingestion.metadata.sections import segment_pages
from researchhelp.retrieval.vectorstore import PaperVectorStore


@dataclass
class ParsedPaper:
    paper_id: str
    title: str
    filename: str
    num_pages: int
    chunks: list[Chunk]

    @property
    def sections(self) -> list[str]:
        """Distinct sections in document order."""
        return list(dict.fromkeys(c.section for c in self.chunks))


def parse_pdf_bytes(
    data: bytes, filename: str, chunk_size: int = 1000, chunk_overlap: int = 150
) -> ParsedPaper:
    pdf = load_pdf_bytes(data)
    paper_id = paper_id_from_sha256(pdf.sha256)
    title = extract_title(pdf, fallback=Path(filename).stem)
    pages = remove_boilerplate(pdf.pages)
    segments = segment_pages(pages)
    chunks = chunk_segments(segments, paper_id, title, chunk_size, chunk_overlap)
    return ParsedPaper(paper_id, title, filename, pdf.num_pages, chunks)


def parse_pdf(path: str | Path, chunk_size: int = 1000, chunk_overlap: int = 150) -> ParsedPaper:
    path = Path(path)
    return parse_pdf_bytes(path.read_bytes(), path.name, chunk_size, chunk_overlap)


def ingest_pdf(
    path: str | Path,
    store: PaperVectorStore,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> ParsedPaper:
    """Parse a PDF and index its chunks. Existing points of the same paper are removed first,
    so re-ingesting after a chunking change leaves no stale chunks behind."""
    paper = parse_pdf(path, chunk_size, chunk_overlap)
    if not paper.chunks:
        raise ValueError(f"No extractable text in {paper.filename} (scanned PDF?)")
    store.ensure_collection()
    store.delete_paper(paper.paper_id)
    store.upsert_chunks(paper.chunks)
    return paper
