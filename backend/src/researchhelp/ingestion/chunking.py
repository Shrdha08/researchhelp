"""Chunking: recursive character splitting applied per (page, section) segment.

Because segments never span pages, no chunk spans pages, so every chunk has exactly one page
number and citations are exact. Chunks also never straddle a detected section boundary, so the
section label is accurate for the whole chunk.
"""

import uuid
from dataclasses import asdict, dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from researchhelp.ingestion.metadata.sections import Segment

# Namespace for deterministic point IDs (UUIDv5 of chunk_id).
_CHUNK_NAMESPACE = uuid.UUID("6f1c1d2e-7a0b-4c55-9a53-2b8f3e1d9c10")
MIN_CHUNK_CHARS = 30


@dataclass(frozen=True)
class Chunk:
    text: str
    paper_id: str
    paper_title: str
    page: int
    section: str
    chunk_index: int  # position of the chunk within the paper
    chunk_id: str  # human-readable stable ID, e.g. "3f2a...:p5:c2"

    @property
    def point_id(self) -> str:
        return str(uuid.uuid5(_CHUNK_NAMESPACE, self.chunk_id))

    def payload(self) -> dict:
        return asdict(self)


def make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        keep_separator="end",
    )


def chunk_segments(
    segments: list[Segment],
    paper_id: str,
    paper_title: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[Chunk]:
    splitter = make_splitter(chunk_size, chunk_overlap)
    chunks: list[Chunk] = []
    per_page_index: dict[int, int] = {}

    for segment in segments:
        for piece in splitter.split_text(segment.text):
            text = piece.strip()
            if len(text) < MIN_CHUNK_CHARS:
                continue
            idx_on_page = per_page_index.get(segment.page, 0)
            per_page_index[segment.page] = idx_on_page + 1
            chunks.append(
                Chunk(
                    text=text,
                    paper_id=paper_id,
                    paper_title=paper_title,
                    page=segment.page,
                    section=segment.section,
                    chunk_index=len(chunks),
                    chunk_id=f"{paper_id}:p{segment.page}:c{idx_on_page}",
                )
            )
    return chunks
