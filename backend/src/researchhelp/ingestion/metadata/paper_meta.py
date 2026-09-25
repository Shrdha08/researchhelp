"""Paper-level metadata: stable paper ID and title."""

import re

from researchhelp.ingestion.cleaning import normalize_text
from researchhelp.ingestion.loaders.pdf_loader import LoadedPdf

_BAD_TITLE = re.compile(r"(\.pdf|\.tex|\.docx?|untitled|microsoft word|^arxiv)", re.IGNORECASE)


def paper_id_from_sha256(sha256: str) -> str:
    """First 16 hex chars of the file hash: deterministic, deduplicates identical uploads,
    and links PostgreSQL rows to Qdrant points."""
    return sha256[:16]


def extract_title(pdf: LoadedPdf, fallback: str = "Untitled paper") -> str:
    meta_title = normalize_text(pdf.metadata.get("title") or "")
    if len(meta_title) >= 10 and not _BAD_TITLE.search(meta_title):
        return meta_title

    if not pdf.pages or not pdf.pages[0].lines:
        return fallback

    # Otherwise: the largest-font text near the top of page 1 (consecutive lines of that size).
    lines = pdf.pages[0].lines[:40]
    largest = max(line.size for line in lines)
    title_lines: list[str] = []
    for line in lines:
        if line.size >= largest - 0.5:
            title_lines.append(normalize_text(line.text))
        elif title_lines:
            break
    title = " ".join(title_lines).strip()
    return title if len(title) >= 5 else fallback
