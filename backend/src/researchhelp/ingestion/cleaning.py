"""Text cleaning for extracted PDF lines.

Deliberately small: ligature/unicode normalisation, removal of running headers/footers and page
numbers, and paragraph reconstruction with de-hyphenation. No attempt is made to repair
equations or tables.
"""

import re
import unicodedata
from collections import Counter

from researchhelp.ingestion.loaders.pdf_loader import Line, PageContent

_PAGE_NUMBER = re.compile(r"^(page\s*)?\d{1,4}(\s*(/|of)\s*\d{1,4})?$", re.IGNORECASE)
_DIGITS = re.compile(r"\d+")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """NFKC turns ligatures (U+FB01 'fi' -> 'fi') and full-width forms into plain characters."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("­", "")  # soft hyphen
    return _WHITESPACE.sub(" ", text).strip()


def _line_key(text: str) -> str:
    # Digits are masked so "Page 3" and "Page 4" count as the same running header.
    return _DIGITS.sub("#", text.lower())


def _edge_indices(n_lines: int, edge: int) -> set[int]:
    return set(range(min(edge, n_lines))) | set(range(max(0, n_lines - edge), n_lines))


def remove_boilerplate(
    pages: list[PageContent], min_fraction: float = 0.5, edge: int = 2
) -> list[PageContent]:
    """Drop page numbers and running headers/footers (venue name, "Preprint. Under review.").

    Only the first/last ``edge`` lines of a page are candidates, and a candidate is removed when
    the same (digit-masked) line sits at a page edge on at least ``min_fraction`` of the pages.
    Restricting to page edges keeps body lines that merely look similar ("Table 1", "Table 2")."""
    counts: Counter[str] = Counter()
    for page in pages:
        edges = _edge_indices(len(page.lines), edge)
        counts.update({_line_key(page.lines[i].text) for i in edges})

    threshold = max(3, int(len(pages) * min_fraction + 0.999))
    repeated = {key for key, n in counts.items() if n >= threshold} if len(pages) >= 3 else set()

    cleaned = []
    for page in pages:
        edges = _edge_indices(len(page.lines), edge)
        kept = [
            line
            for i, line in enumerate(page.lines)
            if not (
                i in edges
                and (_PAGE_NUMBER.match(line.text.strip()) or _line_key(line.text) in repeated)
            )
        ]
        cleaned.append(PageContent(number=page.number, lines=kept))
    return cleaned


def join_lines(lines: list[Line]) -> str:
    """Rebuild text from lines: lines in the same block form a paragraph (joined with spaces,
    words hyphenated across a line break are re-joined); blocks are separated by blank lines."""
    parts: list[str] = []
    prev: Line | None = None
    for line in lines:
        text = normalize_text(line.text)
        if not text:
            continue
        if prev is None:
            parts.append(text)
        elif parts[-1].endswith("-") and text[:1].islower():
            # Word split across a line break; PyMuPDF sometimes also starts a new block here.
            parts[-1] = parts[-1][:-1]
            parts.append(text)
        elif line.block != prev.block:
            parts.append("\n\n" + text)
        else:
            parts.append(" " + text)
        prev = line
    return "".join(parts).strip()
