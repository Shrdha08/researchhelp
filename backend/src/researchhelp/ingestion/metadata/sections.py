"""Heuristic section detection.

A line is treated as a heading when BOTH hold:
  1. its text looks like a heading: a numbered heading ("3.1 Training Details", "IV. RESULTS",
     "A Hyperparameters"), or one of a few section names that papers commonly leave unnumbered
     ("Abstract", "References", "Acknowledgements", ...), and
  2. it is typographically distinct: larger than the body font or bold.

Unnumbered generic words such as "Model" or "Results" are NOT accepted on their own, because
bold table headers use exactly those words.

This is intentionally not a full academic-PDF parser (e.g. GROBID). It labels most chunks with
a useful section name and otherwise falls back to the previous section.
"""

import re
from collections import Counter
from dataclasses import dataclass, replace

from researchhelp.ingestion.cleaning import join_lines, normalize_text
from researchhelp.ingestion.loaders.pdf_loader import Line, PageContent

FRONT_MATTER = "Front Matter"
REFERENCES = "References"

# Canonical names used when a heading title matches (numbered or not).
CANONICAL = {
    "abstract": "Abstract",
    "introduction": "Introduction",
    "background": "Background",
    "related work": "Related Work",
    "related works": "Related Work",
    "preliminaries": "Preliminaries",
    "method": "Method",
    "methods": "Method",
    "methodology": "Method",
    "approach": "Method",
    "experiments": "Experiments",
    "experimental setup": "Experimental Setup",
    "experimental results": "Results",
    "results": "Results",
    "evaluation": "Evaluation",
    "analysis": "Analysis",
    "discussion": "Discussion",
    "limitations": "Limitations",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "future work": "Future Work",
    "acknowledgements": "Acknowledgements",
    "acknowledgments": "Acknowledgements",
    "acknowledgement": "Acknowledgements",
    "acknowledgment": "Acknowledgements",
    "references": REFERENCES,
    "bibliography": REFERENCES,
    "appendix": "Appendix",
}
# Names accepted as headings even without a section number.
UNNUMBERED_OK = {
    "abstract",
    "introduction",
    "related work",
    "limitations",
    "conclusion",
    "conclusions",
    "acknowledgements",
    "acknowledgments",
    "acknowledgement",
    "acknowledgment",
    "references",
    "bibliography",
    "appendix",
}

# "3", "3.1", "3.1.2", "IV", "A", "A.1" followed by an optional dot and a capitalised title.
_NUMBERED = re.compile(
    r"^(?P<num>(\d{1,2}(\.\d{1,2}){0,3})|([IVX]{1,5})|([A-H](\.\d{1,2}){0,2}))\.?\s+"
    r"(?P<title>[A-Z][^\n]{1,80})$"
)
_BARE_NUMBER = re.compile(r"^(\d{1,2}(\.\d{1,2}){0,3}|[IVX]{1,5}|[A-H](\.\d{1,2}){0,2})\.?$")
_MAX_HEADING_WORDS = 12


@dataclass(frozen=True)
class Segment:
    """Contiguous text on one page belonging to one section."""

    page: int
    section: str
    text: str


def body_font_size(pages: list[PageContent]) -> float:
    """Most common font size, weighted by number of characters."""
    sizes: Counter[float] = Counter()
    for page in pages:
        for line in page.lines:
            sizes[line.size] += len(line.text)
    return sizes.most_common(1)[0][0] if sizes else 10.0


def _is_distinct(line: Line, body_size: float) -> bool:
    return line.bold or line.size >= body_size + 0.5


def _pretty(title: str) -> str:
    canonical = CANONICAL.get(title.lower())
    if canonical:
        return canonical
    return title.title() if title.isupper() else title


def heading_title(line: Line, body_size: float, in_front_matter: bool = False) -> str | None:
    """Return the normalised section title if ``line`` is a heading, else None."""
    text = normalize_text(line.text).rstrip(":")
    if not text or len(text.split()) > _MAX_HEADING_WORDS:
        return None
    if not _is_distinct(line, body_size):
        return None

    if text.lower() in UNNUMBERED_OK:
        return CANONICAL[text.lower()]

    match = _NUMBERED.match(text)
    # Trailing sentence punctuation means a bold sentence, not a heading.
    if not match or text.endswith((".", ",", ";")):
        return None
    title = match.group("title").strip().rstrip(":")
    if not _looks_like_title(title):
        return None
    # In the front matter, superscript affiliation markers look like numbers ("1 Inria, Paris").
    if in_front_matter and ("," in title or "@" in title):
        return None
    return _pretty(title)


def _looks_like_title(title: str) -> bool:
    """Reject bold table cells and formula fragments that happen to follow a number
    ("7B Yes", "1 Jurassic-1 (178B)", "N + 1[", "DPR (Karpukhin et al., 2020)")."""
    chars = [c for c in title if not c.isspace()]
    letters = sum(c.isalpha() for c in chars)
    return letters >= 4 and letters / len(chars) >= 0.75 and "et al" not in title


def merge_split_numbers(lines: list[Line], body_size: float) -> list[Line]:
    """Some templates put the section number on its own line ("1" / "INTRODUCTION").
    Merge a bare, typographically distinct number with the distinct line that follows it."""
    merged: list[Line] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else None
        if (
            nxt is not None
            and _BARE_NUMBER.match(line.text.strip())
            and _is_distinct(line, body_size)
            and _is_distinct(nxt, body_size)
        ):
            text = f"{line.text.strip().rstrip('.')} {nxt.text.strip()}"
            merged.append(replace(nxt, text=text, size=max(line.size, nxt.size)))
            i += 2
            continue
        merged.append(line)
        i += 1
    return merged


def segment_pages(pages: list[PageContent]) -> list[Segment]:
    """Split each page into (section, text) segments. The current section carries over page
    breaks, so a page that starts mid-section is labelled correctly."""
    body_size = body_font_size(pages)
    segments: list[Segment] = []
    section = FRONT_MATTER

    for page in pages:
        current: list[Line] = []
        for line in merge_split_numbers(page.lines, body_size):
            title = heading_title(line, body_size, in_front_matter=section == FRONT_MATTER)
            if title and title != section:
                if current:
                    segments.append(Segment(page.number, section, join_lines(current)))
                section = title
                current = []
            current.append(line)
        if current:
            segments.append(Segment(page.number, section, join_lines(current)))

    return [s for s in segments if s.text]
