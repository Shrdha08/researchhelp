"""PDF loading with PyMuPDF.

We use ``page.get_text("dict")`` rather than plain text extraction because it gives us, per
line, the font size and weight (needed for heading detection).

Reading order: we deliberately keep the PDF's content-stream order (``sort=False``). On the
evaluation corpus (LaTeX two-column papers) ``sort=True`` interleaved left- and right-column
blocks by vertical position, e.g. placing "References" before "Conclusion" in FiD and
"3 DPR" before "2 Background" in DPR. LaTeX emits text column by column, so stream order is
the better reading order for this kind of document.
"""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

# PyMuPDF span flag bit for bold text.
_BOLD_FLAG = 1 << 4


@dataclass(frozen=True)
class Line:
    text: str
    size: float  # dominant font size of the line (points)
    bold: bool
    block: int  # index of the text block on the page; used to rebuild paragraphs


@dataclass
class PageContent:
    number: int  # 1-based page number, as shown in citations
    lines: list[Line] = field(default_factory=list)


@dataclass
class LoadedPdf:
    sha256: str
    metadata: dict
    pages: list[PageContent]

    @property
    def num_pages(self) -> int:
        return len(self.pages)


def _line_from_spans(spans: list[dict], block: int) -> Line | None:
    text = "".join(s["text"] for s in spans).strip()
    if not text:
        return None
    # Weight = that of the span covering the most characters.
    main = max(spans, key=lambda s: len(s["text"].strip()))
    bold = bool(main["flags"] & _BOLD_FLAG) or "bold" in main["font"].lower()
    # Size = largest span containing letters/digits. Small-caps headings ("I" at 12pt followed
    # by "NTRODUCTION" at 9.6pt) would otherwise look smaller than body text.
    wordy = [s["size"] for s in spans if any(ch.isalnum() for ch in s["text"])]
    size = max(wordy) if wordy else main["size"]
    return Line(text=text, size=round(size, 1), bold=bold, block=block)


def load_pdf_bytes(data: bytes) -> LoadedPdf:
    pages: list[PageContent] = []
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        metadata = dict(doc.metadata or {})
        for page_index, page in enumerate(doc):
            content = PageContent(number=page_index + 1)
            page_dict = page.get_text("dict", sort=False)
            for block_no, block in enumerate(page_dict["blocks"]):
                if block.get("type") != 0:  # 0 = text, 1 = image
                    continue
                for raw_line in block["lines"]:
                    # Skip rotated text, e.g. the vertical arXiv identifier in the margin.
                    dx, dy = raw_line["dir"]
                    if abs(dy) > 1e-3 or dx <= 0:
                        continue
                    line = _line_from_spans(raw_line["spans"], block_no)
                    if line:
                        content.lines.append(line)
            pages.append(content)
    return LoadedPdf(sha256=hashlib.sha256(data).hexdigest(), metadata=metadata, pages=pages)


def load_pdf(path: str | Path) -> LoadedPdf:
    return load_pdf_bytes(Path(path).read_bytes())
